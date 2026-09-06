"""Evaluation-only direct-Deux counterfactual cascade."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.io.atomic import atomic_output_path
from upmixer.eval.origins import (
    OriginViewOutput,
    _ACCOUNTING_FIELDS,
    _peak_memory_bytes,
    _roformer_accounting,
)
from upmixer.eval.reference_targets import validate_audio as _validate_audio
from upmixer.eval.types import RunSettings

SeparateFn = Callable[[str], tuple[dict[str, np.ndarray], RunSettings]]

CASCADE_RECIPE = "q40-deux-counterfactual-half-v1"
CASCADE_STEM_NAMES = frozenset({"Vocals", "_deux_inst"})
_CONSERVATION_TOLERANCE = 1e-6


@dataclass
class CascadeArmOutput:
    """One retained Q40 arm, including derived complementary controls."""

    arm_id: str
    role: str
    input_label: str
    stems: dict[str, np.ndarray]
    input_frame_count: int
    raw_output_frame_count: int
    runtime_s: float
    inference: bool = False
    output_paths: dict[str, str] | None = None


@dataclass
class CascadeEvaluationResult:
    """Fixed-half candidate plus all Q40 arms used to produce it."""

    stems: dict[str, np.ndarray]
    settings: RunSettings
    arms: tuple[CascadeArmOutput, ...]
    runtime_s: float
    peak_memory_bytes: int | None
    source_sample_rate: int
    source_frame_count: int

    @property
    def arm_outputs(self) -> tuple[CascadeArmOutput, ...]:
        """Compatibility name matching the Q30 per-view result."""
        return self.arms

    def provenance(
        self,
        *,
        recording_id: str | None = None,
        item_id: str | None = None,
        split: str | None = None,
        category: str | None = None,
    ) -> dict[str, object]:
        """Return JSON-safe arm, accounting, and runtime provenance."""
        arm_rows: list[dict[str, object]] = []
        reasons: list[str] = []
        for arm in self.arms:
            accounting: dict[str, object] = {
                field: None for field in _ACCOUNTING_FIELDS
            }
            if arm.inference:
                view = OriginViewOutput(
                    origin_samples=0,
                    stems=arm.stems,
                    input_frame_count=arm.input_frame_count,
                    raw_output_frame_count=arm.raw_output_frame_count,
                    runtime_s=arm.runtime_s,
                    input_sample_rate=self.source_sample_rate,
                )
                accounting = _roformer_accounting(self.settings, view)
                reason = accounting.get("unsupported_reason")
                if isinstance(reason, str):
                    reasons.append(reason)
            arm_rows.append(
                {
                    "arm_id": arm.arm_id,
                    "role": arm.role,
                    "input": arm.input_label,
                    "inference": arm.inference,
                    "input_frame_count": arm.input_frame_count,
                    "raw_output_frame_count": arm.raw_output_frame_count,
                    "aligned_frame_count": next(iter(arm.stems.values())).shape[0],
                    "stem_names": sorted(arm.stems),
                    "runtime_s": float(arm.runtime_s),
                    "output_paths": arm.output_paths or {},
                    **accounting,
                }
            )
        if reasons:
            totals: dict[str, object] = {
                field: None for field in _ACCOUNTING_FIELDS
            }
            totals["unsupported_reason"] = "; ".join(dict.fromkeys(reasons))
        else:
            totals = {
                field: sum(
                    int(row[field])
                    for row in arm_rows
                    if isinstance(row[field], int)
                )
                for field in _ACCOUNTING_FIELDS
            }
        payload: dict[str, object] = {
            "recipe": CASCADE_RECIPE,
            "source_sample_rate": self.source_sample_rate,
            "source_frame_count": self.source_frame_count,
            "stem_names": sorted(CASCADE_STEM_NAMES),
            "separation_call_count": sum(arm.inference for arm in self.arms),
            "runtime_s": float(self.runtime_s),
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_scope": "parent_process_lifetime_peak_rss",
            "conservation_tolerance": _CONSERVATION_TOLERANCE,
            "arms": arm_rows,
            "totals": totals,
        }
        if recording_id is not None:
            payload.update(
                recording_id=recording_id,
                item_id=item_id,
                split=split,
                category=category,
            )
        return payload


def write_cascade_arms(
    retained_arms: list[tuple[CascadeArmOutput, dict[str, np.ndarray]]],
    *,
    item_dir: Path,
    output_dir: Path,
    sample_rate: int,
    filename_for: Callable[[str, set[str]], str],
) -> list[dict[str, object]]:
    """Write validated arms under stable IDs and return index entries."""
    entries: list[dict[str, object]] = []
    for arm, arm_stems in retained_arms:
        arm_dir = item_dir / "arms" / arm.arm_id
        arm_dir.mkdir(parents=True, exist_ok=False)
        paths: dict[str, str] = {}
        used_filenames: set[str] = set()
        for stem_name in sorted(arm_stems, key=str):
            destination = arm_dir / filename_for(stem_name, used_filenames)
            with atomic_output_path(destination) as temporary:
                sf.write(
                    str(temporary), arm_stems[stem_name], sample_rate, subtype="FLOAT"
                )
            paths[stem_name] = destination.relative_to(output_dir).as_posix()
        arm.output_paths = paths
        entries.append(
            {
                "arm_id": arm.arm_id,
                "role": arm.role,
                "input": arm.input_label,
                "stems": paths,
            }
        )
    return entries


def _validated_stems(
    stems: object,
    *,
    frame_count: int,
    label: str,
) -> dict[str, np.ndarray]:
    if not isinstance(stems, dict) or set(stems) != CASCADE_STEM_NAMES:
        names = sorted(stems) if isinstance(stems, dict) else stems
        raise ValueError(
            f"{label} must return exactly {sorted(CASCADE_STEM_NAMES)}, got {names}"
        )
    checked: dict[str, np.ndarray] = {}
    for name in sorted(CASCADE_STEM_NAMES):
        audio = stems[name]
        if not isinstance(audio, np.ndarray) or audio.ndim != 2:
            raise ValueError(f"{label} stem {name} must be a 2D array")
        if audio.shape[1] != 2:
            raise ValueError(f"{label} stem {name} must be linked stereo")
        if audio.shape != (frame_count, 2):
            raise ValueError(
                f"{label} stem {name} shape {audio.shape} does not match "
                f"({frame_count}, 2)"
            )
        _validate_audio(audio, f"{label} stem {name}")
        with np.errstate(over="ignore", invalid="ignore"):
            checked_audio = np.asarray(audio, dtype=np.float32).copy()
        _validate_audio(checked_audio, f"{label} stem {name}")
        checked[name] = checked_audio
    return checked


def _validate_conservation(
    source: np.ndarray,
    vocals: np.ndarray,
    instrumental: np.ndarray,
    label: str,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(np.add(vocals, instrumental, dtype=np.float32) - source)
    max_error = float(np.max(error))
    if not np.isfinite(max_error) or max_error > _CONSERVATION_TOLERANCE:
        raise ValueError(
            f"{label} violates float32 source conservation: "
            f"max abs error {max_error} > {_CONSERVATION_TOLERANCE}"
        )


def _derived_arm(
    arm_id: str,
    role: str,
    input_label: str,
    vocals: np.ndarray,
    source: np.ndarray,
) -> CascadeArmOutput:
    started = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        instrumental = np.subtract(source, vocals, dtype=np.float32)
    stems = {"Vocals": vocals.copy(), "_deux_inst": instrumental}
    stems = _validated_stems(
        stems,
        frame_count=source.shape[0],
        label=f"{arm_id} arm",
    )
    _validate_conservation(
        source, stems["Vocals"], stems["_deux_inst"], f"{arm_id} arm"
    )
    return CascadeArmOutput(
        arm_id=arm_id,
        role=role,
        input_label=input_label,
        stems=stems,
        input_frame_count=source.shape[0],
        raw_output_frame_count=source.shape[0],
        runtime_s=time.perf_counter() - started,
    )


def separate_with_deux_cascade(
    mixture_path: str,
    separate_fn: SeparateFn,
) -> CascadeEvaluationResult:
    """Run Deux on ``X`` and ``X - 0.5 * I0`` for the fixed Q40 recipe."""
    source, source_rate = sf.read(
        mixture_path, dtype="float32", always_2d=True
    )
    source = np.asarray(source, dtype=np.float32).copy()
    _validate_audio(source, f"mixture {mixture_path}")
    if source.shape[1] != 2:
        raise ValueError("cascade mixture must be linked stereo")
    source_frames = source.shape[0]
    total_started = time.perf_counter()

    def run(path: str, label: str):
        started = time.perf_counter()
        result = separate_fn(path)
        elapsed = time.perf_counter() - started
        if (
            not isinstance(result, tuple)
            or len(result) != 2
        ):
            raise ValueError(f"{label} separation returned invalid result")
        stems, settings = result
        if not isinstance(settings, RunSettings):
            raise ValueError(f"{label} separation returned invalid RunSettings")
        if settings.sample_rate != source_rate:
            raise ValueError(
                f"{label} separation sample rate {settings.sample_rate} does "
                f"not match mixture sample rate {source_rate}"
            )
        return (
            _validated_stems(
                stems,
                frame_count=source_frames,
                label=label,
            ),
            settings,
            elapsed,
        )

    baseline, settings, baseline_elapsed = run(mixture_path, "independent")
    with np.errstate(over="ignore", invalid="ignore"):
        counterfactual_input = np.subtract(
            source,
            np.multiply(baseline["_deux_inst"], np.float32(0.5), dtype=np.float32),
            dtype=np.float32,
        )
    _validate_audio(counterfactual_input, "counterfactual input")
    with TemporaryDirectory(prefix="upmixer_eval_cascade_") as work_dir:
        counterfactual_path = Path(work_dir) / "counterfactual.wav"
        sf.write(
            str(counterfactual_path),
            counterfactual_input,
            source_rate,
            subtype="FLOAT",
        )
        refined, refined_settings, refined_elapsed = run(
            str(counterfactual_path), "counterfactual"
        )

    if refined_settings != settings:
        raise ValueError("cascade separation calls require identical RunSettings")

    independent = CascadeArmOutput(
        arm_id="independent-deux",
        role="co-reported incumbent",
        input_label="X",
        stems=baseline,
        input_frame_count=source_frames,
        raw_output_frame_count=source_frames,
        runtime_s=baseline_elapsed,
        inference=True,
    )
    complementary = _derived_arm(
        "complementary-v0",
        "candidate control",
        "X",
        baseline["Vocals"],
        source,
    )
    counterfactual = CascadeArmOutput(
        arm_id="counterfactual-v1",
        role="counterfactual heads",
        input_label="X-0.5*I0",
        stems=refined,
        input_frame_count=source_frames,
        raw_output_frame_count=source_frames,
        runtime_s=refined_elapsed,
        inference=True,
    )
    refined_complement = _derived_arm(
        "refined-complement",
        "counterfactual complementary arm",
        "X-0.5*I0",
        refined["Vocals"],
        source,
    )
    started = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_vocals = np.add(
            np.multiply(baseline["Vocals"], np.float32(0.5), dtype=np.float32),
            np.multiply(refined["Vocals"], np.float32(0.5), dtype=np.float32),
            dtype=np.float32,
        )
    candidate = _derived_arm(
        "fixed-half-recipe",
        "scored candidate",
        "X and X-0.5*I0",
        candidate_vocals,
        source,
    )
    candidate.runtime_s = time.perf_counter() - started
    arms = (
        independent,
        complementary,
        counterfactual,
        refined_complement,
        candidate,
    )
    return CascadeEvaluationResult(
        stems={name: audio.copy() for name, audio in candidate.stems.items()},
        settings=settings,
        arms=arms,
        runtime_s=time.perf_counter() - total_started,
        peak_memory_bytes=_peak_memory_bytes(),
        source_sample_rate=source_rate,
        source_frame_count=source_frames,
    )
