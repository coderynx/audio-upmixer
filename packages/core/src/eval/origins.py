"""Evaluation-only distinct time-origin views for Q30 experiments."""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.eval.reference_targets import validate_audio as _validate_audio
from upmixer.eval.types import RunSettings


SeparateFn = Callable[[str], tuple[dict[str, np.ndarray], "RunSettings"]]


@dataclass
class OriginViewOutput:
    """One retained, aligned output from an evaluation origin view."""

    origin_samples: int
    stems: dict[str, np.ndarray]
    input_frame_count: int
    raw_output_frame_count: int
    runtime_s: float
    output_paths: dict[str, str] | None = None


@dataclass
class OriginEvaluationResult:
    """Fused output plus individual views used by a Q30 evaluation."""

    stems: dict[str, np.ndarray]
    settings: "RunSettings"
    view_outputs: tuple[OriginViewOutput, ...]
    runtime_s: float
    peak_memory_bytes: int | None

    def provenance(
        self,
        *,
        recording_id: str | None = None,
        item_id: str | None = None,
        split: str | None = None,
        category: str | None = None,
    ) -> dict[str, object]:
        """Return JSON-safe run and per-view provenance without audio arrays."""
        payload: dict[str, object] = {
            "origin_schedule": list(self.settings.origin_schedule or ()),
            "effective_view_count": len(self.view_outputs),
            "separation_call_count": len(self.view_outputs),
            "runtime_s": float(self.runtime_s),
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_scope": "parent_process_rusage",
            "views": [
                {
                    "origin_samples": view.origin_samples,
                    "input_frame_count": view.input_frame_count,
                    "raw_output_frame_count": view.raw_output_frame_count,
                    "aligned_frame_count": next(iter(view.stems.values())).shape[0]
                    if view.stems
                    else 0,
                    "stem_names": sorted(view.stems),
                    "runtime_s": float(view.runtime_s),
                    "output_paths": view.output_paths or {},
                }
                for view in self.view_outputs
            ],
        }
        if recording_id is not None:
            payload.update(
                recording_id=recording_id,
                item_id=item_id,
                split=split,
                category=category,
            )
        return payload


def _peak_memory_bytes() -> int | None:
    """Return parent-process peak RSS where the host exposes it."""
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if platform.system() != "Darwin":
        peak *= 1024
    return peak


def _origin_stems(
    stems: object,
    *,
    frame_count: int,
    channels: int,
    label: str,
) -> dict[str, np.ndarray]:
    if not isinstance(stems, dict) or not stems:
        raise ValueError(f"{label} returned no stems")
    checked: dict[str, np.ndarray] = {}
    for name, audio in stems.items():
        if not isinstance(name, str):
            raise ValueError(f"{label} returned a non-string stem name")
        _validate_audio(audio, f"{label} stem {name}")
        if audio.shape != (frame_count, channels):
            raise ValueError(
                f"{label} stem {name} has shape {audio.shape}, expected "
                f"({frame_count}, {channels})"
            )
        audio = np.asarray(audio, dtype=np.float32)
        _validate_audio(audio, f"{label} stem {name}")
        checked[name] = audio
    return checked


def separate_with_extra_origin(
    mixture_path: str,
    separate_fn: SeparateFn,
    *,
    origin_samples: int,
) -> OriginEvaluationResult:
    """Evaluate one extra zero-padded origin and fuse linked stereo views.

    ``origin_samples`` is measured at the mixture sample rate.  The separator
    must return that same rate so inverse alignment is an exact integer slice.
    The output is an equal-weight waveform average of origins 0 and N.
    """
    if isinstance(origin_samples, bool) or not isinstance(origin_samples, int):
        raise TypeError("origin_samples must be an integer")
    if origin_samples < 1:
        raise ValueError("origin_samples must be at least 1")

    source, source_rate = sf.read(mixture_path, dtype="float32", always_2d=True)
    _validate_audio(source, f"mixture {mixture_path}")
    source_frames, channels = source.shape
    total_started = time.perf_counter()

    started = time.perf_counter()
    baseline_stems, baseline_settings = separate_fn(mixture_path)
    if not isinstance(baseline_settings, RunSettings):
        raise ValueError("separation returned invalid RunSettings")
    if baseline_settings.sample_rate != source_rate:
        raise ValueError(
            "extra-origin evaluation requires separator and mixture sample rates "
            f"to match ({baseline_settings.sample_rate} != {source_rate})"
        )
    baseline = _origin_stems(
        baseline_stems,
        frame_count=source_frames,
        channels=channels,
        label="baseline origin",
    )
    baseline_view = OriginViewOutput(
        origin_samples=0,
        stems=baseline,
        input_frame_count=source_frames,
        raw_output_frame_count=source_frames,
        runtime_s=time.perf_counter() - started,
    )

    padded = np.pad(
        source,
        ((origin_samples, origin_samples), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    with TemporaryDirectory(prefix="upmixer_eval_origin_") as view_dir:
        view_path = Path(view_dir) / f"origin_{origin_samples}.wav"
        sf.write(str(view_path), padded, source_rate, subtype="FLOAT")
        started = time.perf_counter()
        extra_stems, extra_settings = separate_fn(str(view_path))
        extra_elapsed = time.perf_counter() - started

    if not isinstance(extra_settings, RunSettings):
        raise ValueError("extra-origin separation returned invalid RunSettings")
    if extra_settings.sample_rate != source_rate:
        raise ValueError(
            "extra-origin separator sample rate does not match the mixture "
            f"({extra_settings.sample_rate} != {source_rate})"
        )
    extra_raw = _origin_stems(
        extra_stems,
        frame_count=source_frames + 2 * origin_samples,
        channels=channels,
        label="extra origin",
    )
    if set(extra_raw) != set(baseline):
        raise ValueError("baseline and extra-origin outputs have different stems")
    extra = {
        name: audio[origin_samples : origin_samples + source_frames]
        for name, audio in extra_raw.items()
    }
    extra_view = OriginViewOutput(
        origin_samples=origin_samples,
        stems=extra,
        input_frame_count=source_frames + 2 * origin_samples,
        raw_output_frame_count=source_frames + 2 * origin_samples,
        runtime_s=extra_elapsed,
    )
    settings = replace(
        baseline_settings,
        origin_schedule=(0, origin_samples),
    )
    fused = {
        name: ((baseline[name] + extra[name]) * 0.5).astype(np.float32)
        for name in sorted(baseline)
    }
    return OriginEvaluationResult(
        stems=fused,
        settings=settings,
        view_outputs=(baseline_view, extra_view),
        runtime_s=time.perf_counter() - total_started,
        peak_memory_bytes=_peak_memory_bytes(),
    )
