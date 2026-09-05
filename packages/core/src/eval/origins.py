"""Evaluation-only distinct time-origin views for Q30 experiments."""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.eval.reference_targets import validate_audio as _validate_audio
from upmixer.eval.types import RunSettings


SeparateFn = Callable[[str], tuple[dict[str, np.ndarray], "RunSettings"]]
_ACCOUNTING_FIELDS = (
    "scheduled_windows",
    "evaluated_unique_windows",
    "tail_replay_contributions",
    "model_forward_calls",
)


@dataclass
class OriginViewOutput:
    """One retained, aligned output from an evaluation origin view."""

    origin_samples: int
    stems: dict[str, np.ndarray]
    input_frame_count: int
    raw_output_frame_count: int
    runtime_s: float
    output_paths: dict[str, str] | None = None
    input_sample_rate: int | None = None


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
        views: list[dict[str, object]] = []
        reasons: list[str] = []
        for view in self.view_outputs:
            accounting = _roformer_accounting(self.settings, view)
            reason = accounting.get("unsupported_reason")
            if isinstance(reason, str):
                reasons.append(reason)
            views.append(
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
                field: sum(int(view[field]) for view in views)
                for field in _ACCOUNTING_FIELDS
            }
        payload: dict[str, object] = {
            "origin_schedule": [view.origin_samples for view in self.view_outputs],
            "effective_view_count": len(self.view_outputs),
            "separation_call_count": len(self.view_outputs),
            "runtime_s": float(self.runtime_s),
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_scope": "parent_process_lifetime_peak_rss",
            "views": views,
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


def _unsupported_accounting(reason: str) -> dict[str, object]:
    return {
        **{field: None for field in _ACCOUNTING_FIELDS},
        "unsupported_reason": f"unsupported: {reason}",
    }


def _roformer_accounting(
    settings: RunSettings, view: OriginViewOutput
) -> dict[str, object]:
    """Reconstruct the eval-only Roformer window schedule from run metadata."""
    if settings.model_arch not in {"bs_roformer", "mel_band_roformer"}:
        return _unsupported_accounting(
            f"architecture {settings.model_arch or 'unknown'}"
        )
    if settings.tta:
        return _unsupported_accounting("test-time augmentation")
    if settings.pitch_shift is not None:
        return _unsupported_accounting("pitch shift")
    if settings.chunk_duration_s is not None:
        return _unsupported_accounting("long-file chunking")
    if settings.ensemble_algorithm or settings.ensemble_models or settings.plan:
        return _unsupported_accounting("ensemble or production-tree context")
    if view.input_sample_rate != settings.sample_rate:
        return _unsupported_accounting("input resampling")
    if (
        settings.input_sample_rate is not None
        and settings.input_sample_rate != view.input_sample_rate
    ):
        return _unsupported_accounting("inconsistent input sample-rate metadata")
    if not settings.stage_settings:
        return _unsupported_accounting("missing direct-stage metadata")
    if len(settings.stage_settings) != 1:
        return _unsupported_accounting("multiple model stages")
    stage = settings.stage_settings[0]
    if stage.model_arch != settings.model_arch:
        return _unsupported_accounting("inconsistent architecture metadata")
    if stage.oom_fallback_count or stage.oom_fallback_attempts:
        return _unsupported_accounting("OOM fallback retry")
    try:
        from upmixer.separation.inference.config import load_model_config

        config = load_model_config(settings.model_config_name or "")
        segment_size = int(
            settings.segment_size
            if settings.segment_size is not None
            else config.default_segment_size
        )
        hop_length = int(config.stft_hop_length)
        overlap = int(settings.overlap) if settings.overlap is not None else 0
        batch_size = int(settings.batch_size) if settings.batch_size is not None else 0
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        return _unsupported_accounting(f"model config unavailable ({exc})")
    if segment_size < 2 or hop_length < 1:
        return _unsupported_accounting("invalid effective model chunk settings")
    if overlap < 1 or batch_size < 1:
        return _unsupported_accounting("missing effective overlap or batch settings")

    chunk_size = hop_length * (segment_size - 1)
    n_samples = max(view.input_frame_count, chunk_size)
    step = max(1, chunk_size // overlap)
    starts = [
        start if start + chunk_size <= n_samples else n_samples - chunk_size
        for start in range(0, n_samples, step)
    ]
    if not starts:
        return _unsupported_accounting("empty input schedule")
    tail_start = starts[-1]
    unique_end = len(starts) - 1
    while unique_end > 0 and starts[unique_end - 1] == tail_start:
        unique_end -= 1
    unique_end += 1
    unique_windows = unique_end
    tail_replays = len(starts) - unique_windows
    return {
        "scheduled_windows": len(starts),
        "evaluated_unique_windows": unique_windows,
        "tail_replay_contributions": tail_replays,
        "model_forward_calls": (unique_windows + batch_size - 1) // batch_size,
    }


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
        input_sample_rate=source_rate,
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
    if extra_settings != baseline_settings:
        raise ValueError("extra-origin separation settings differ from baseline")
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
        input_sample_rate=source_rate,
    )
    fused = {
        name: ((baseline[name] + extra[name]) * 0.5).astype(np.float32)
        for name in sorted(baseline)
    }
    return OriginEvaluationResult(
        stems=fused,
        settings=baseline_settings,
        view_outputs=(baseline_view, extra_view),
        runtime_s=time.perf_counter() - total_started,
        peak_memory_bytes=_peak_memory_bytes(),
    )
