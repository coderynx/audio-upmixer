"""Shared evaluation settings value types."""
from __future__ import annotations

from dataclasses import dataclass

from upmixer.separation.separator import SeparationSettings


@dataclass
class RunSettings:
    """Inference configuration recorded alongside every score."""

    model: str
    sample_rate: int
    segment_size: int | None = None
    overlap: int | None = None
    batch_size: int | None = None
    ensemble_algorithm: str | None = None
    ensemble_models: tuple[str, ...] | None = None
    chunk_duration_s: float | None = None
    tta: bool = False
    pitch_shift: float | None = None
    backend: str | None = None
    model_arch: str | None = None
    model_config_name: str | None = None
    model_native_sample_rate: int | None = None
    stage_settings: tuple[SeparationSettings, ...] = ()
    device: str | None = None
    input_sample_rate: int | None = None
    separation_sample_rate: int | None = None
    output_sample_rate: int | None = None
    scoring_sample_rate: int | None = None
    plan: dict[str, object] | None = None
    stem_primary_remask: bool | None = None
    stem_drum_remask: bool | None = None
    stem_bleed_reduction: bool | None = None
    stem_ensemble: bool | None = None
    stem_silence_skip: bool | None = None
    stem_silence_threshold_db: float | None = None
    stem_silence_min_duration_s: float | None = None
    stem_silence_crossfade_ms: float | None = None
    stem_silence_pad_ms: float | None = None
    rate_arm: str | None = None
    input_frame_count: int | None = None
    separation_frame_count: int | None = None
    output_frame_count: int | None = None
    resampler: str | None = None
    origin_schedule: tuple[int, ...] | None = None


@dataclass
class ItemRunSettings:
    """Effective settings recorded for one corpus item."""

    recording_id: str | None
    item_id: str | None
    split: str | None
    category: str
    settings: RunSettings
