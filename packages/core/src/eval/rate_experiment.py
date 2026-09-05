"""Opt-in Q20 sample-rate arms for the evaluation harness.

The production separator still works at its requested output rate.  These
helpers make a native-rate comparison by resampling a temporary input, running
the existing public separator/tree, and converting its estimates back to the
delivery rate before scoring.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval.harness import (
    RunSettings,
    separate_for_eval,
    separate_tree_for_eval,
)
from upmixer.separation.inference.config import load_model_config
from upmixer.separation.inference.registry import get_model_spec
from upmixer.resample import resample_channels
from upmixer.separation.stem_plan import (
    DEFAULT_STEMS,
    normalize_stems,
    resolve_separation_plan,
    terminal_plan_stems,
)

RateArm = Literal["delivery", "native"]
_RESAMPLER_ID = "upmixer.resample.resample_channels:120dB-kaiser-fir"
_INCUMBENT_RESAMPLER_ID = "incumbent:librosa.load"
_NO_RESAMPLER = "none"


def _validate_rate(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_arm(rate_arm: str) -> RateArm:
    if rate_arm not in {"delivery", "native"}:
        raise ValueError("rate_arm must be 'delivery' or 'native'")
    return rate_arm  # type: ignore[return-value]


def _source_info(path: str) -> tuple[int, int]:
    info = sf.info(path)
    return info.samplerate, info.frames


def _target_frames(frames: int, source_rate: int, target_rate: int) -> int:
    """Round the target frame count from the source duration in seconds."""
    return int(round(frames * target_rate / source_rate))


def _match_length(values: np.ndarray, frames: int) -> np.ndarray:
    current = values.shape[-1]
    if current < frames:
        values = np.pad(
            values,
            [(0, 0)] * (values.ndim - 1) + [(0, frames - current)],
        )
    return values[..., :frames]


def _resample(audio: np.ndarray, source_rate: int, target_rate: int, frames: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if source_rate != target_rate:
        values = resample_channels(
            {"audio": values}, source_rate, target_rate
        )["audio"]
    return np.asarray(_match_length(values.T, frames).T, dtype=np.float32)


def _normalise_stems(
    stems: dict[str, np.ndarray],
    source_rate: int,
    target_rate: int,
    target_frames: int,
) -> dict[str, np.ndarray]:
    return {
        name: _resample(audio, source_rate, target_rate, target_frames)
        for name, audio in stems.items()
    }


def _common_frame_count(stems: dict[str, np.ndarray]) -> int:
    counts = {audio.shape[0] for audio in stems.values()}
    if len(counts) > 1:
        raise ValueError(
            "rate experiment separator returned stems with different frame counts"
        )
    return counts.pop() if counts else 0


def _terminal_public_stems(
    stems: dict[str, np.ndarray], config: UpmixConfig
) -> dict[str, np.ndarray]:
    """Keep public outputs that no later plan task consumes."""
    return _terminal_stems(stems, config)


def _terminal_stems(
    stems: dict[str, np.ndarray],
    config: UpmixConfig,
    *,
    include_private: bool = False,
) -> dict[str, np.ndarray]:
    """Keep plan outputs that no later task consumes."""
    canonical = normalize_stems(config.stems) if config.stems else list(DEFAULT_STEMS)
    plan = resolve_separation_plan(canonical, config.stem_ensemble)
    terminal = terminal_plan_stems(plan, include_private=include_private)
    return {
        key: audio
        for key, audio in stems.items()
        if key.split("@", 1)[0] in terminal
    }


def _native_model_rate(model: str) -> int:
    spec = get_model_spec(model)
    rate = load_model_config(spec.config_name).sample_rate
    return _validate_rate(rate, f"native sample rate for {model}")


def _native_tree_rate(config: UpmixConfig) -> int:
    canonical = normalize_stems(config.stems) if config.stems else list(DEFAULT_STEMS)
    plan = resolve_separation_plan(canonical, config.stem_ensemble)
    models = tuple(
        dict.fromkeys(
            model
            for task in plan.tasks
            for model in (task.model, *task.ensemble_models)
        )
    )
    rates = {_native_model_rate(model) for model in models}
    if not rates:
        raise ValueError("native-rate tree experiment requires at least one model task")
    if len(rates) != 1:
        raise ValueError(
            "native-rate tree experiment requires one model rate; "
            f"found {sorted(rates)}"
        )
    return rates.pop()


def _normalise_settings(
    settings: RunSettings,
    *,
    rate_arm: RateArm,
    source_rate: int,
    separation_rate: int,
    delivery_rate: int,
    input_frames: int,
    separation_frames: int,
    output_frames: int,
    resampler: str,
) -> RunSettings:
    if not isinstance(settings, RunSettings):
        raise ValueError("rate experiment separator returned invalid RunSettings")
    if settings.sample_rate != separation_rate:
        raise ValueError(
            "rate experiment separator settings sample rate does not match "
            f"its working rate ({settings.sample_rate} != {separation_rate})"
        )
    return replace(
        settings,
        rate_arm=rate_arm,
        sample_rate=delivery_rate,
        input_sample_rate=source_rate,
        separation_sample_rate=separation_rate,
        output_sample_rate=delivery_rate,
        scoring_sample_rate=delivery_rate,
        input_frame_count=input_frames,
        separation_frame_count=separation_frames,
        output_frame_count=output_frames,
        resampler=resampler,
    )


def _native_input(path: str, native_rate: int, temp_dir: str) -> str:
    source_rate, source_frames = _source_info(path)
    audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
    if file_rate != source_rate:
        raise RuntimeError("source metadata changed while reading rate experiment input")
    converted = _resample(
        audio,
        source_rate,
        native_rate,
        _target_frames(source_frames, source_rate, native_rate),
    )
    destination = Path(temp_dir) / "native-input.wav"
    sf.write(destination, converted, native_rate, subtype="FLOAT")
    return str(destination)


def separate_model_for_rate_experiment(
    mixture_path: str,
    *,
    delivery_sample_rate: int,
    model: str,
    rate_arm: RateArm = "delivery",
    batch_size: int | None = None,
    segment_size: int | None = None,
    chunk_duration_s: float | None = None,
    overlap: int | None = None,
    tta: bool = False,
    pitch_shift: float | None = None,
) -> tuple[dict[str, np.ndarray], RunSettings]:
    """Run one model at the incumbent or model-native rate for Q20 scoring."""
    delivery_rate = _validate_rate(delivery_sample_rate, "delivery sample rate")
    arm = _validate_arm(rate_arm)
    source_rate, source_frames = _source_info(mixture_path)
    delivery_frames = _target_frames(source_frames, source_rate, delivery_rate)
    if arm == "delivery":
        raw_stems, settings = separate_for_eval(
            mixture_path,
            sample_rate=delivery_rate,
            model=model,
            batch_size=batch_size,
            segment_size=segment_size,
            chunk_duration_s=chunk_duration_s,
            overlap=overlap,
            tta=tta,
            pitch_shift=pitch_shift,
        )
        raw_output_frames = _common_frame_count(raw_stems)
        stems = _normalise_stems(
            raw_stems, delivery_rate, delivery_rate, delivery_frames
        )
        return stems, _normalise_settings(
            settings,
            rate_arm=arm,
            source_rate=source_rate,
            separation_rate=delivery_rate,
            delivery_rate=delivery_rate,
            input_frames=source_frames,
            separation_frames=raw_output_frames,
            output_frames=delivery_frames,
            resampler=(
                _INCUMBENT_RESAMPLER_ID
                if source_rate != delivery_rate
                else _NO_RESAMPLER
            ),
        )

    native_rate = _native_model_rate(model)
    with TemporaryDirectory(prefix="upmixer_q20_rate_") as temp_dir:
        native_path = _native_input(mixture_path, native_rate, temp_dir)
        native_stems, settings = separate_for_eval(
            native_path,
            sample_rate=native_rate,
            model=model,
            batch_size=batch_size,
            segment_size=segment_size,
            chunk_duration_s=chunk_duration_s,
            overlap=overlap,
            tta=tta,
            pitch_shift=pitch_shift,
        )
    raw_output_frames = _common_frame_count(native_stems)
    stems = _normalise_stems(
        native_stems, native_rate, delivery_rate, delivery_frames
    )
    return stems, _normalise_settings(
        settings,
        rate_arm=arm,
        source_rate=source_rate,
        separation_rate=native_rate,
        delivery_rate=delivery_rate,
        input_frames=source_frames,
        separation_frames=raw_output_frames,
        output_frames=delivery_frames,
        resampler=(
            _RESAMPLER_ID
            if source_rate != native_rate or native_rate != delivery_rate
            else _NO_RESAMPLER
        ),
    )


def separate_tree_for_rate_experiment(
    mixture_path: str,
    *,
    delivery_sample_rate: int,
    config: UpmixConfig,
    rate_arm: RateArm = "delivery",
) -> tuple[dict[str, np.ndarray], RunSettings]:
    """Run the production tree at delivery/native rate for Q20 scoring."""
    delivery_rate = _validate_rate(delivery_sample_rate, "delivery sample rate")
    arm = _validate_arm(rate_arm)
    source_rate, source_frames = _source_info(mixture_path)
    delivery_frames = _target_frames(source_frames, source_rate, delivery_rate)
    if arm == "delivery":
        raw_stems, settings = separate_tree_for_eval(
            mixture_path,
            delivery_rate,
            config,
            include_all_public=True,
            include_private=True,
        )
        raw_stems = _terminal_stems(raw_stems, config, include_private=True)
        raw_output_frames = _common_frame_count(raw_stems)
        stems = _normalise_stems(
            raw_stems, delivery_rate, delivery_rate, delivery_frames
        )
        return stems, _normalise_settings(
            settings,
            rate_arm=arm,
            source_rate=source_rate,
            separation_rate=delivery_rate,
            delivery_rate=delivery_rate,
            input_frames=source_frames,
            separation_frames=raw_output_frames,
            output_frames=delivery_frames,
            resampler=(
                _INCUMBENT_RESAMPLER_ID
                if source_rate != delivery_rate
                else _NO_RESAMPLER
            ),
        )

    native_rate = _native_tree_rate(config)
    native_config = replace(config, output_sample_rate=native_rate)
    with TemporaryDirectory(prefix="upmixer_q20_rate_") as temp_dir:
        native_path = _native_input(mixture_path, native_rate, temp_dir)
        native_stems, settings = separate_tree_for_eval(
            native_path,
            native_rate,
            native_config,
            include_all_public=True,
            include_private=True,
        )
    native_stems = _terminal_stems(native_stems, native_config, include_private=True)
    raw_output_frames = _common_frame_count(native_stems)
    stems = _normalise_stems(
        native_stems, native_rate, delivery_rate, delivery_frames
    )
    return stems, _normalise_settings(
        settings,
        rate_arm=arm,
        source_rate=source_rate,
        separation_rate=native_rate,
        delivery_rate=delivery_rate,
        input_frames=source_frames,
        separation_frames=raw_output_frames,
        output_frames=delivery_frames,
        resampler=(
            _RESAMPLER_ID
            if source_rate != native_rate or native_rate != delivery_rate
            else _NO_RESAMPLER
        ),
    )
