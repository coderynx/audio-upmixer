"""Opt-in Q20 sample-rate arms for the evaluation harness.

The production separator still works at its requested output rate.  These
helpers make a native-rate comparison by resampling a temporary input, running
the existing public separator/tree, and converting its estimates back to the
delivery rate before scoring.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from upmixer.config import UpmixConfig
from upmixer.eval.harness import (
    RunSettings,
    separate_for_eval,
    separate_tree_for_eval,
)
from upmixer.separation.inference.config import load_model_config
from upmixer.separation.inference.registry import get_model_spec
from upmixer.separation.stem_plan import (
    DEFAULT_STEMS,
    normalize_stems,
    resolve_separation_plan,
)

RateArm = Literal["delivery", "native"]


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
    return int(round(frames * target_rate / source_rate))


def _resample(audio: np.ndarray, source_rate: int, target_rate: int, frames: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        values = resample_poly(
            values,
            target_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    if values.shape[0] < frames:
        values = np.pad(values, ((0, frames - values.shape[0]), (0, 0)))
    return np.asarray(values[:frames], dtype=np.float32)


def _native_model_rate(model: str) -> int:
    spec = get_model_spec(model)
    rate = load_model_config(spec.config_name).sample_rate
    return _validate_rate(rate, f"native sample rate for {model}")


def _native_tree_rate(config: UpmixConfig) -> int:
    canonical = normalize_stems(config.stems) if config.stems else list(DEFAULT_STEMS)
    plan = resolve_separation_plan(canonical, config.stem_ensemble)
    rates = {_native_model_rate(task.model) for task in plan.tasks}
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
    if arm == "delivery":
        stems, settings = separate_for_eval(
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
        return stems, _normalise_settings(
            settings,
            rate_arm=arm,
            source_rate=source_rate,
            separation_rate=delivery_rate,
            delivery_rate=delivery_rate,
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
    delivery_frames = _target_frames(source_frames, source_rate, delivery_rate)
    stems = {
        name: _resample(audio, native_rate, delivery_rate, delivery_frames)
        for name, audio in native_stems.items()
    }
    return stems, _normalise_settings(
        settings,
        rate_arm=arm,
        source_rate=source_rate,
        separation_rate=native_rate,
        delivery_rate=delivery_rate,
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
    if arm == "delivery":
        stems, settings = separate_tree_for_eval(
            mixture_path, delivery_rate, config
        )
        return stems, _normalise_settings(
            settings,
            rate_arm=arm,
            source_rate=source_rate,
            separation_rate=delivery_rate,
            delivery_rate=delivery_rate,
        )

    native_rate = _native_tree_rate(config)
    native_config = replace(config, output_sample_rate=native_rate)
    with TemporaryDirectory(prefix="upmixer_q20_rate_") as temp_dir:
        native_path = _native_input(mixture_path, native_rate, temp_dir)
        native_stems, settings = separate_tree_for_eval(
            native_path, native_rate, native_config
        )
    delivery_frames = _target_frames(source_frames, source_rate, delivery_rate)
    stems = {
        name: _resample(audio, native_rate, delivery_rate, delivery_frames)
        for name, audio in native_stems.items()
    }
    return stems, _normalise_settings(
        settings,
        rate_arm=arm,
        source_rate=source_rate,
        separation_rate=native_rate,
        delivery_rate=delivery_rate,
    )
