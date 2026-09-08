"""Native-rate separation contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.separation.stem_pipeline import StemUpmixPipeline
from upmixer.separation.stem_pipeline_separate import (
    _load_cached_stems,
    _resample_stems,
    _resolve_separation_sample_rate,
)
from upmixer.separation.stem_plan import (
    SeparationPlan,
    SeparationTask,
    resolve_separation_plan,
)
from upmixer.separation.stem_store import PlainStemStore


def _source(path: Path, frames: int = 480, sample_rate: int = 48_000) -> str:
    sf.write(
        path, np.zeros((frames, 2), dtype=np.float32), sample_rate, subtype="FLOAT"
    )
    return str(path)


def test_native_separation_uses_model_config_rate_and_delivery_length(tmp_path: Path):
    source = _source(tmp_path / "source.wav")
    seen: list[tuple[int, int]] = []

    def fake_execute(_get_separator, plan, sep_path, sep_sr, *_args, **_kwargs):
        audio, source_sr = sf.read(sep_path, dtype="float32", always_2d=True)
        seen.append((source_sr, sep_sr))
        frames = round(len(audio) * sep_sr / source_sr)
        return {
            name: np.ones((frames, 2), dtype=np.float32)
            for name in plan.requested_stems
        }

    config = UpmixConfig(
        stems=["Vocals"],
        output_sample_rate=96_000,
        stem_silence_skip=False,
    )
    pipeline = StemUpmixPipeline(config)
    try:
        with patch(
            "upmixer.separation.stem_pipeline_separate.execute_plan",
            side_effect=fake_execute,
        ):
            result = pipeline._separate(source, None, lambda *_: None)
    finally:
        pipeline.close()

    assert seen == [(44_100, 44_100)]
    assert result.sep_sr == 96_000
    assert result.out_sr == 96_000
    assert len(result.all_stems["Vocals"]) == round(480 * 96_000 / 48_000)


def test_native_cache_reuses_canonical_source_rate_for_movement(tmp_path: Path):
    source = _source(tmp_path / "source.wav")
    cache_dir = tmp_path / "cache"
    prepared_dir = tmp_path / "prepared"
    seen: list[tuple[int, int]] = []

    def fake_execute(_get_separator, plan, sep_path, sep_sr, *_args, **_kwargs):
        audio, source_sr = sf.read(sep_path, dtype="float32", always_2d=True)
        seen.append((source_sr, sep_sr))
        return {
            name: np.ones((len(audio), 2), dtype=np.float32)
            for name in plan.requested_stems
        }

    config = UpmixConfig(
        stems=["Vocals"],
        output_sample_rate=96_000,
        stem_cache_dir=str(cache_dir),
        stem_output_dir=str(prepared_dir),
        stem_silence_skip=False,
    )
    pipeline = StemUpmixPipeline(config)
    try:
        with patch(
            "upmixer.separation.stem_pipeline_separate.execute_plan",
            side_effect=fake_execute,
        ):
            first = pipeline._separate(source, None, lambda *_: None)
            second = pipeline._separate(source, None, lambda *_: None)
    finally:
        pipeline.close()

    assert seen == [(44_100, 44_100)]
    assert first.movement_features == second.movement_features
    assert first.movement_features["sample_rate"] == 44_100
    stored, stored_rate = PlainStemStore(str(prepared_dir)).load()
    assert stored_rate == 44_100
    assert PlainStemStore(str(prepared_dir)).load_features(list(stored)) == first.movement_features
    assert len(second.all_stems["Vocals"]) == round(480 * 96_000 / 48_000)


def test_native_separation_keeps_silence_skip_zone_at_native_rate(tmp_path: Path):
    source_path = tmp_path / "active-source.wav"
    source_audio = np.zeros((4_800, 2), dtype=np.float32)
    source_audio[1_000] = 1.0
    sf.write(source_path, source_audio, 48_000, subtype="FLOAT")
    seen: list[tuple[int, int, int]] = []

    def fake_execute(_get_separator, plan, sep_path, sep_sr, *_args, **_kwargs):
        audio, source_sr = sf.read(sep_path, dtype="float32", always_2d=True)
        seen.append((source_sr, sep_sr, len(audio)))
        return {
            name: np.ones((len(audio), 2), dtype=np.float32)
            for name in plan.requested_stems
        }

    config = UpmixConfig(
        stems=["Vocals"],
        output_sample_rate=48_000,
        stem_silence_skip=True,
    )
    pipeline = StemUpmixPipeline(config)
    try:
        with patch(
            "upmixer.separation.stem_pipeline_exec.execute_plan",
            side_effect=fake_execute,
        ):
            result = pipeline._separate(str(source_path), None, lambda *_: None)
    finally:
        pipeline.close()

    assert seen == [(44_100, 44_100, round(4_800 * 44_100 / 48_000))]
    assert len(result.all_stems["Vocals"]) == 4_800


def test_native_separation_converts_supplied_stems_to_delivery_rate(tmp_path: Path):
    source = _source(tmp_path / "source.wav", frames=480)
    store = tmp_path / "prepared"
    PlainStemStore(str(store)).write(
        {"Vocals": np.ones((441, 2), dtype=np.float32)}, 44_100
    )
    config = UpmixConfig(
        stems=["Vocals"],
        output_sample_rate=48_000,
        stem_input_dir=str(store),
    )
    pipeline = StemUpmixPipeline(config)
    try:
        with patch(
            "upmixer.separation.stem_pipeline_separate.execute_plan",
            side_effect=AssertionError("supplied stems must bypass inference"),
        ):
            result = pipeline._separate(source, None, lambda *_: None)
    finally:
        pipeline.close()

    assert result.sep_sr == 48_000
    assert len(result.all_stems["Vocals"]) == 480


def test_native_separation_rejects_mixed_model_rates(monkeypatch):
    plan = SeparationPlan(
        tasks=[
            SeparationTask(
                "first.ckpt", "original", frozenset({"A"}), frozenset({"A"})
            ),
            SeparationTask("second.ckpt", "A", frozenset({"B"}), frozenset({"B"})),
        ],
        requested_stems=frozenset({"B"}),
        stems_hash="hash",
    )
    rates = {"first.ckpt": 44_100, "second.ckpt": 48_000}
    monkeypatch.setattr(
        "upmixer.separation.stem_pipeline_separate.resolve_model_native_sample_rate",
        rates.__getitem__,
    )

    with pytest.raises(ValueError, match="mixed-rate plans"):
        _resolve_separation_sample_rate(
            plan, 48_000
        )


def test_native_cache_identity_is_unconditional():
    from upmixer.separation.stem_identity import stem_cache_identity

    plan = resolve_separation_plan(["Vocals"])
    config = UpmixConfig()
    assert stem_cache_identity(plan, config, 44_100) != stem_cache_identity(
        plan, config, 48_000
    )
    with pytest.raises(ValueError, match="native_sample_rate"):
        stem_cache_identity(plan, config)


def test_native_cache_identity_preserves_legacy_suffix_order():
    from upmixer.separation.stem_identity import stem_cache_identity

    plan = resolve_separation_plan(["Bass"])
    config = UpmixConfig(stem_bleed_reduction=True)
    expected_raw = (
        f"{plan.inference_hash}|batch=None|segment=None|chunk=None"
        "|overlap=None|tta=False|pitch=None|stemcleanup=1|primaryremask"
        "|native-rate-v1|sr=44100"
    )
    expected = hashlib.sha256(expected_raw.encode()).hexdigest()[:20]

    assert stem_cache_identity(plan, config, 44_100) == expected


def test_native_cache_identity_does_not_load_legacy_stem_hash_cache(tmp_path: Path):
    from upmixer.separation.stem_cache import StemCache
    from upmixer.separation.stem_identity import stem_cache_identity

    source = _source(tmp_path / "source.wav")
    cache_dir = tmp_path / "cache"
    plan = resolve_separation_plan(["Vocals"])
    cached = {"Vocals": np.ones((480, 2), dtype=np.float32)}
    StemCache(str(cache_dir)).save(source, plan.stems_hash, 48_000, cached, 48_000)

    config = UpmixConfig(stems=["Vocals"], stem_cache_dir=str(cache_dir))
    native_identity = stem_cache_identity(plan, config, 44_100)
    assert _load_cached_stems(config, source, 48_000, native_identity) is None


def test_supplied_stems_skip_mixed_native_rate_validation(tmp_path: Path, monkeypatch):
    plan = resolve_separation_plan(["Bass", "Kick"])
    mixed_rates = {
        task.model: rate
        for task, rate in zip(plan.tasks, (44_100, 48_000, 48_000))
    }
    assert len(set(mixed_rates.values())) > 1
    monkeypatch.setattr(
        "upmixer.separation.stem_pipeline_separate.resolve_model_native_sample_rate",
        mixed_rates.__getitem__,
    )

    source = _source(tmp_path / "source.wav")
    store = tmp_path / "prepared"
    PlainStemStore(str(store)).write(
        {
            "Bass": np.ones((480, 2), dtype=np.float32),
            "Kick": np.ones((480, 2), dtype=np.float32),
        },
        48_000,
    )
    pipeline = StemUpmixPipeline(
        UpmixConfig(
            stems=["Bass", "Kick"],
            output_sample_rate=48_000,
            stem_input_dir=str(store),
        )
    )
    try:
        result = pipeline._separate(source, None, lambda *_: None)
    finally:
        pipeline.close()

    assert result.sep_sr == 48_000


def test_native_boundary_resampling_uses_exact_rounded_lengths():
    source = np.zeros((480, 2), dtype=np.float32)
    expected = round(480 * 44_100 / 48_000)

    stems = _resample_stems({"Vocals": source}, 48_000, 44_100, expected)

    assert stems["Vocals"].shape == (expected, 2)


def test_matching_rate_stems_are_reused_without_copy():
    stems = {"Vocals": np.ones((480, 2), dtype=np.float32)}

    assert _resample_stems(stems, 48_000, 48_000, 480) is stems
