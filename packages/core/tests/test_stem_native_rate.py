"""Opt-in native-rate separation contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.separation.stem_pipeline import StemUpmixPipeline
from upmixer.separation.stem_pipeline_separate import _resolve_separation_sample_rate
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


def test_native_policy_uses_model_config_rate_and_delivery_length(tmp_path: Path):
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
        stem_native_rate=True,
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

    assert seen == [(48_000, 44_100)]
    assert result.sep_sr == 96_000
    assert result.out_sr == 96_000
    assert len(result.all_stems["Vocals"]) == round(480 * 96_000 / 48_000)


def test_native_policy_converts_supplied_stems_to_delivery_rate(tmp_path: Path):
    source = _source(tmp_path / "source.wav", frames=480)
    store = tmp_path / "prepared"
    PlainStemStore(str(store)).write(
        {"Vocals": np.ones((441, 2), dtype=np.float32)}, 44_100
    )
    config = UpmixConfig(
        stems=["Vocals"],
        output_sample_rate=48_000,
        stem_native_rate=True,
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


def test_native_policy_rejects_mixed_model_rates(monkeypatch):
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
            UpmixConfig(stem_native_rate=True), plan, 48_000
        )


def test_native_policy_changes_cache_identity():
    from upmixer.separation.stem_identity import stem_cache_identity

    plan = resolve_separation_plan(["Vocals"])
    assert stem_cache_identity(plan, UpmixConfig()) != stem_cache_identity(
        plan, UpmixConfig(stem_native_rate=True)
    )
