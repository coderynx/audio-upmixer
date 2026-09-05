"""Focused tests for the production-tree evaluation adapter."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from upmixer.config import UpmixConfig
from upmixer.eval import separate_tree_for_eval
from upmixer.separation.separator import SeparationSettings
from upmixer.separation.stem_plan import (
    ENSEMBLE_ALGORITHM,
    MODEL_DEUX,
    MODEL_ENSEMBLE,
    MODEL_PRIMARY,
    resolve_separation_plan,
)
from upmixer.separation.stem_store import PlainStemStore


def _settings(model: str) -> SeparationSettings:
    return SeparationSettings(
        model=model,
        sample_rate=44_100,
        batch_size=1,
        segment_size=64,
        chunk_duration_s=60.0,
        overlap=2,
        tta=False,
        pitch_shift=None,
        backend="cpu",
        model_arch="bs_roformer",
        model_config_name="first-config",
        model_native_sample_rate=44_100,
        device="cpu",
    )


def _fake_pipeline(
    stems: dict[str, np.ndarray],
    result_stems: list[str],
    *,
    store_rate: int = 44_100,
    stage_settings: tuple[SeparationSettings, ...] = (),
    write_store: bool = True,
    input_sample_rate: int | None = None,
    output_sample_rate: int | None = None,
):
    class FakePipeline:
        instances: list[FakePipeline] = []

        def __init__(self, config):
            self.config = config
            self.last_separation_settings = stage_settings
            self.prepare_calls: list[str] = []
            self.retain_private_calls: list[bool] = []
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def prepare_stems(self, input_path: str, *, retain_private: bool = False):
            self.prepare_calls.append(input_path)
            self.retain_private_calls.append(retain_private)
            if write_store:
                PlainStemStore(self.config.stem_output_dir).write(stems, store_rate)
            return SimpleNamespace(
                stems=result_stems,
                input_sample_rate=input_sample_rate,
                output_sample_rate=output_sample_rate,
            )

        def process_file(self, *_args, **_kwargs):
            raise AssertionError("evaluation must use public prepare_stems")

    return FakePipeline


def _stems(*names: str) -> dict[str, np.ndarray]:
    audio = np.ones((8, 2), dtype=np.float32)
    return {name: audio for name in names}


def test_tree_uses_public_prepare_and_fresh_isolated_store(tmp_path):
    fake = _fake_pipeline(_stems("Vocals"), ["Vocals"])
    original = UpmixConfig(
        stem_cache_dir=str(tmp_path / "cache"),
        stem_input_dir=str(tmp_path / "input"),
        stem_output_dir=str(tmp_path / "output"),
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        stems, settings = separate_tree_for_eval("mix.wav", 44_100, original)
        second_stems, _ = separate_tree_for_eval("mix.wav", 44_100, original)

    assert stems.keys() == second_stems.keys() == {"Vocals"}
    assert len(fake.instances) == 2
    output_dirs = [instance.config.stem_output_dir for instance in fake.instances]
    assert len(set(output_dirs)) == 2
    for instance in fake.instances:
        assert instance.prepare_calls == ["mix.wav"]
        assert instance.config.stem_cache_dir is None
        assert instance.config.stem_input_dir is None
        assert instance.config.stem_output_dir != original.stem_output_dir
        assert not Path(instance.config.stem_output_dir).exists()
    assert settings.model == "production-tree"
    assert original.stem_cache_dir.endswith("cache")
    assert original.stem_input_dir.endswith("input")
    assert original.stem_output_dir.endswith("output")


def test_tree_preserves_zone_keys_and_stage_order(tmp_path):
    stage_settings = (_settings("first.ckpt"), _settings("second.ckpt"))
    fake = _fake_pipeline(
        _stems("Other@front", "Vocals@front", "Vocals@surround"),
        ["Vocals"],
        stage_settings=stage_settings,
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        stems, settings = separate_tree_for_eval("mix.wav", 44_100, UpmixConfig())

    assert list(stems) == ["Vocals@front", "Vocals@surround"]
    assert settings.stage_settings == stage_settings


def test_tree_opt_in_exposes_all_public_store_outputs(tmp_path):
    fake = _fake_pipeline(
        _stems("Vocals@front", "Bass@front", "Bass@surround", "_deux_inst@front"),
        ["Bass"],
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        requested, _ = separate_tree_for_eval(
            "mix.wav", 44_100, UpmixConfig(stems=["Bass"])
        )
        public, _ = separate_tree_for_eval(
            "mix.wav",
            44_100,
            UpmixConfig(stems=["Bass"]),
            include_all_public=True,
        )

    assert set(requested) == {"Bass@front", "Bass@surround"}
    assert set(public) == {"Vocals@front", "Bass@front", "Bass@surround"}


def test_tree_opt_in_exposes_only_private_terminal_outputs(tmp_path):
    fake = _fake_pipeline(
        _stems("Vocals@front", "_deux_inst@front", "_crowd_other@front"),
        ["Vocals"],
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        stems, _ = separate_tree_for_eval(
            "mix.wav",
            44_100,
            UpmixConfig(stems=["Vocals"]),
            include_private=True,
        )

    assert set(stems) == {"Vocals@front", "_deux_inst@front"}
    assert fake.instances[-1].retain_private_calls == [True]


def test_tree_does_not_report_unrun_ensemble():
    fake = _fake_pipeline(
        _stems("Vocals"),
        ["Vocals"],
        stage_settings=(_settings("becruily_deux.ckpt"),),
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        _, settings = separate_tree_for_eval(
            "mix.wav",
            44_100,
            UpmixConfig(stems=["Vocals"], stem_ensemble=True),
        )

    assert settings.ensemble_algorithm is None
    assert settings.ensemble_models is None
    assert settings.stem_ensemble is False


def test_tree_reports_observed_ensemble_pair():
    fake = _fake_pipeline(
        _stems("Bass"),
        ["Bass"],
        stage_settings=(_settings(MODEL_PRIMARY), _settings(MODEL_ENSEMBLE)),
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        _, settings = separate_tree_for_eval(
            "mix.wav",
            44_100,
            UpmixConfig(stems=["Bass"], stem_ensemble=True),
        )

    assert settings.ensemble_algorithm == ENSEMBLE_ALGORITHM
    assert settings.ensemble_models == (MODEL_PRIMARY, MODEL_ENSEMBLE)


def test_tree_leaves_inference_settings_unresolved_when_all_silent():
    fake = _fake_pipeline(
        {"Vocals": np.zeros((8, 2), dtype=np.float32)},
        ["Vocals"],
    )
    config = UpmixConfig(
        stems=["Vocals"],
        stem_silence_skip=True,
        stem_silence_threshold_db=-72.0,
        stem_silence_min_duration_s=3.0,
        stem_silence_crossfade_ms=25.0,
        stem_silence_pad_ms=300.0,
        stem_primary_remask=False,
        stem_drum_remask=False,
        stem_bleed_reduction=True,
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        stems, settings = separate_tree_for_eval("silent.wav", 44_100, config)

    assert not np.any(stems["Vocals"])
    assert settings.stage_settings == ()
    for name in (
        "batch_size",
        "segment_size",
        "chunk_duration_s",
        "overlap",
        "tta",
        "pitch_shift",
        "backend",
        "model_arch",
        "model_config_name",
        "model_native_sample_rate",
        "device",
    ):
        assert getattr(settings, name) is None
    assert settings.stem_silence_skip is True
    assert settings.stem_silence_threshold_db == -72.0
    assert settings.stem_silence_min_duration_s == 3.0
    assert settings.stem_silence_crossfade_ms == 25.0
    assert settings.stem_silence_pad_ms == 300.0
    assert settings.stem_primary_remask is False
    assert settings.stem_drum_remask is False
    assert settings.stem_bleed_reduction is True


def test_tree_records_observed_context_and_effective_stage_values():
    stage_settings = (
        _settings(MODEL_DEUX),
        _settings(MODEL_PRIMARY),
        _settings(MODEL_ENSEMBLE),
    )
    config = UpmixConfig(
        stems=["Bass"],
        stem_ensemble=True,
        stem_primary_remask=True,
        stem_drum_remask=False,
        stem_bleed_reduction=True,
        stem_silence_skip=False,
        stem_silence_threshold_db=-72.0,
        stem_silence_min_duration_s=3.0,
        stem_silence_crossfade_ms=25.0,
        stem_silence_pad_ms=300.0,
    )
    fake = _fake_pipeline(
        _stems("Bass"),
        ["Bass"],
        store_rate=44_100,
        stage_settings=stage_settings,
        input_sample_rate=48_000,
        output_sample_rate=44_100,
    )

    with patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake):
        _, settings = separate_tree_for_eval("mix.wav", 44_100, config)

    expected_plan = resolve_separation_plan(["Bass"], stem_ensemble=True)
    assert settings.input_sample_rate == 48_000
    assert settings.separation_sample_rate == 44_100
    assert settings.output_sample_rate == 44_100
    assert settings.scoring_sample_rate == 44_100
    assert settings.batch_size == 1
    assert settings.segment_size == 64
    assert settings.chunk_duration_s == 60.0
    assert settings.overlap == 2
    assert settings.tta is False
    assert settings.backend == "cpu"
    assert settings.device == "cpu"
    assert settings.model_arch == "bs_roformer"
    assert settings.model_config_name == "first-config"
    assert settings.model_native_sample_rate == 44_100
    assert settings.stem_primary_remask is True
    assert settings.stem_drum_remask is False
    assert settings.stem_bleed_reduction is True
    assert settings.stem_ensemble is True
    assert settings.stem_silence_skip is False
    assert settings.stem_silence_threshold_db == -72.0
    assert settings.stem_silence_min_duration_s == 3.0
    assert settings.stem_silence_crossfade_ms == 25.0
    assert settings.stem_silence_pad_ms == 300.0
    assert settings.plan == {
        "requested_stems": sorted(expected_plan.requested_stems),
        "stems_hash": expected_plan.stems_hash,
        "inference_hash": expected_plan.inference_hash,
        "tasks": [
            {
                "model": task.model,
                "input_source": task.input_source,
                "output_stems": sorted(task.output_stems),
                "keep_stems": sorted(task.keep_stems),
                "ensemble_models": list(task.ensemble_models),
                "ensemble_stems": sorted(task.ensemble_stems),
                "ensemble_algorithm": (
                    ENSEMBLE_ALGORITHM if task.ensemble_models else None
                ),
                "executed": True,
            }
            for task in expected_plan.tasks
        ],
    }


def test_tree_rejects_missing_store():
    fake = _fake_pipeline({}, ["Vocals"], write_store=False)

    with (
        patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake),
        pytest.raises(RuntimeError, match="stem store is missing"),
    ):
        separate_tree_for_eval("mix.wav", 44_100, UpmixConfig())


def test_tree_rejects_store_rate_mismatch():
    fake = _fake_pipeline(_stems("Vocals"), ["Vocals"], store_rate=48_000)

    with (
        patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake),
        pytest.raises(ValueError, match="sample rate 48000.*44100"),
    ):
        separate_tree_for_eval("mix.wav", 44_100, UpmixConfig())


def test_tree_rejects_empty_filtered_output():
    fake = _fake_pipeline(_stems("Other"), ["Vocals"])

    with (
        patch("upmixer.separation.stem_pipeline.StemUpmixPipeline", fake),
        pytest.raises(RuntimeError, match="no requested stems"),
    ):
        separate_tree_for_eval("mix.wav", 44_100, UpmixConfig())
