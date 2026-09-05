"""Q20 rate-arm checks with a delay-sensitive fake separator."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval.harness import RunSettings
from upmixer.eval.rate_experiment import (
    separate_model_for_rate_experiment,
    separate_tree_for_rate_experiment,
)
from upmixer.eval.report import EvalReport, StemScore, format_report


def _load_runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("upmixer_rate_run_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source(path, sample_rate: int = 48_000, frames: int = 480) -> str:
    audio = np.zeros((frames, 2), dtype=np.float32)
    audio[100, 0] = 1.0
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return str(path)


def test_delivery_arm_preserves_incumbent_rate_and_does_not_probe_model_config(
    tmp_path,
):
    source = _write_source(tmp_path / "mix.wav")
    calls: list[tuple[int, int, int]] = []

    def fake_separate(path, *, sample_rate, model, **_kwargs):
        audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
        delay = max(1, round(sample_rate * 0.001))
        assert file_rate == sample_rate
        assert abs(len(audio) / sample_rate - 0.01) < 1e-9
        calls.append((file_rate, len(audio), delay))
        delayed = np.zeros_like(audio)
        delayed[delay:] = audio[:-delay]
        return {"Vocals": delayed}, RunSettings(model=model, sample_rate=sample_rate)

    with patch(
        "upmixer.eval.rate_experiment.separate_for_eval",
        side_effect=fake_separate,
    ) as separate:
        stems, settings = separate_model_for_rate_experiment(
            source,
            delivery_sample_rate=48_000,
            model="fake.ckpt",
            rate_arm="delivery",
        )

    separate.assert_called_once()
    assert calls == [(48_000, 480, 48)]
    assert stems["Vocals"].shape == (480, 2)
    assert np.argmax(np.abs(stems["Vocals"][:, 0])) == 100 + round(48_000 * 0.001)
    assert settings.sample_rate == 48_000
    assert settings.input_sample_rate == 48_000
    assert settings.separation_sample_rate == 48_000
    assert settings.output_sample_rate == 48_000
    assert settings.scoring_sample_rate == 48_000


def test_native_arm_uses_config_rate_and_preserves_duration_and_delay(tmp_path):
    source = _write_source(tmp_path / "mix.wav")
    calls: list[tuple[int, int, int]] = []

    def fake_separate(path, *, sample_rate, model, **_kwargs):
        audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
        delay = max(1, round(sample_rate * 0.001))
        assert file_rate == sample_rate
        assert abs(len(audio) / sample_rate - 0.01) < 1e-9
        calls.append((file_rate, len(audio), delay))
        delayed = np.zeros_like(audio)
        delayed[delay:] = audio[:-delay]
        return {"Vocals": delayed}, RunSettings(model=model, sample_rate=sample_rate)

    spec = SimpleNamespace(config_name="fake-config")
    config = SimpleNamespace(sample_rate=44_100)
    with (
        patch("upmixer.eval.rate_experiment.get_model_spec", return_value=spec) as get_spec,
        patch(
            "upmixer.eval.rate_experiment.load_model_config", return_value=config
        ) as load_config,
        patch(
            "upmixer.eval.rate_experiment.separate_for_eval",
            side_effect=fake_separate,
        ) as separate,
    ):
        stems, settings = separate_model_for_rate_experiment(
            source,
            delivery_sample_rate=48_000,
            model="fake.ckpt",
            rate_arm="native",
        )

    get_spec.assert_called_once_with("fake.ckpt")
    load_config.assert_called_once_with("fake-config")
    separate.assert_called_once()
    assert calls == [(44_100, 441, 44)]
    assert stems["Vocals"].shape == (480, 2)
    assert np.argmax(np.abs(stems["Vocals"][:, 0])) == 100 + round(48_000 * 0.001)
    assert settings.sample_rate == 48_000
    assert settings.input_sample_rate == 48_000
    assert settings.separation_sample_rate == 44_100
    assert settings.output_sample_rate == 48_000
    assert settings.scoring_sample_rate == 48_000


def test_rate_arm_is_serialized_for_comparison_reports():
    settings = RunSettings(model="fake.ckpt", sample_rate=48_000, rate_arm="native")
    report = EvalReport(
        settings=settings,
        scores=[StemScore("Vocals", "default", 0.0, 0.0, 0.0)],
    )
    assert settings.rate_arm == "native"
    assert "rate_arm=native" in format_report(report)


def test_native_tree_arm_resamples_the_complete_tree_input_and_outputs(tmp_path):
    source = _write_source(tmp_path / "mix.wav")
    calls: list[tuple[int, int]] = []

    def fake_tree(path, sample_rate, config):
        audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
        assert file_rate == sample_rate == 44_100
        calls.append((file_rate, len(audio)))
        return {"Vocals": audio}, RunSettings(
            model="production-tree", sample_rate=sample_rate
        )

    spec = SimpleNamespace(config_name="fake-config")
    config = SimpleNamespace(sample_rate=44_100)
    with (
        patch("upmixer.eval.rate_experiment.get_model_spec", return_value=spec),
        patch("upmixer.eval.rate_experiment.load_model_config", return_value=config),
        patch(
            "upmixer.eval.rate_experiment.separate_tree_for_eval",
            side_effect=fake_tree,
        ),
    ):
        stems, settings = separate_tree_for_rate_experiment(
            source,
            delivery_sample_rate=48_000,
            config=UpmixConfig(stems=["Vocals"], output_sample_rate=48_000),
            rate_arm="native",
        )

    assert calls == [(44_100, 441)]
    assert stems["Vocals"].shape == (480, 2)
    assert settings.rate_arm == "native"
    assert settings.separation_sample_rate == 44_100
    assert settings.output_sample_rate == 48_000


def test_eval_runner_exposes_native_arm_as_an_explicit_opt_in():
    runner = _load_runner()
    args = runner._parser().parse_args(
        [
            "--corpus",
            "licensed",
            "--variant",
            "real-model",
            "--model",
            "fake.ckpt",
            "--sample-rate",
            "48000",
            "--rate-arm",
            "native",
            "--output-dir",
            "/tmp/q20-rate-test",
        ]
    )
    captured = {}

    def fake_native(mixture_path, **kwargs):
        kwargs["mixture_path"] = mixture_path
        captured.update(kwargs)
        return {}, RunSettings(model="fake.ckpt", sample_rate=48_000)

    with patch.object(runner, "separate_model_for_rate_experiment", fake_native):
        runner._real_separator(args)("mix.wav")

    assert captured == {
        "mixture_path": "mix.wav",
        "delivery_sample_rate": 48_000,
        "model": "fake.ckpt",
        "rate_arm": "native",
        "batch_size": None,
        "segment_size": None,
        "chunk_duration_s": None,
        "overlap": None,
        "tta": False,
        "pitch_shift": None,
    }
