"""Q20 rate-arm checks with a delay-sensitive fake separator."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval.harness import RunSettings
from upmixer.eval.rate_experiment import (
    _INCUMBENT_RESAMPLER_ID,
    _RESAMPLER_ID,
    _terminal_public_stems,
    separate_model_for_rate_experiment,
    separate_tree_for_rate_experiment,
)
from upmixer.eval.report import EvalReport, StemScore, format_report
from upmixer.resample import resample_channels
from upmixer.separation.stem_plan import MODEL_ENSEMBLE, MODEL_PRIMARY


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
        patch(
            "upmixer.eval.rate_experiment.resample_channels",
            wraps=resample_channels,
        ) as resample,
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
    assert resample.call_count == 2
    assert calls == [(44_100, 441, 44)]
    assert stems["Vocals"].shape == (480, 2)
    assert np.argmax(np.abs(stems["Vocals"][:, 0])) == 100 + round(48_000 * 0.001)
    assert settings.sample_rate == 48_000
    assert settings.input_sample_rate == 48_000
    assert settings.separation_sample_rate == 44_100
    assert settings.output_sample_rate == 48_000
    assert settings.scoring_sample_rate == 48_000


def test_rate_arm_is_serialized_for_comparison_reports():
    assert RunSettings(model="ordinary", sample_rate=48_000).rate_arm is None
    settings = RunSettings(
        model="fake.ckpt",
        sample_rate=48_000,
        rate_arm="native",
        input_frame_count=2_001,
        separation_frame_count=1_838,
        output_frame_count=2_001,
        resampler=_RESAMPLER_ID,
    )
    report = EvalReport(
        settings=settings,
        scores=[
            StemScore(
                "Vocals",
                "default",
                0.0,
                0.0,
                0.0,
                recording_id="recording-0",
                item_id="item-0",
            )
        ],
    )
    assert settings.rate_arm == "native"
    serialized = report.to_dict()["settings"]
    assert serialized["input_frame_count"] == 2_001
    assert serialized["separation_frame_count"] == 1_838
    assert serialized["output_frame_count"] == 2_001
    assert serialized["resampler"] == _RESAMPLER_ID
    text = format_report(report)
    assert "rate_arm=native" in text
    assert "input_frame_count=2001" in text
    assert "separation_frame_count=1838" in text
    assert "output_frame_count=2001" in text
    assert f"resampler={_RESAMPLER_ID}" in text


@pytest.mark.parametrize("rate_arm", ["delivery", "native"])
def test_rate_arms_normalise_non_divisible_source_duration(tmp_path, rate_arm):
    source = _write_source(tmp_path / f"mix-{rate_arm}.wav", frames=2_001)
    target_frames = round(2_001 * 44_100 / 48_000)
    raw_frames = target_frames + 1 if rate_arm == "delivery" else target_frames - 1
    calls: list[tuple[int, int, int]] = []

    def fake_separate(path, *, sample_rate, model, **_kwargs):
        info = sf.info(path)
        calls.append((info.samplerate, info.frames, sample_rate))
        raw = np.ones((raw_frames, 2), dtype=np.float32)
        return {"Vocals": raw}, RunSettings(model=model, sample_rate=sample_rate)

    spec = SimpleNamespace(config_name="fake-config")
    native_config = SimpleNamespace(sample_rate=44_100)
    with (
        patch("upmixer.eval.rate_experiment.get_model_spec", return_value=spec),
        patch(
            "upmixer.eval.rate_experiment.load_model_config",
            return_value=native_config,
        ),
        patch(
            "upmixer.eval.rate_experiment.separate_for_eval",
            side_effect=fake_separate,
        ),
    ):
        stems, settings = separate_model_for_rate_experiment(
            source,
            delivery_sample_rate=44_100,
            model="fake.ckpt",
            rate_arm=rate_arm,
        )

    expected_call = (
        (48_000, 2_001, 44_100)
        if rate_arm == "delivery"
        else (44_100, target_frames, 44_100)
    )
    assert calls == [expected_call]
    assert stems["Vocals"].shape == (target_frames, 2)
    expected_tail = 0.0 if rate_arm == "native" else 1.0
    assert np.all(stems["Vocals"][-1] == expected_tail)
    assert settings.input_frame_count == 2_001
    assert settings.separation_frame_count == raw_frames
    assert settings.output_frame_count == target_frames
    assert settings.resampler == (
        _INCUMBENT_RESAMPLER_ID if rate_arm == "delivery" else _RESAMPLER_ID
    )


def test_native_tree_rejects_mixed_model_and_ensemble_rates(tmp_path):
    source = _write_source(tmp_path / "mix.wav")
    specs = {
        "becruily_deux.ckpt": SimpleNamespace(config_name="deux-config"),
        MODEL_PRIMARY: SimpleNamespace(config_name="primary-config"),
        MODEL_ENSEMBLE: SimpleNamespace(config_name="ensemble-config"),
    }
    rates = {
        "deux-config": 44_100,
        "primary-config": 44_100,
        "ensemble-config": 48_000,
    }
    with (
        patch(
            "upmixer.eval.rate_experiment.get_model_spec",
            side_effect=lambda model: specs[model],
        ) as get_spec,
        patch(
            "upmixer.eval.rate_experiment.load_model_config",
            side_effect=lambda name: SimpleNamespace(sample_rate=rates[name]),
        ) as load_config,
        patch("upmixer.eval.rate_experiment.separate_tree_for_eval") as separate,
        pytest.raises(ValueError, match="one model rate"),
    ):
        separate_tree_for_rate_experiment(
            source,
            delivery_sample_rate=48_000,
            config=UpmixConfig(
                stems=["Bass"], output_sample_rate=48_000, stem_ensemble=True
            ),
            rate_arm="native",
        )

    assert get_spec.call_count == 3
    assert load_config.call_count == 3
    separate.assert_not_called()


@pytest.mark.parametrize(
    ("requested", "outputs", "expected"),
    [
        (
            "Bass",
            ("Vocals", "Bass", "Drums", "Guitar", "Piano", "Other"),
            {"Vocals", "Bass", "Drums", "Guitar", "Piano", "Other"},
        ),
        (
            "Kick",
            (
                "Vocals",
                "Bass",
                "Drums",
                "Guitar",
                "Piano",
                "Other",
                "Kick",
                "Snare",
                "Toms",
                "Hi-Hat",
                "Ride",
                "Crash",
            ),
            {
                "Vocals",
                "Bass",
                "Guitar",
                "Piano",
                "Other",
                "Kick",
                "Snare",
                "Toms",
                "Hi-Hat",
                "Ride",
                "Crash",
            },
        ),
        (
            "Lead Vocals",
            ("Vocals", "Lead Vocals", "Backing Vocals"),
            {"Lead Vocals", "Backing Vocals"},
        ),
    ],
)
def test_terminal_public_stems_exclude_consumed_parents(
    requested, outputs, expected
):
    audio = np.zeros((8, 2), dtype=np.float32)
    stems = {
        f"{name}@front": audio
        for name in outputs
    }

    terminal = _terminal_public_stems(
        stems, UpmixConfig(stems=[requested])
    )

    assert set(terminal) == {f"{name}@front" for name in expected}


@pytest.mark.parametrize("rate_arm", ["delivery", "native"])
def test_tree_rate_arms_request_all_public_and_return_terminals(tmp_path, rate_arm):
    source = _write_source(tmp_path / f"mix-{rate_arm}.wav")
    output_names = ("Vocals", "Bass", "Drums", "Guitar", "Piano", "Other")
    calls = []

    def fake_tree(path, sample_rate, config, *, include_all_public):
        audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
        calls.append((file_rate, sample_rate, config.output_sample_rate, include_all_public))
        return {
            name: audio for name in output_names
        }, RunSettings(model="production-tree", sample_rate=sample_rate)

    config = UpmixConfig(stems=["Bass"], output_sample_rate=48_000)
    with (
        patch(
            "upmixer.eval.rate_experiment.separate_tree_for_eval",
            side_effect=fake_tree,
        ),
        patch("upmixer.eval.rate_experiment._native_tree_rate", return_value=44_100),
    ):
        stems, settings = separate_tree_for_rate_experiment(
            source,
            delivery_sample_rate=48_000,
            config=config,
            rate_arm=rate_arm,
        )

    expected_rate = 48_000 if rate_arm == "delivery" else 44_100
    assert calls == [(expected_rate, expected_rate, expected_rate, True)]
    assert set(stems) == set(output_names)
    assert all(audio.shape == (480, 2) for audio in stems.values())
    assert settings.sample_rate == settings.output_sample_rate == 48_000


def test_native_tree_arm_resamples_the_complete_tree_input_and_outputs(tmp_path):
    source = _write_source(tmp_path / "mix.wav")
    calls: list[tuple[int, int]] = []

    def fake_tree(path, sample_rate, config, *, include_all_public):
        assert include_all_public is True
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


@pytest.mark.parametrize("rate_arm", ["native", "delivery"])
def test_eval_runner_exposes_rate_arms_as_an_explicit_opt_in(rate_arm):
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
            rate_arm,
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
        "rate_arm": rate_arm,
        "batch_size": None,
        "segment_size": None,
        "chunk_duration_s": None,
        "overlap": None,
        "tta": False,
        "pitch_shift": None,
    }


def test_eval_runner_without_rate_arm_keeps_incumbent_path():
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
            "--output-dir",
            "/tmp/q20-rate-default-test",
        ]
    )
    assert args.rate_arm is None
    captured = {}

    def fake_incumbent(mixture_path, **kwargs):
        captured["mixture_path"] = mixture_path
        captured.update(kwargs)
        return {}, RunSettings(model="fake.ckpt", sample_rate=48_000)

    with (
        patch.object(
            runner, "separate_for_eval", side_effect=fake_incumbent
        ) as incumbent,
        patch.object(runner, "separate_model_for_rate_experiment") as q20,
    ):
        runner._real_separator(args)("mix.wav")

    incumbent.assert_called_once()
    q20.assert_not_called()
    assert captured["mixture_path"] == "mix.wav"
    assert captured["model"] == "fake.ckpt"
    assert captured["sample_rate"] == 48_000
