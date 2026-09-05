"""Real-model check for the separation evaluation harness.

Runs the harness against the synthetic corpus with the default model and
prints the per-stem SDR/fullness/bleedless report. Requires the separation
extra (torch) and a model download; skipped unless run with -m perf.

    pytest -m perf -k eval -s
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval.corpus import synthetic_corpus
from upmixer.eval.harness import RunSettings, evaluate_corpus, separate_for_eval
from upmixer.eval.report import EvalReport, StemScore, format_report
from upmixer.separation.separator import DEFAULT_MODEL, SeparationSettings


class _FakeSeparator:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.backend = "cpu"
        self.closed = False
        self.run_settings = SeparationSettings(
            model=kwargs["model"],
            sample_rate=kwargs["sample_rate"],
            batch_size=3,
            segment_size=128,
            chunk_duration_s=45.0,
            overlap=4,
            tta=kwargs["tta"],
            pitch_shift=kwargs["pitch_shift"],
            backend="cpu",
            model_arch="bs_roformer",
            model_config_name="observed-config",
            model_native_sample_rate=44_100,
            device="cpu",
        )
        self.__class__.instances.append(self)

    def separate(self, _mixture_path):
        return {"Vocals": np.zeros((8, 2), dtype=np.float32)}

    def close(self):
        self.closed = True


@pytest.mark.perf
def test_eval_harness_reports_default_model_quality(tmp_path):
    pytest.importorskip("torch")

    sample_rate = 44100
    corpus = synthetic_corpus(sample_rate=sample_rate, out_dir=str(tmp_path / "corpus"))
    separate_fn = partial(separate_for_eval, sample_rate=sample_rate, model=DEFAULT_MODEL)

    report = evaluate_corpus(corpus, separate_fn, sample_rate=sample_rate)

    assert report.scores, "expected at least one scored stem"
    for mean_sdr, mean_fullness, mean_bleedless in report.by_stem().values():
        assert np.isfinite(mean_sdr)
        assert 0.0 <= mean_fullness <= 1.0
        assert 0.0 <= mean_bleedless <= 1.0

    print(format_report(report))


def test_separate_for_eval_records_completed_run_settings_without_registry_reload(
    tmp_path,
):
    mixture_path = tmp_path / "mixture.wav"
    sf.write(mixture_path, np.zeros((8, 2), dtype=np.float32), 8000, subtype="FLOAT")

    _FakeSeparator.instances.clear()
    with (
        patch("upmixer.eval.harness.StemSeparator", _FakeSeparator),
        patch(
            "upmixer.separation.inference.registry.get_model_spec",
            side_effect=AssertionError("registry metadata must come from snapshot"),
        ),
        patch(
            "upmixer.separation.inference.config.load_model_config",
            side_effect=AssertionError("config metadata must come from snapshot"),
        ),
    ):
        stems, settings = separate_for_eval(
            str(mixture_path),
            sample_rate=8000,
            model=DEFAULT_MODEL,
            batch_size=None,
            segment_size=None,
            chunk_duration_s=None,
            overlap=None,
            tta=True,
            pitch_shift=0.75,
        )

    assert set(stems) == {"Vocals"}
    separator = _FakeSeparator.instances[0]
    assert separator.kwargs == {
        "model": DEFAULT_MODEL,
        "sample_rate": 8000,
        "batch_size": None,
        "segment_size": None,
        "chunk_duration_s": None,
        "overlap": None,
        "tta": True,
        "pitch_shift": 0.75,
    }
    assert separator.closed
    assert settings.chunk_duration_s == 45.0
    assert settings.tta is True
    assert settings.pitch_shift == 0.75
    assert settings.backend == "cpu"
    assert settings.model_arch == "bs_roformer"
    assert settings.model_config_name == "observed-config"
    assert settings.model_native_sample_rate == 44_100
    assert settings.segment_size == 128
    assert settings.overlap == 4
    assert settings.batch_size == 3
    assert settings.device == "cpu"
    assert settings.stage_settings == (separator.run_settings,)


def test_separate_for_eval_ensemble_uses_public_tree_without_model_loading():
    expected_settings = RunSettings(model="production-tree", sample_rate=44_100)
    captured = {}

    def fake_tree(mixture_path, sample_rate, config):
        captured.update(
            mixture_path=mixture_path,
            sample_rate=sample_rate,
            config=config,
        )
        return {"Bass": np.zeros((8, 2), dtype=np.float32)}, expected_settings

    with patch("upmixer.eval.harness.separate_tree_for_eval", fake_tree):
        stems, settings = separate_for_eval(
            "mixture.wav",
            sample_rate=44_100,
            batch_size=2,
            segment_size=128,
            chunk_duration_s=30.0,
            overlap=4,
            stem_ensemble=True,
            tta=True,
            pitch_shift=0.75,
        )

    assert stems["Bass"].shape == (8, 2)
    assert settings is expected_settings
    assert captured["mixture_path"] == "mixture.wav"
    assert captured["sample_rate"] == 44_100
    config = captured["config"]
    assert isinstance(config, UpmixConfig)
    assert config.output_sample_rate == 44_100
    assert config.stem_batch_size == 2
    assert config.stem_segment_size == 128
    assert config.stem_chunk_duration_s == 30.0
    assert config.stem_overlap == 4
    assert config.stem_ensemble is True
    assert config.stem_tta is True
    assert config.stem_pitch_shift == 0.75


def test_format_report_accepts_legacy_settings_shape():
    settings = SimpleNamespace(
        model="legacy",
        sample_rate=8000,
        segment_size=None,
        overlap=None,
        batch_size=None,
        ensemble_algorithm=None,
        ensemble_models=None,
    )
    report = EvalReport(
        settings=settings,
        scores=[
            StemScore(
                stem="Vocals",
                category="default",
                sdr=1.0,
                fullness=0.5,
                bleedless=0.4,
            )
        ],
    )

    text = format_report(report)

    assert "model=legacy" in text
    assert "chunk_duration_s=None" in text
    assert "model_native_sample_rate=None" in text
    assert "device=None" in text


def test_separate_for_eval_requires_completed_run_settings(tmp_path):
    mixture_path = tmp_path / "mixture.wav"
    sf.write(mixture_path, np.zeros((8, 2), dtype=np.float32), 8000, subtype="FLOAT")

    class NoSnapshotSeparator(_FakeSeparator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_settings = None

    with patch("upmixer.eval.harness.StemSeparator", NoSnapshotSeparator):
        with pytest.raises(RuntimeError, match="run-settings snapshot"):
            separate_for_eval(str(mixture_path), sample_rate=8000)
