"""Q40 direct-Deux counterfactual cascade evaluation."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from upmixer.eval import (
    CascadeEvaluationResult,
    CorpusItem,
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    separate_with_deux_cascade,
)
from upmixer.eval.report import EvalReport, StemScore, format_report
from upmixer.separation.separator import SeparationSettings


def _runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("q40_eval_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(tmp_path: Path, frames: int = 8, sample_rate: int = 44_100):
    values = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) / 10
    path = tmp_path / "mixture.wav"
    sf.write(path, values, sample_rate, subtype="FLOAT")
    return path, values


def _settings(sample_rate: int = 44_100, *, device: str | None = None) -> RunSettings:
    stage = SeparationSettings(
        model="becruily_deux.ckpt",
        sample_rate=sample_rate,
        batch_size=1,
        segment_size=1601,
        chunk_duration_s=None,
        overlap=2,
        tta=False,
        pitch_shift=None,
        backend="test",
        model_arch="bs_roformer",
        model_config_name="becruily_deux",
        model_native_sample_rate=44_100,
        device=device,
        checkpoint_sha256=(
            "10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa"
        ),
        model_config_sha256=(
            "0539c3ee9a4800ccb3a4bf7a507dfa97e77857649651169eb907087130d0ad77"
        ),
        runtime_precision="float32",
        normalization_policy="peak-downscale-to-0.9; inverse-output-scale",
    )
    return RunSettings(
        model="becruily_deux.ckpt",
        sample_rate=sample_rate,
        segment_size=1601,
        overlap=2,
        batch_size=1,
        model_arch="bs_roformer",
        model_config_name="becruily_deux",
        model_native_sample_rate=44_100,
        stage_settings=(stage,),
        device=device,
        input_sample_rate=sample_rate,
        separation_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        scoring_sample_rate=sample_rate,
    )


def test_deux_cascade_uses_exact_counterfactual_and_declared_formulas(tmp_path):
    mixture, source = _source(tmp_path)
    original = source.copy()
    seen: list[np.ndarray] = []
    settings = _settings()

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        seen.append(audio.copy())
        if len(seen) == 1:
            return {"Vocals": audio * 0.25, "_deux_inst": audio * 0.5}, settings
        return {"Vocals": audio * 0.75, "_deux_inst": audio * 0.125}, settings

    result = separate_with_deux_cascade(str(mixture), separate)

    expected_counterfactual = (source - np.float32(0.5) * (source * 0.5)).astype(
        np.float32
    )
    np.testing.assert_array_equal(seen[0], source)
    np.testing.assert_array_equal(seen[1], expected_counterfactual)
    np.testing.assert_array_equal(source, original)
    v0 = source * 0.25
    v1 = expected_counterfactual * 0.75
    np.testing.assert_array_equal(
        result.stems["Vocals"], (v0 * 0.5 + v1 * 0.5).astype(np.float32)
    )
    np.testing.assert_array_equal(
        result.stems["_deux_inst"], (source - result.stems["Vocals"]).astype(np.float32)
    )
    assert [arm.arm_id for arm in result.arms] == [
        "independent-deux",
        "complementary-v0",
        "counterfactual-v1",
        "refined-complement",
        "fixed-half-recipe",
    ]
    assert set(result.stems) == {"Vocals", "_deux_inst"}


def test_deux_cascade_retains_leakage_in_i0_only_at_half_formula(tmp_path):
    mixture, source = _source(tmp_path)
    leaked = np.full_like(source, 0.2)
    settings = _settings()
    seen: list[np.ndarray] = []

    def separate(path: str):
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        seen.append(audio.copy())
        if len(seen) == 1:
            return {"Vocals": source.copy(), "_deux_inst": leaked.copy()}, settings
        return {"Vocals": audio.copy(), "_deux_inst": np.zeros_like(audio)}, settings

    separate_with_deux_cascade(str(mixture), separate)

    np.testing.assert_array_equal(seen[1], source - np.float32(0.5) * leaked)


@pytest.mark.parametrize(
    ("bad_outputs", "match"),
    [
        ({"Vocals": np.zeros((8, 2), dtype=np.float32)}, "exactly"),
        (
            {
                "Vocals": np.zeros((8, 2), dtype=np.float32),
                "_deux_inst": np.zeros((7, 2), dtype=np.float32),
            },
            "shape",
        ),
        (
            {
                "Vocals": np.zeros((8, 1), dtype=np.float32),
                "_deux_inst": np.zeros((8, 1), dtype=np.float32),
            },
            "stereo",
        ),
        (
            {
                "Vocals": np.full((8, 2), np.nan, dtype=np.float32),
                "_deux_inst": np.zeros((8, 2), dtype=np.float32),
            },
            "finite",
        ),
    ],
)
def test_deux_cascade_validates_stereo_shape_finite_and_exact_stems(
    tmp_path, bad_outputs, match
):
    mixture, _ = _source(tmp_path)

    def separate(_path: str):
        return bad_outputs, _settings()

    with pytest.raises(ValueError, match=match):
        separate_with_deux_cascade(str(mixture), separate)


def test_deux_cascade_requires_same_settings_and_sample_rate(tmp_path):
    mixture, _ = _source(tmp_path)
    calls = 0

    def changing_settings(_path: str):
        nonlocal calls
        calls += 1
        return {
            "Vocals": np.zeros((8, 2), dtype=np.float32),
            "_deux_inst": np.zeros((8, 2), dtype=np.float32),
        }, replace(_settings(), device=f"device-{calls}")

    with pytest.raises(ValueError, match="identical RunSettings"):
        separate_with_deux_cascade(str(mixture), changing_settings)

    def wrong_rate(_path: str):
        return {
            "Vocals": np.zeros((8, 2), dtype=np.float32),
            "_deux_inst": np.zeros((8, 2), dtype=np.float32),
        }, _settings(48_000)

    with pytest.raises(ValueError, match="sample rate"):
        separate_with_deux_cascade(str(mixture), wrong_rate)


def test_deux_cascade_rejects_non_frozen_observed_settings(tmp_path):
    mixture, source = _source(tmp_path)
    settings = _settings()
    bad_settings = (
        replace(settings, segment_size=1600),
        replace(
            settings,
            stage_settings=(
                replace(settings.stage_settings[0], checkpoint_sha256="wrong"),
            ),
        ),
        replace(settings, tta=True),
    )

    for bad in bad_settings:
        with pytest.raises(ValueError, match="frozen settings"):
            separate_with_deux_cascade(
                str(mixture),
                lambda _path, bad=bad: (
                    {
                        "Vocals": source.copy(),
                        "_deux_inst": np.zeros_like(source),
                    },
                    bad,
                ),
            )


def test_deux_cascade_conserves_source_in_float32(tmp_path):
    mixture, source = _source(tmp_path)
    settings = _settings()

    def separate(_path: str):
        return {
            "Vocals": np.full_like(source, 0.13),
            "_deux_inst": np.full_like(source, 0.77),
        }, settings

    result = separate_with_deux_cascade(str(mixture), separate)
    assert isinstance(result, CascadeEvaluationResult)
    for arm in result.arms:
        assert set(arm.stems) == {"Vocals", "_deux_inst"}
        assert all(audio.dtype == np.float32 for audio in arm.stems.values())
        assert all(np.isfinite(audio).all() for audio in arm.stems.values())
        if arm.arm_id != "independent-deux" and arm.arm_id != "counterfactual-v1":
            np.testing.assert_allclose(
                arm.stems["Vocals"] + arm.stems["_deux_inst"], source, atol=1e-6, rtol=0
            )
    np.testing.assert_allclose(
        result.stems["Vocals"] + result.stems["_deux_inst"], source, atol=1e-6, rtol=0
    )


def test_deux_cascade_provenance_and_report_field(tmp_path):
    mixture, source = _source(tmp_path)
    reference = tmp_path / "vocals.wav"
    instrumental = tmp_path / "instrumental.wav"
    sf.write(reference, source, 44_100, subtype="FLOAT")
    sf.write(instrumental, source * 0.2, 44_100, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Vocals": str(reference), "Instrumental": str(instrumental)},
                estimate_stems={"Instrumental": ("_deux_inst",)},
                recording_id="recording-0",
                item_id="item-0",
            )
        ],
        corpus_id="q40-test-v1",
    )
    settings = _settings()

    def separate(_path: str):
        return {"Vocals": source.copy(), "_deux_inst": np.zeros_like(source)}, settings

    result = separate_with_deux_cascade(str(mixture), separate)
    provenance = result.provenance()
    assert provenance["recipe"] == "q40-deux-counterfactual-half-v1"
    assert provenance["separation_call_count"] == 2
    assert len(provenance["arms"]) == 5
    assert provenance["memory_scope"] == "parent_process_lifetime_peak_rss"
    runtimes = [arm["runtime_s"] for arm in provenance["arms"]]
    assert runtimes == sorted(runtimes)
    assert provenance["control_runtime_s"] == runtimes[1]
    assert provenance["candidate_runtime_s"] == runtimes[-1]
    assert provenance["candidate_to_control_runtime_ratio"] >= 1
    report = evaluate_corpus(
        corpus, lambda path: separate_with_deux_cascade(path, separate), 44_100
    )
    payload = report.to_dict()
    assert len(payload["cascade_provenance"]) == 1
    assert len(payload["cascade_arm_scores"]) == 10
    assert {row["arm_id"] for row in payload["cascade_arm_scores"]} == {
        "independent-deux",
        "complementary-v0",
        "counterfactual-v1",
        "refined-complement",
        "fixed-half-recipe",
    }
    assert {row["stem"] for row in payload["cascade_arm_scores"]} == {
        "Vocals",
        "Instrumental",
    }
    assert all(
        {"arm_id", "role", "input", "stem", "category", "sdr", "fullness", "bleedless"}
        <= row.keys()
        for row in payload["cascade_arm_scores"]
    )
    text = format_report(report)
    assert "Cascade arm scores" in text
    assert "fixed-half-recipe" in text
    assert "origin_provenance" not in payload
    ordinary = EvalReport(
        settings=settings,
        scores=[StemScore("Vocals", "default", 1.0, 1.0, 1.0, "recording-0", "item-0")],
    ).to_dict()
    assert "cascade_provenance" not in ordinary
    assert "cascade_arm_scores" not in ordinary
    assert (
        json.loads(EvalReport(settings=settings, scores=[]).to_json())
        == EvalReport(settings=settings, scores=[]).to_dict()
    )


def test_runner_validates_and_wires_frozen_cascade(tmp_path):
    runner = _runner()
    mixture, _ = _source(tmp_path, sample_rate=44_100)
    args = runner._parser().parse_args(
        [
            "--corpus",
            "licensed",
            "--variant",
            "real-model",
            "--output-dir",
            str(tmp_path),
            "--cascade-vocal-repair",
            "--model",
            "becruily_deux.ckpt",
        ]
    )
    captured = {}

    def fake_separator(path: str, **options):
        captured.update(options)
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        return {
            "Vocals": audio,
            "_deux_inst": np.zeros_like(audio),
        }, _settings(44_100)

    runner.separate_for_eval = fake_separator
    separator = runner._real_separator(args)
    assert callable(separator)
    separator(str(mixture))
    assert captured == {
        "model": "becruily_deux.ckpt",
        "sample_rate": 44_100,
        "batch_size": 1,
        "segment_size": None,
        "chunk_duration_s": None,
        "overlap": 2,
        "stem_ensemble": False,
        "tta": False,
        "pitch_shift": None,
    }


def test_runner_retains_all_cascade_arm_paths_and_provenance(tmp_path):
    runner = _runner()
    mixture, source = _source(tmp_path)
    reference = tmp_path / "vocals.wav"
    sf.write(reference, source, 44_100, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [CorpusItem(mixture=str(mixture), stems={"Vocals": str(reference)})],
        corpus_id="q40-retention-v1",
    )
    settings = _settings()

    def separate(_path: str):
        return {
            "Vocals": source.copy(),
            "_deux_inst": np.zeros_like(source),
        }, settings

    output_dir = tmp_path / "report"
    retaining = runner._retaining_separator(
        lambda path: separate_with_deux_cascade(path, separate),
        corpus,
        output_dir,
        44_100,
    )
    result = retaining(str(mixture))
    index = json.loads((output_dir / "stems/index.json").read_text())
    entry = index["items"][0]
    assert [arm["arm_id"] for arm in entry["cascade_arms"]] == [
        "independent-deux",
        "complementary-v0",
        "counterfactual-v1",
        "refined-complement",
        "fixed-half-recipe",
    ]
    for arm in result.provenance()["arms"]:
        path = Path(arm["output_paths"]["Vocals"])
        assert (output_dir / path).exists()
        assert path.parts[:3] == ("stems", "0000", "arms")
    assert entry["stems"] == {
        "_deux_inst": "stems/0000/_deux_inst.wav",
        "Vocals": "stems/0000/Vocals.wav",
    }


@pytest.mark.parametrize(
    "option",
    [
        ["--sample-rate", "48000"],
        ["--model", "BS-Roformer-SW.ckpt"],
        ["--batch-size", "2"],
        ["--overlap", "4"],
        ["--segment-size", "1601"],
        ["--chunk-duration-s", "60"],
        ["--tta"],
        ["--pitch-shift", "1"],
        ["--rate-arm", "delivery"],
        ["--stem-ensemble"],
        ["--extra-origin-samples", "1"],
    ],
)
def test_runner_rejects_non_frozen_cascade_options(tmp_path, capsys, option):
    runner = _runner()
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "licensed",
                "--variant",
                "real-model",
                "--output-dir",
                str(tmp_path / "report"),
                "--cascade-vocal-repair",
                "--model",
                "becruily_deux.ckpt",
                "--retain-stems",
                *option,
            ]
        )
    assert "cascade-vocal-repair" in capsys.readouterr().err


def test_runner_requires_retained_stems_for_cascade(tmp_path, capsys):
    runner = _runner()
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "real-model",
                "--output-dir",
                str(tmp_path / "report"),
                "--cascade-vocal-repair",
                "--model",
                "becruily_deux.ckpt",
            ]
        )
    assert "requires --retain-stems" in capsys.readouterr().err


def test_runner_validates_frozen_corpus_identity(tmp_path, monkeypatch, capsys):
    runner = _runner()
    manifest = tmp_path / "corpus.json"
    manifest.write_text("frozen", encoding="utf-8")
    paths = {
        "mixture": tmp_path / "mixture.wav",
        "Vocals": tmp_path / "vocals.wav",
        "Instrumental": tmp_path / "instrumental.wav",
    }
    item = CorpusItem(
        mixture=str(paths["mixture"]),
        stems={
            "Vocals": str(paths["Vocals"]),
            "Instrumental": str(paths["Instrumental"]),
        },
        category="q40-deux-counterfactual-half",
        recording_id="musdb18-hq/Hollow Ground - Ill Fate",
        item_id="musdb18-hq/tuning/Hollow Ground - Ill Fate#60s-72s",
        split="tuning",
        estimate_stems={"Instrumental": ("_deux_inst",)},
    )
    corpus = ReferenceCorpus(
        [item], corpus_id="upmixer-musdb18hq-v1-q40-deux-counterfactual-half-v1"
    )
    hashes = {str(manifest): runner._CASCADE_CORPUS_SHA256}
    hashes.update(
        {str(paths[name]): value for name, value in runner._CASCADE_FILE_SHA256.items()}
    )
    monkeypatch.setattr(runner, "_sha256_file", lambda path: hashes[str(path)])
    monkeypatch.setattr(
        runner.sf,
        "info",
        lambda _path: SimpleNamespace(samplerate=44_100, channels=2, frames=529_200),
    )
    runner._validate_frozen_cascade_corpus(corpus, tmp_path, runner._parser())

    with pytest.raises(SystemExit):
        runner._validate_frozen_cascade_corpus(
            replace(corpus, corpus_id="wrong"), tmp_path, runner._parser()
        )
    assert "requires corpus ID" in capsys.readouterr().err
