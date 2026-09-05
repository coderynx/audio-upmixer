"""Q30 evaluation-only distinct time-origin views."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from upmixer.eval import (
    CorpusItem,
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    separate_with_extra_origin,
)
from upmixer.separation.separator import SeparationSettings


def test_extra_origin_is_zero_padded_aligned_stereo_and_exact_length(tmp_path):
    sample_rate = 8_000
    source = np.zeros((11, 2), dtype=np.float32)
    source[0] = (0.25, -0.5)
    source[-1] = (0.75, -0.125)
    source[5] = (-0.25, 0.625)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")
    seen: list[np.ndarray] = []

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == sample_rate
        seen.append(audio)
        return {"Bass": audio}, RunSettings(model="fake", sample_rate=rate)

    result = separate_with_extra_origin(
        str(mixture), separate, origin_samples=3
    )

    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], source)
    np.testing.assert_array_equal(seen[1][:3], np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(seen[1][3:-3], source)
    np.testing.assert_array_equal(seen[1][-3:], np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_allclose(result.stems["Bass"], source, atol=0.0, rtol=0.0)
    assert not hasattr(result.settings, "origin_schedule")
    assert result.provenance()["effective_view_count"] == 2
    assert result.provenance()["separation_call_count"] == 2


def test_report_derives_origin_schedule_and_each_view_output(tmp_path):
    sample_rate = 8_000
    source = np.linspace(-0.25, 0.25, 12, dtype=np.float32)
    stereo = np.column_stack([source, -source])
    mixture = tmp_path / "mixture.wav"
    reference = tmp_path / "bass.wav"
    sf.write(mixture, stereo, sample_rate, subtype="FLOAT")
    sf.write(reference, stereo, sample_rate, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Bass": str(reference)},
                recording_id="recording-0",
                item_id="item-0",
                split="tuning",
            )
        ],
        corpus_id="q30-test-v1",
    )

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {"Bass": audio}, RunSettings(model="fake", sample_rate=rate)

    report = evaluate_corpus(
        corpus,
        lambda path: separate_with_extra_origin(
            path, separate, origin_samples=2
        ),
        sample_rate=sample_rate,
    )

    provenance = report.to_dict()["origin_provenance"]
    assert len(provenance) == 1
    assert provenance[0]["origin_schedule"] == [0, 2]
    assert provenance[0]["effective_view_count"] == 2
    assert provenance[0]["separation_call_count"] == 2
    assert provenance[0]["memory_scope"] == "parent_process_lifetime_peak_rss"
    assert [view["origin_samples"] for view in provenance[0]["views"]] == [0, 2]
    assert all(view["stem_names"] == ["Bass"] for view in provenance[0]["views"])


def test_runner_wires_extra_origin_to_direct_model_separator(tmp_path, monkeypatch):
    runner_path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("q30_eval_runner_options", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sample_rate = 8_000
    source = np.zeros((8, 2), dtype=np.float32)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")

    def fake_separator(path: str, **_kwargs):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {"Bass": audio}, RunSettings(model="fake", sample_rate=rate)

    monkeypatch.setattr(runner, "separate_for_eval", fake_separator)
    args = runner._parser().parse_args(
        [
            "--corpus",
            "licensed",
            "--variant",
            "real-model",
            "--sample-rate",
            str(sample_rate),
            "--output-dir",
            str(tmp_path / "report"),
            "--extra-origin-samples",
            "2",
        ]
    )

    result = runner._real_separator(args)(str(mixture))

    assert not hasattr(result.settings, "origin_schedule")
    assert result.provenance()["separation_call_count"] == 2


def test_runner_records_per_view_outputs_when_retaining_origin_audio(tmp_path, monkeypatch):
    runner_path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("q30_eval_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    sample_rate = 8_000
    source = np.column_stack(
        [np.linspace(-0.25, 0.25, 12), np.linspace(0.25, -0.25, 12)]
    ).astype(np.float32)
    mixture = tmp_path / "mixture.wav"
    reference = tmp_path / "bass.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")
    sf.write(reference, source, sample_rate, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Bass": str(reference)},
                recording_id="recording-0",
                item_id="item-0",
                split="tuning",
            )
        ],
        corpus_id="q30-runner-v1",
    )
    monkeypatch.setattr(
        runner.ReferenceCorpus,
        "from_dir",
        classmethod(lambda _cls, _path: corpus),
    )

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {"Bass": audio}, RunSettings(model="fake", sample_rate=rate)

    monkeypatch.setattr(
        runner,
        "_real_separator",
        lambda _args: lambda path: separate_with_extra_origin(
            path, separate, origin_samples=2
        ),
    )
    output_dir = tmp_path / "report"

    assert (
        runner.main(
            [
                "--corpus",
                "licensed",
                "--variant",
                "real-model",
                "--sample-rate",
                str(sample_rate),
                "--output-dir",
                str(output_dir),
                "--retain-stems",
                "--extra-origin-samples",
                "2",
            ]
        )
        == 0
    )

    payload = json.loads(
        (output_dir / "report.json").read_text(encoding="utf-8")
    )
    views = payload["origin_provenance"][0]["views"]
    assert views[1]["origin_samples"] == 2
    retained = Path(views[1]["output_paths"]["Bass"])
    assert (output_dir / retained).exists()


def test_extra_origin_rejects_changed_effective_settings(tmp_path):
    sample_rate = 8_000
    source = np.zeros((8, 2), dtype=np.float32)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")
    calls = 0

    def separate(path: str):
        nonlocal calls
        calls += 1
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {
            "Bass": audio
        }, RunSettings(model="fake", sample_rate=rate, batch_size=calls)

    with pytest.raises(ValueError, match="extra-origin separation settings differ"):
        separate_with_extra_origin(str(mixture), separate, origin_samples=2)


def test_roformer_origin_accounting_uses_effective_schedule(tmp_path):
    sample_rate = 44_100
    source = np.zeros((529_200, 2), dtype=np.float32)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")
    stage = SeparationSettings(
        model="BS-Roformer-SW.ckpt",
        sample_rate=sample_rate,
        batch_size=1,
        segment_size=1_724,
        chunk_duration_s=None,
        overlap=2,
        tta=False,
        pitch_shift=None,
        backend="cpu",
        model_arch="bs_roformer",
        model_config_name="BS-Roformer-SW",
        model_native_sample_rate=sample_rate,
        device="cpu",
    )
    settings = RunSettings(
        model=stage.model,
        sample_rate=sample_rate,
        segment_size=stage.segment_size,
        overlap=stage.overlap,
        batch_size=stage.batch_size,
        tta=False,
        pitch_shift=None,
        backend=stage.backend,
        model_arch=stage.model_arch,
        model_config_name=stage.model_config_name,
        model_native_sample_rate=stage.model_native_sample_rate,
        stage_settings=(stage,),
        device=stage.device,
        input_sample_rate=sample_rate,
        separation_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        scoring_sample_rate=sample_rate,
    )

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {"Bass": audio}, settings

    result = separate_with_extra_origin(str(mixture), separate, origin_samples=17)
    provenance = result.provenance()

    assert [
        {
            key: view[key]
            for key in (
                "scheduled_windows",
                "evaluated_unique_windows",
                "tail_replay_contributions",
                "model_forward_calls",
            )
        }
        for view in provenance["views"]
    ] == [
        {
            "scheduled_windows": 2,
            "evaluated_unique_windows": 1,
            "tail_replay_contributions": 1,
            "model_forward_calls": 1,
        },
        {
            "scheduled_windows": 2,
            "evaluated_unique_windows": 1,
            "tail_replay_contributions": 1,
            "model_forward_calls": 1,
        },
    ]
    assert provenance["totals"] == {
        "scheduled_windows": 4,
        "evaluated_unique_windows": 2,
        "tail_replay_contributions": 2,
        "model_forward_calls": 2,
    }


def test_origin_accounting_marks_unsupported_context(tmp_path):
    sample_rate = 8_000
    source = np.zeros((8, 2), dtype=np.float32)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, source, sample_rate, subtype="FLOAT")

    def separate(path: str):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return {"Bass": audio}, RunSettings(model="fake", sample_rate=rate)

    result = separate_with_extra_origin(str(mixture), separate, origin_samples=2)
    view = result.provenance()["views"][0]
    assert view["scheduled_windows"] is None
    assert "unsupported" in view["unsupported_reason"]


@pytest.mark.parametrize(
    "extra_option", [["--tta"], ["--pitch-shift", "1.1"]]
)
def test_runner_blocks_origin_with_augmentation(extra_option, tmp_path, capsys):
    runner_path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("q30_eval_runner_guard", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "real-model",
                "--output-dir",
                str(tmp_path / "report"),
                "--extra-origin-samples",
                "2",
                *extra_option,
            ]
        )
    assert "cannot be combined" in capsys.readouterr().err
