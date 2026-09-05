"""Failure and coverage reporting for evaluation runs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from upmixer.eval import (
    CoverageRow,
    EvalReport,
    EvaluationSkipped,
    RunSettings,
    evaluate_corpus,
    format_report,
)
from upmixer.eval.corpus import CorpusItem, ReferenceCorpus


def _corpus(tmp_path: Path) -> ReferenceCorpus:
    signal = np.ones((8, 2), dtype=np.float32)
    items: list[CorpusItem] = []
    for item_id, stems in (
        ("partial", ("Vocals", "Bass")),
        ("failed", ("Vocals",)),
        ("skipped", ("Vocals",)),
    ):
        mixture = tmp_path / f"{item_id}-mix.wav"
        sf.write(mixture, signal, 8000, subtype="FLOAT")
        reference_paths = {}
        for stem in stems:
            reference = tmp_path / f"{item_id}-{stem.lower()}.wav"
            sf.write(reference, signal, 8000, subtype="FLOAT")
            reference_paths[stem] = str(reference)
        items.append(
            CorpusItem(
                mixture=str(mixture),
                stems=reference_paths,
                unavailable_stems=("Crowd",) if item_id == "failed" else (),
                recording_id=f"recording-{item_id}",
                item_id=item_id,
                split="holdout",
            )
        )
    return ReferenceCorpus(items)


def _settings() -> RunSettings:
    return RunSettings(model="incumbent", sample_rate=8000)


def test_reporting_mode_records_failed_absent_skipped_and_unavailable_rows(tmp_path):
    corpus = _corpus(tmp_path)
    signal, _ = sf.read(
        corpus.items[0].stems["Vocals"], dtype="float32", always_2d=True
    )

    def separate(mixture: str):
        item_id = next(item.item_id for item in corpus.items if item.mixture == mixture)
        if item_id == "failed":
            raise RuntimeError("model weights unavailable")
        if item_id == "skipped":
            raise EvaluationSkipped("CUDA hardware unavailable")
        return {"Vocals": signal}, _settings()

    report = evaluate_corpus(
        corpus,
        separate,
        sample_rate=8000,
        report_failures=True,
    )

    assert report.settings is None
    assert [score.item_id for score in report.scores] == ["partial"]
    assert [(row.item_id, row.stem, row.status) for row in report.coverage] == [
        ("partial", "Vocals", "scored"),
        ("partial", "Bass", "absent"),
        ("failed", "Vocals", "failed"),
        ("failed", "Crowd", "unavailable"),
        ("skipped", "Vocals", "skipped"),
    ]
    assert report.coverage[1].detail == "missing from separator output"
    assert report.coverage[2].detail == "model weights unavailable"
    assert report.coverage[4].detail == "CUDA hardware unavailable"
    payload = report.to_dict()
    assert payload["coverage"][2]["detail"] == "model weights unavailable"
    text = format_report(report)
    assert "Coverage: total=5" in text
    assert "  failed: 1" in text
    assert "  absent: 1" in text
    assert "  skipped: 1" in text
    assert "failed recording_id=recording-failed item_id=failed" in text


def test_reporting_mode_marks_per_stem_validation_error_and_continues(tmp_path):
    signal = np.ones((8, 2), dtype=np.float32)
    paths = {}
    for name in ("mix", "vocals", "bass"):
        path = tmp_path / f"{name}.wav"
        sf.write(path, signal, 8000, subtype="FLOAT")
        paths[name] = str(path)
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                paths["mix"],
                {"Vocals": paths["vocals"], "Bass": paths["bass"]},
                recording_id="recording-0",
                item_id="item-0",
            )
        ]
    )

    def separate(_mixture: str):
        return {
            "Vocals": signal,
            "Bass": np.full((7, 2), 0.5, dtype=np.float32),
        }, _settings()

    report = evaluate_corpus(
        corpus,
        separate,
        sample_rate=8000,
        report_failures=True,
    )

    assert [score.stem for score in report.scores] == ["Vocals"]
    assert [(row.stem, row.status) for row in report.coverage] == [
        ("Vocals", "scored"),
        ("Bass", "failed"),
    ]
    assert "frame count mismatch" in (report.coverage[1].detail or "")


def test_reporting_mode_marks_empty_outputs_absent(tmp_path):
    corpus = _corpus(tmp_path)
    report = evaluate_corpus(
        ReferenceCorpus([corpus.items[0]]),
        lambda _mixture: ({}, _settings()),
        sample_rate=8000,
        report_failures=True,
    )

    assert [(row.stem, row.status) for row in report.coverage] == [
        ("Vocals", "absent"),
        ("Bass", "absent"),
    ]
    assert all(row.detail == "missing from separator output" for row in report.coverage)


def test_typed_skip_propagates_in_strict_mode(tmp_path):
    corpus = _corpus(tmp_path)

    def separate(_mixture: str):
        raise EvaluationSkipped("no accelerator")

    with pytest.raises(EvaluationSkipped, match="no accelerator"):
        evaluate_corpus(corpus, separate, sample_rate=8000)


def _load_runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("upmixer_eval_runner_failure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_report(status: str) -> EvalReport:
    return EvalReport(
        settings=None,
        scores=[],
        coverage=[
            CoverageRow(
                stem="Vocals",
                category="default",
                status=status,
                recording_id="recording-0",
                item_id="item-0",
                detail="hardware unavailable" if status == "skipped" else "failed",
            )
        ],
    )


@pytest.mark.parametrize(("status", "expected_code"), [("failed", 1), ("skipped", 0)])
def test_runner_writes_report_and_uses_nonzero_only_for_required_failures(
    tmp_path: Path, status: str, expected_code: int
):
    runner = _load_runner()
    output_dir = tmp_path / status
    observed: dict[str, object] = {}

    def fake_evaluate(*_args, **kwargs):
        observed["report_failures"] = kwargs["report_failures"]
        return _runner_report(status)

    with (
        patch.object(runner, "evaluate_corpus", side_effect=fake_evaluate),
        patch.object(runner, "format_report", side_effect=format_report),
    ):
        code = runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "synthetic-reference",
                "--sample-rate",
                "8000",
                "--output-dir",
                str(output_dir),
            ]
        )

    assert code == expected_code
    assert observed == {"report_failures": True}
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["coverage"][0]["status"] == status
    assert (output_dir / "report.txt").is_file()


def test_strict_defaults_remain_fail_fast(tmp_path):
    corpus = _corpus(tmp_path)

    def separate(_mixture: str):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_corpus(corpus, separate, sample_rate=8000)
