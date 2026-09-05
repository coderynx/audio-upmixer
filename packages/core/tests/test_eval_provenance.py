"""Provenance identifiers for evaluation corpora and reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from upmixer.eval.corpus import CorpusItem, ReferenceCorpus, synthetic_corpus
from upmixer.eval.harness import RunSettings, evaluate_corpus
from upmixer.eval.report import format_report


def _load_runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("upmixer_eval_runner_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_id_defaults_and_manifest_override(tmp_path):
    assert ReferenceCorpus([]).corpus_id is None

    manifest = {
        "corpus_id": "licensed-v1",
        "items": [
            {
                "mixture": "mix.wav",
                "stems": {"Vocals": "vocals.wav"},
            }
        ],
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert ReferenceCorpus.from_dir(str(tmp_path)).corpus_id == "licensed-v1"


def test_synthetic_corpus_has_versioned_corpus_id(tmp_path):
    corpus = synthetic_corpus(sample_rate=8000, out_dir=str(tmp_path / "synthetic"))

    assert corpus.corpus_id == "upmixer-synthetic-v1"


def test_evaluate_corpus_copies_and_renders_provenance(tmp_path):
    signal = np.ones((8, 2), dtype=np.float32)
    mixture = tmp_path / "mix.wav"
    reference = tmp_path / "vocals.wav"
    sf.write(mixture, signal, 8000, subtype="FLOAT")
    sf.write(reference, signal, 8000, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Vocals": str(reference)},
                recording_id="recording-0",
                item_id="item-0",
                split="holdout",
            )
        ],
        corpus_id="licensed-v1",
    )
    settings = RunSettings(model="incumbent", sample_rate=8000)

    def separate(_mixture_path):
        estimate, _ = sf.read(reference, dtype="float32", always_2d=True)
        return {"Vocals": estimate}, settings

    report = evaluate_corpus(
        corpus,
        separate,
        sample_rate=8000,
        protocol_id="protocol-v1",
        code_revision="abc123-dirty",
    )

    assert report.protocol_id == "protocol-v1"
    assert report.corpus_id == "licensed-v1"
    assert report.code_revision == "abc123-dirty"
    payload = report.to_dict()
    assert {
        key: payload[key]
        for key in ("protocol_id", "corpus_id", "code_revision")
    } == {
        "protocol_id": "protocol-v1",
        "corpus_id": "licensed-v1",
        "code_revision": "abc123-dirty",
    }
    text = format_report(report)
    assert "Protocol: protocol-v1" in text
    assert "Corpus: licensed-v1" in text
    assert "Code revision: abc123-dirty" in text


def test_runner_revision_returns_none_when_git_is_unavailable():
    runner = _load_runner()

    with patch.object(runner.subprocess, "run", side_effect=OSError("git missing")):
        assert runner._git_revision() is None


def test_runner_revision_marks_dirty_head():
    runner = _load_runner()
    responses = [
        SimpleNamespace(stdout="0123456789abcdef\n"),
        SimpleNamespace(stdout=" M changed.py\n"),
    ]

    with patch.object(runner.subprocess, "run", side_effect=responses):
        assert runner._git_revision() == "0123456789abcdef-dirty"


def test_runner_captures_revision_before_creating_output(tmp_path):
    runner = _load_runner()
    output_dir = tmp_path / "eval"
    observed = {}

    def fake_revision():
        observed["output_exists_at_revision"] = output_dir.exists()
        return "0123456789abcdef"

    def fake_evaluate(*_args, **kwargs):
        observed["code_revision"] = kwargs["code_revision"]
        return SimpleNamespace(write_json=lambda _path: None)

    with (
        patch.object(runner, "_git_revision", side_effect=fake_revision),
        patch.object(
            runner,
            "synthetic_corpus",
            return_value=SimpleNamespace(items=[]),
        ),
        patch.object(runner, "evaluate_corpus", side_effect=fake_evaluate),
        patch.object(runner, "format_report", return_value="report"),
    ):
        assert runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "synthetic-reference",
                "--output-dir",
                str(output_dir),
            ]
        ) == 0

    assert observed == {
        "output_exists_at_revision": False,
        "code_revision": "0123456789abcdef",
    }
