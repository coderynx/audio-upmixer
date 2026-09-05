"""Subprocess coverage for the offline evaluation runner."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from upmixer.eval import CorpusItem, ReferenceCorpus, RunSettings


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_eval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("upmixer_run_eval", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(output_dir: Path, *options: str) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--corpus",
        "synthetic",
        "--variant",
        "synthetic-reference",
        "--sample-rate",
        "22050",
        "--output-dir",
        str(output_dir),
        *options,
    ]


def _run(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(output_dir),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_synthetic_runner_writes_reproducible_report_and_text(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_result = _run(first_dir)
    second_result = _run(second_dir)

    first_payload = json.loads((first_dir / "report.json").read_text(encoding="utf-8"))
    second_payload = json.loads((second_dir / "report.json").read_text(encoding="utf-8"))
    assert first_payload == second_payload
    assert first_payload["schema_version"] == 1
    assert first_payload["settings"]["model"] == "synthetic-reference"
    assert first_payload["protocol_id"] == "upmixer-separation-q00-v1"
    assert first_payload["corpus_id"] == "upmixer-synthetic-v1"
    assert first_payload["code_revision"]
    assert first_payload["scores"]

    report_text = (first_dir / "report.txt").read_text(encoding="utf-8")
    assert all(metric in report_text for metric in ("SDR", "fullness", "bleedless"))
    assert "synthetic-reference" in first_result.stdout
    assert "synthetic-reference" in second_result.stdout

    refused = subprocess.run(
        _command(first_dir),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "fresh and empty" in refused.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        (option, value)
        for option in ("--chunk-duration-s", "--pitch-shift")
        for value in ("0", "nan", "inf")
    ],
)
def test_runner_rejects_non_finite_or_non_positive_settings(
    tmp_path: Path, option: str, value: str
):
    result = subprocess.run(
        _command(tmp_path / f"{option[2:].replace('-', '_')}-{value}", option, value),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"argument {option}" in result.stderr
    assert "finite positive number" in result.stderr


def test_production_tree_routes_runner_options_without_model_loading(tmp_path):
    runner = _load_runner()
    args = runner._parser().parse_args(
        [
            "--corpus",
            "synthetic",
            "--variant",
            "production-tree",
            "--sample-rate",
            "48000",
            "--output-dir",
            str(tmp_path),
            "--batch-size",
            "2",
            "--segment-size",
            "128",
            "--chunk-duration-s",
            "30",
            "--overlap",
            "4",
            "--stem-ensemble",
            "--tta",
            "--pitch-shift",
            "0.75",
            "--stems",
            "vocals,bass",
        ]
    )
    args.stems = runner.normalize_stems(args.stems)
    captured = {}

    def fake_tree(mixture_path, sample_rate, config):
        captured.update(
            mixture_path=mixture_path,
            sample_rate=sample_rate,
            config=config,
        )
        return {}, object()

    runner.separate_tree_for_eval = fake_tree
    runner._real_separator(args)("mix.wav")

    config = captured["config"]
    assert captured["mixture_path"] == "mix.wav"
    assert captured["sample_rate"] == 48000
    assert config.output_sample_rate == 48000
    assert config.stems == ["Vocals", "Bass"]
    assert config.stem_batch_size == 2
    assert config.stem_segment_size == 128
    assert config.stem_chunk_duration_s == 30.0
    assert config.stem_overlap == 4
    assert config.stem_ensemble is True
    assert config.stem_tta is True
    assert config.stem_pitch_shift == 0.75


def test_production_tree_rejects_model_option(tmp_path, capsys):
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "production-tree",
                "--model",
                "BS-Roformer-SW.ckpt",
                "--output-dir",
                str(tmp_path / "model"),
            ]
        )

    assert "does not accept --model" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


def test_stems_option_is_restricted_to_production_tree(tmp_path, capsys):
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "synthetic-reference",
                "--stems",
                "vocals",
                "--output-dir",
                str(tmp_path / "stems"),
            ]
        )

    assert "requires --variant production-tree" in capsys.readouterr().err
    assert not (tmp_path / "stems").exists()


def test_unknown_stem_is_rejected_before_output_or_corpus_creation(tmp_path, capsys):
    runner = _load_runner()
    output_dir = tmp_path / "unknown"

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "production-tree",
                "--stems",
                "vocals,theremin",
                "--output-dir",
                str(output_dir),
            ]
        )

    assert "Unknown stem name 'theremin'" in capsys.readouterr().err
    assert not output_dir.exists()


def _retention_corpus(tmp_path: Path) -> ReferenceCorpus:
    items = []
    for index, directory in enumerate(("one", "two", "later")):
        reference = tmp_path / directory / "vocals.wav"
        reference.parent.mkdir(parents=True)
        samples = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
        sf.write(reference, np.column_stack([samples, samples]), 22_050, subtype="FLOAT")
        items.append(
            CorpusItem(
                mixture=str(tmp_path / directory / "mix.wav"),
                stems={"Vocals": str(reference)},
                recording_id=f"recording-{index}",
                item_id=f"item-{index}",
                split="tuning",
            )
        )
    return ReferenceCorpus(items, corpus_id="retention-test-v1")


def test_retain_stems_indexes_completed_items_without_copying_corpus_audio(tmp_path, monkeypatch):
    runner = _load_runner()
    corpus = _retention_corpus(tmp_path)
    monkeypatch.setattr(
        runner.ReferenceCorpus,
        "from_dir",
        classmethod(lambda _cls, _path: corpus),
    )

    def fake_separator(mixture_path):
        item = next(item for item in corpus.items if item.mixture == mixture_path)
        if item.item_id == "item-2":
            raise RuntimeError("synthetic later-item failure")
        audio, _ = sf.read(item.stems["Vocals"], dtype="float64", always_2d=True)
        return {"Vocals": audio}, RunSettings(model="fake", sample_rate=22_050)

    monkeypatch.setattr(runner, "_real_separator", lambda _args: fake_separator)
    output_dir = tmp_path / "report"
    return_code = runner.main(
        [
            "--corpus",
            "licensed-test",
            "--variant",
            "real-model",
            "--sample-rate",
            "22050",
            "--output-dir",
            str(output_dir),
            "--model",
            "fake",
            "--retain-stems",
        ]
    )

    assert return_code == 1
    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert [item["index"] for item in index["items"]] == [0, 1]
    assert [item["item_id"] for item in index["items"]] == ["item-0", "item-1"]
    assert [item["stems"] for item in index["items"]] == [
        {"Vocals": "stems/0000/Vocals.wav"},
        {"Vocals": "stems/0001/Vocals.wav"},
    ]
    assert index["items"][0]["recording_id"] == "recording-0"
    for item in index["items"]:
        retained = output_dir / item["stems"]["Vocals"]
        info = sf.info(retained)
        assert info.samplerate == 22_050
        assert info.subtype == "FLOAT"
        audio, rate = sf.read(retained, dtype="float32", always_2d=True)
        assert rate == 22_050
        assert audio.shape == (32, 2)

    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == {
        "report.json",
        "report.txt",
        "stems/index.json",
        "stems/0000/Vocals.wav",
        "stems/0001/Vocals.wav",
    }


def test_retain_stems_indexes_mapped_components_without_target(
    tmp_path: Path, monkeypatch
):
    runner = _load_runner()
    reference = tmp_path / "reference.wav"
    samples = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
    sf.write(reference, np.column_stack([samples, samples]), 22_050, subtype="FLOAT")
    components = ("Bass", "Drums", "Guitar", "Other", "Piano", "Vocals")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(tmp_path / "mix.wav"),
                stems={"CompleteLeafSum": str(reference)},
                estimate_stems={"CompleteLeafSum": list(components)},
                recording_id="recording-0",
                item_id="item-0",
                split="tuning",
            )
        ],
        corpus_id="mapped-retention-test-v1",
    )
    monkeypatch.setattr(
        runner.ReferenceCorpus,
        "from_dir",
        classmethod(lambda _cls, _path: corpus),
    )

    def fake_separator(_mixture_path):
        audio, _ = sf.read(reference, dtype="float32", always_2d=True)
        return {
            component: audio.copy()
            for component in components
        }, RunSettings(model="fake", sample_rate=22_050)

    monkeypatch.setattr(runner, "_real_separator", lambda _args: fake_separator)
    output_dir = tmp_path / "mapped-report"
    assert (
        runner.main(
            [
                "--corpus",
                "licensed-test",
                "--variant",
                "real-model",
                "--sample-rate",
                "22050",
                "--output-dir",
                str(output_dir),
                "--model",
                "fake",
                "--retain-stems",
            ]
        )
        == 0
    )

    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert set(index["items"][0]["stems"]) == set(components)
    assert "CompleteLeafSum" not in index["items"][0]["stems"]
    for component in components:
        retained = output_dir / index["items"][0]["stems"][component]
        audio, rate = sf.read(retained, dtype="float32", always_2d=True)
        assert rate == 22_050
        assert audio.shape == (32, 2)


def test_retain_stems_is_rejected_for_synthetic_reference(tmp_path, capsys):
    runner = _load_runner()
    output_dir = tmp_path / "synthetic"

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--corpus",
                "synthetic",
                "--variant",
                "synthetic-reference",
                "--output-dir",
                str(output_dir),
                "--retain-stems",
            ]
        )

    assert "--retain-stems requires" in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "bad_audio",
    [
        lambda: np.ones(32, dtype=np.float32),
        lambda: np.full((32, 2), np.nan, dtype=np.float32),
    ],
)
def test_retain_stems_does_not_index_invalid_separator_outputs(
    tmp_path: Path, monkeypatch, bad_audio
):
    runner = _load_runner()
    corpus = _retention_corpus(tmp_path)
    monkeypatch.setattr(
        runner.ReferenceCorpus,
        "from_dir",
        classmethod(lambda _cls, _path: ReferenceCorpus(corpus.items[:1])),
    )
    monkeypatch.setattr(
        runner,
        "_real_separator",
        lambda _args: lambda _path: (
            {"Vocals": bad_audio()},
            RunSettings(model="fake", sample_rate=22_050),
        ),
    )

    output_dir = tmp_path / "invalid"
    assert (
        runner.main(
            [
                "--corpus",
                "licensed-test",
                "--variant",
                "real-model",
                "--sample-rate",
                "22050",
                "--output-dir",
                str(output_dir),
                "--model",
                "fake",
                "--retain-stems",
            ]
        )
        == 1
    )

    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert index["items"] == []
    assert not (output_dir / "stems/0000").exists()


@pytest.mark.parametrize(
    "settings",
    [object(), RunSettings(model="fake", sample_rate=48_000)],
)
def test_retain_stems_does_not_index_invalid_settings(tmp_path: Path, settings):
    runner = _load_runner()
    corpus = _retention_corpus(tmp_path)
    output_dir = tmp_path / "invalid-settings"
    retaining = runner._retaining_separator(
        lambda _path: (
            {"Vocals": np.zeros((32, 2), dtype=np.float32)},
            settings,
        ),
        ReferenceCorpus(corpus.items[:1]),
        output_dir,
        22_050,
    )

    retaining(corpus.items[0].mixture)

    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert index["items"] == []
    assert not (output_dir / "stems/0000").exists()


def test_retain_stems_rejects_mismatched_mixture_identity(tmp_path: Path):
    runner = _load_runner()
    corpus = _retention_corpus(tmp_path)
    output_dir = tmp_path / "wrong-mixture"
    called = False

    def separator(_path):
        nonlocal called
        called = True
        return {}, RunSettings(model="fake", sample_rate=22_050)

    retaining = runner._retaining_separator(
        separator, ReferenceCorpus(corpus.items[:1]), output_dir, 22_050
    )
    with pytest.raises(ValueError, match="does not match corpus item"):
        retaining("some-other-mix.wav")

    assert not called
    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert index["items"] == []


def test_retain_stems_checks_extra_outputs_against_reference_boundary(tmp_path: Path):
    runner = _load_runner()
    corpus = _retention_corpus(tmp_path)
    output_dir = tmp_path / "extra-mismatch"
    retaining = runner._retaining_separator(
        lambda _path: (
            {
                "Vocals": np.zeros((32, 2), dtype=np.float32),
                "Guitar": np.zeros((31, 2), dtype=np.float32),
            },
            RunSettings(model="fake", sample_rate=22_050),
        ),
        ReferenceCorpus(corpus.items[:1]),
        output_dir,
        22_050,
    )

    retaining(corpus.items[0].mixture)

    index = json.loads((output_dir / "stems/index.json").read_text(encoding="utf-8"))
    assert index["items"] == []
    assert not (output_dir / "stems/0000").exists()
