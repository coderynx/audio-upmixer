"""Subprocess coverage for the offline evaluation runner."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    assert config.stems == ["vocals", "bass"]
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
