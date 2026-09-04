"""Subprocess coverage for the offline evaluation runner."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_eval.py"


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
