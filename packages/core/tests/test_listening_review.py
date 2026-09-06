"""Checks for the Q20 listening review server."""

from __future__ import annotations

import csv
import importlib.util
import json
from http.client import HTTPResponse
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_listening_review.py"
_SPEC = importlib.util.spec_from_file_location("upmixer_listening_review", _SCRIPT)
assert _SPEC and _SPEC.loader
_REVIEW = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REVIEW)

_COLUMNS = [
    "listener_id",
    "round_id",
    "case_id",
    "preference",
    "confidence_1_5",
    "fullness_a_0_3",
    "quality_0_3",
    "comparison_gain_db",
    "original_level_check",
    "notes",
]


def _pack(tmp_path: Path) -> Path:
    (tmp_path / "audio").mkdir(parents=True)
    (tmp_path / "audio" / "a.wav").write_bytes(b"AUDIO-A")
    (tmp_path / "audio" / "b.wav").write_bytes(b"AUDIO-B")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-a",
                        "category": "flag",
                        "audio": {"A": {"path": "audio/a.wav"}},
                    },
                    {"case_id": "case-b", "audio": {"B": {"path": "audio/b.wav"}}},
                ]
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "ratings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerow({"case_id": "case-a"})
        writer.writerow(
            {
                "listener_id": "saved-listener",
                "round_id": "saved-round",
                "case_id": "case-b",
                "preference": "B",
                "confidence_1_5": "4",
                "fullness_a_0_3": "1",
                "quality_0_3": "2",
                "comparison_gain_db": "0.5",
                "original_level_check": "yes",
                "notes": "keep",
            }
        )
    return tmp_path


def _post(base: str, payload: dict[str, str]) -> HTTPResponse:
    request = Request(
        f"{base}/api/save",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request)


def test_review_serves_manifest_audio_and_persists_replacements(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    server = _REVIEW.create_server(pack, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    payload = {
        "listener_id": "listener-1",
        "round_id": "round-1",
        "case_id": "case-a",
        "preference": "A",
        "confidence_1_5": "5",
        "fullness_a_0_3": "3",
        "quality_0_3": "0",
        "comparison_gain_db": "-1.25",
        "original_level_check": "matched",
        "notes": "first",
    }
    try:
        page = urlopen(f"{base}/").read().decode("utf-8")
        state = json.loads(urlopen(f"{base}/api/state").read())
        assert "rows" not in state
        assert all(
            value not in page for value in ("saved-listener", "saved-round", "keep")
        )
        assert "/audio/case-a/A" in page and "/audio/case-b/B" in page
        assert all(column in page for column in _COLUMNS if column != "case_id")
        assert urlopen(f"{base}/audio/case-a/A").read() == b"AUDIO-A"
        with pytest.raises(HTTPError) as missing:
            urlopen(f"{base}/audio/case-a/B")
        assert missing.value.code == 404

        assert json.loads(_post(base, payload).read()) == payload
        payload["preference"] = "TIE"
        _post(base, payload).read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    rows = _REVIEW.ReviewStore(pack).rows()
    assert len(rows) == 2
    assert (
        next(row for row in rows if row["case_id"] == "case-a")["preference"] == "TIE"
    )
    assert next(row for row in rows if row["case_id"] == "case-b")["notes"] == "keep"


def test_review_rejects_invalid_values_and_escaping_audio(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"outside")
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    manifest["cases"][0]["audio"]["outside"] = {"path": "../outside.wav"}
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes pack"):
        _REVIEW.ReviewStore(pack)

    store = _REVIEW.ReviewStore(_pack(tmp_path / "valid"))
    valid = {
        "listener_id": "listener",
        "round_id": "round",
        "case_id": "case-a",
        "preference": "A",
        "confidence_1_5": "3",
        "fullness_a_0_3": "0",
        "quality_0_3": "1",
        "comparison_gain_db": "0",
        "original_level_check": "yes",
    }
    for field, value in (
        ("preference", "C"),
        ("confidence_1_5", "6"),
        ("fullness_a_0_3", "4"),
        ("quality_0_3", "1.5"),
        ("comparison_gain_db", "nan"),
        ("original_level_check", " "),
    ):
        candidate = valid | {field: value}
        with pytest.raises(ValueError):
            store.save(candidate)
