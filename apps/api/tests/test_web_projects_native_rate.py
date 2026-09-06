"""Q21 native-rate project delivery integration."""

import io
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

import soundfile as sf

from _helpers import InProcessJobRun, _wav_bytes


def test_native_rate_project_prepares_delivers_and_exports(
    web_client, in_process_jobs, monkeypatch
):
    monkeypatch.setattr(
        "upmixer_web.features.projects.worker.JobSubprocess", InProcessJobRun
    )

    imported = web_client.post(
        "/api/v1/imports",
        files=[("files", ("tone.wav", _wav_bytes(), "audio/wav"))],
        data={"relative_paths": "tone.wav"},
    )
    assert imported.status_code == 201, imported.text

    created = web_client.post(
        "/api/v1/projects",
        json={
            "name": "Native-rate project",
            "manifest": {
                "version": "1.0.0",
                "engine": {
                    "mode": "stem",
                    "stems": ["Vocals"],
                    "stem_native_rate": True,
                },
                "mixing": {"channel_layout": "5.1"},
                "format": {
                    "type": "multichannel",
                    "codec": "wav_pcm",
                    "subtype": "PCM_24",
                    "sample_rate": 48_000,
                },
            },
        },
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]

    added = web_client.post(
        f"/api/v1/projects/{project_id}/assets",
        json={"import_id": imported.json()["id"]},
    )
    assert added.status_code == 201, added.text

    def wait_for_ready() -> dict:
        deadline = time.monotonic() + 10
        body: dict = {}
        while time.monotonic() < deadline:
            body = web_client.get(f"/api/v1/projects/{project_id}").json()
            if body["status"] in {"ready", "failed", "expansion_failed"}:
                assert body["status"] == "ready", body
                return body
            time.sleep(0.05)
        raise AssertionError(body)

    prepared = wait_for_ready()
    assert prepared["manifest"]["engine"]["stem_native_rate"] is True
    first_generation = prepared["stem_generation"]

    reprepared = web_client.post(
        f"/api/v1/projects/{project_id}/stems/reprepare",
        json={"stem_native_rate": True},
    )
    assert reprepared.status_code == 200, reprepared.text
    assert reprepared.json()["status"] == "expanding"
    assert reprepared.json()["manifest"]["engine"]["stem_native_rate"] is True

    prepared = wait_for_ready()
    assert prepared["stem_generation"] == first_generation + 1
    stem = prepared["tracks"][0]["stems"][0]
    assert stem["sample_rate"] == 48_000
    full = web_client.get(stem["audio_url"])
    assert full.status_code == 200
    assert sf.info(io.BytesIO(full.content)).samplerate == 48_000
    preview = web_client.get(stem["preview_url"])
    assert preview.status_code == 200
    assert sf.info(io.BytesIO(preview.content)).samplerate == 48_000

    exported = web_client.post(
        f"/api/v1/projects/{project_id}/exports", json={"layout": "5.1"}
    )
    assert exported.status_code == 201, exported.text
    job_id = exported.json()["id"]
    assert exported.json()["manifest"]["engine"]["stem_native_rate"] is True

    deadline = time.monotonic() + 10
    job = None
    while time.monotonic() < deadline:
        job = web_client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "completed", job.get("error")
    artifact = next(item for item in job["artifacts"] if item["kind"] == "upmix")
    delivery = web_client.get(artifact["download_url"])
    assert delivery.status_code == 200
    assert sf.info(io.BytesIO(delivery.content)).samplerate == 48_000

    changed_manifest = prepared["manifest"]
    changed_manifest["engine"]["stem_native_rate"] = False
    saved = web_client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"manifest": changed_manifest, "scene": prepared["scene"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["manifest"]["engine"]["stem_native_rate"] is False
    snapshot = web_client.get(f"/api/v1/jobs/{job_id}").json()
    assert snapshot["manifest"]["engine"]["stem_native_rate"] is True
