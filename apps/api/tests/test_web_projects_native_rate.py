"""Q21 native-rate project delivery integration."""

import io
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

import soundfile as sf

import _helpers
from _helpers import InProcessJobRun, _wav_bytes


def _queue_worker_project(client, name="Worker race project") -> str:
    imported = client.post(
        "/api/v1/imports",
        files=[("files", ("tone.wav", _wav_bytes(), "audio/wav"))],
        data={"relative_paths": "tone.wav"},
    )
    assert imported.status_code == 201, imported.text
    created = client.post(
        "/api/v1/projects",
        json={
            "name": name,
            "manifest": {
                "version": "1.0.0",
                "engine": {"mode": "stem", "stems": ["Vocals"]},
                "mixing": {"channel_layout": "5.1"},
            },
        },
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    added = client.post(
        f"/api/v1/projects/{project_id}/assets",
        json={"import_id": imported.json()["id"]},
    )
    assert added.status_code == 201, added.text
    return project_id


def test_native_rate_project_prepares_delivers_and_exports(
    web_client, in_process_jobs, monkeypatch
):
    separation_calls: list[tuple[int, int]] = []
    original_fake_execute_plan = _helpers._fake_execute_plan

    def fake_execute_plan(*args, **kwargs):
        sep_path = args[2]
        sep_sr = args[3]
        separation_calls.append((sf.info(sep_path).samplerate, sep_sr))
        multiplier = 0.25 if len(separation_calls) == 1 else 0.75
        return {
            name: audio * np.float32(multiplier)
            for name, audio in original_fake_execute_plan(*args, **kwargs).items()
        }

    monkeypatch.setattr(_helpers, "_fake_execute_plan", fake_execute_plan)
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
                    # Legacy clients may still send this removed setting.
                    "stem_native_rate": False,
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
    assert "stem_native_rate" not in prepared["manifest"]["engine"]
    first_generation = prepared["stem_generation"]
    assert first_generation == 1
    track_id = prepared["tracks"][0]["id"]
    project_stems = web_client.app.state.project_stems
    old_stem_dir = project_stems.generation_stem_dir(
        project_id, track_id, first_generation
    )
    old_stem_audio, old_stem_rate = sf.read(
        str(old_stem_dir / "Vocals.wav"), always_2d=True
    )
    assert (old_stem_dir / "peaks.bin").is_file()
    assert (old_stem_dir / "peaks.json").is_file()
    assert old_stem_rate == 44_100
    old_stem_peak = float(np.abs(old_stem_audio).max())
    assert separation_calls == [(44_100, 44_100)]

    manager = web_client.app.state.manager
    monkeypatch.setattr(manager, "_submit_available", lambda: None)

    exported = web_client.post(
        f"/api/v1/projects/{project_id}/exports", json={"layout": "5.1"}
    )
    assert exported.status_code == 201, exported.text
    job_id = exported.json()["id"]
    assert "stem_native_rate" not in exported.json()["manifest"]["engine"]

    reprepared = web_client.post(
        f"/api/v1/projects/{project_id}/stems/reprepare",
        json={"stem_native_rate": False},
    )
    assert reprepared.status_code == 200, reprepared.text
    assert reprepared.json()["status"] == "expanding"
    assert "stem_native_rate" not in reprepared.json()["manifest"]["engine"]

    manager._run_project(project_id)
    prepared = wait_for_ready()
    assert prepared["stem_generation"] == first_generation + 1
    assert "stem_native_rate" not in prepared["manifest"]["engine"]
    stem = prepared["tracks"][0]["stems"][0]
    assert stem["sample_rate"] == 44_100
    full = web_client.get(stem["audio_url"])
    assert full.status_code == 200
    assert sf.info(io.BytesIO(full.content)).samplerate == 44_100
    preview = web_client.get(stem["preview_url"])
    assert preview.status_code == 200
    assert sf.info(io.BytesIO(preview.content)).samplerate == 48_000
    new_stem_dir = project_stems.generation_stem_dir(
        project_id, track_id, prepared["stem_generation"]
    )
    new_stem_audio, new_stem_rate = sf.read(
        str(new_stem_dir / "Vocals.wav"), always_2d=True
    )
    assert (new_stem_dir / "peaks.bin").is_file()
    assert (new_stem_dir / "peaks.json").is_file()
    assert new_stem_rate == 44_100
    assert float(np.abs(new_stem_audio).max()) > old_stem_peak * 2
    assert separation_calls == [(44_100, 44_100), (44_100, 44_100)]

    export_inputs: list[tuple[str, float]] = []
    from upmixer.separation.stem_pipeline import StemUpmixPipeline

    original_process_file = StemUpmixPipeline.process_file

    def capture_export_input(self, input_path, output_path, *args, **kwargs):
        stem_input_dir = self.config.stem_input_dir
        assert stem_input_dir is not None
        audio, sample_rate = sf.read(
            str(Path(stem_input_dir) / "Vocals.wav"), always_2d=True
        )
        export_inputs.append((stem_input_dir, float(np.abs(audio).max())))
        assert sample_rate == 44_100
        return original_process_file(
            self, input_path, output_path, *args, **kwargs
        )

    monkeypatch.setattr(StemUpmixPipeline, "process_file", capture_export_input)
    manager._run_job(job_id)

    job = web_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job.get("error")
    assert "stem_native_rate" not in job["manifest"]["engine"]
    assert len(export_inputs) == 1
    assert export_inputs[0][0] == str(old_stem_dir)
    assert export_inputs[0][1] == pytest.approx(old_stem_peak, abs=2e-5)
    artifact = next(item for item in job["artifacts"] if item["kind"] == "upmix")
    delivery = web_client.get(artifact["download_url"])
    assert delivery.status_code == 200
    delivery_info = sf.info(io.BytesIO(delivery.content))
    assert delivery_info.format == "WAV"
    assert delivery_info.subtype == "PCM_24"
    assert delivery_info.samplerate == 48_000
    assert delivery_info.channels == 6
    assert delivery_info.frames == 4_800


def test_project_prepare_discards_stale_generation_and_requeues_current_settings(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from upmixer_web.api import create_app
    from upmixer_web.settings import Settings
    from upmixer_web.shared.models import Project

    database_url = f"sqlite:///{tmp_path / 'stale-project.db'}"
    settings = Settings(data_dir=tmp_path, database_url=database_url, worker_count=1)
    monkeypatch.setattr("upmixer_web.worker.WorkerManager.start", lambda _self: None)
    monkeypatch.setattr("upmixer_web.worker.WorkerManager.stop", lambda _self: None)
    monkeypatch.setattr(
        "upmixer_web.features.projects.worker.JobSubprocess", InProcessJobRun
    )

    with TestClient(create_app(settings)) as client:
        imported = client.post(
            "/api/v1/imports",
            files=[("files", ("tone.wav", _wav_bytes(), "audio/wav"))],
            data={"relative_paths": "tone.wav"},
        )
        assert imported.status_code == 201, imported.text
        created = client.post("/api/v1/projects", json={
            "name": "Stale project",
            "manifest": {
                "version": "1.0.0",
                "engine": {"mode": "stem", "stems": ["Vocals"]},
                "mixing": {"channel_layout": "5.1"},
            },
        })
        assert created.status_code == 201, created.text
        project_id = created.json()["id"]
        added = client.post(
            f"/api/v1/projects/{project_id}/assets",
            json={"import_id": imported.json()["id"]},
        )
        assert added.status_code == 201, added.text

        manager = client.app.state.manager
        original_fake_execute_plan = _helpers._fake_execute_plan
        mutation_count = 0

        def mutate_then_separate(*args, **kwargs):
            nonlocal mutation_count
            mutation_count += 1
            if mutation_count == 1:
                with manager.sessions() as session:
                    project = session.get(Project, project_id)
                    assert project is not None
                    manifest = dict(project.manifest)
                    manifest["race_marker"] = "legacy-setting-change"
                    project.manifest = manifest
                    project.revision += 1
                    session.commit()
            return original_fake_execute_plan(*args, **kwargs)

        monkeypatch.setattr(_helpers, "_fake_execute_plan", mutate_then_separate)
        manager._run_project(project_id)

        body = client.get(f"/api/v1/projects/{project_id}").json()
        assert body["status"] == "queued"
        assert body["status_message"] == "Waiting to prepare project stems"
        assert body["manifest"]["race_marker"] == "legacy-setting-change"
        assert body["prepared_stems"] == []
        track_id = body["tracks"][0]["id"]
        assert not manager.project_stems.generation_stem_dir(
            project_id, track_id, 1
        ).exists()


def test_project_prepare_deletion_race_removes_partial_generation(web_client, monkeypatch):
    from upmixer_web.shared.models import Project

    manager = web_client.app.state.manager
    monkeypatch.setattr(manager, "_submit_available", lambda: None)
    monkeypatch.setattr(
        "upmixer_web.features.projects.worker.JobSubprocess", InProcessJobRun
    )
    project_id = _queue_worker_project(web_client, "Delete race project")

    original_catalogue = manager.project_stems.catalogue_track

    def catalogue_then_delete(session, project, track, *args, **kwargs):
        with manager.sessions() as mutation_session:
            current = mutation_session.get(Project, project_id)
            assert current is not None
            current.status = "deleting"
            current.revision += 1
            mutation_session.commit()
        return original_catalogue(session, project, track, *args, **kwargs)

    monkeypatch.setattr(manager.project_stems, "catalogue_track", catalogue_then_delete)
    manager._run_project(project_id)

    assert web_client.get(f"/api/v1/projects/{project_id}").status_code == 404
    assert not (manager.project_stems.root / project_id).exists()


def test_project_prepare_atomic_publish_discards_finalization_race(web_client, monkeypatch):
    from upmixer_web.shared.models import Project

    manager = web_client.app.state.manager
    monkeypatch.setattr(manager, "_submit_available", lambda: None)
    monkeypatch.setattr(
        "upmixer_web.features.projects.worker.JobSubprocess", InProcessJobRun
    )
    project_id = _queue_worker_project(web_client, "Finalization race project")

    original_catalogue = manager.project_stems.catalogue_track

    def catalogue_then_mutate(session, project, track, *args, **kwargs):
        with manager.sessions() as mutation_session:
            current = mutation_session.get(Project, project_id)
            assert current is not None
            current.manifest = {**current.manifest, "race_marker": "new"}
            current.revision += 1
            mutation_session.commit()
        return original_catalogue(session, project, track, *args, **kwargs)

    monkeypatch.setattr(manager.project_stems, "catalogue_track", catalogue_then_mutate)
    manager._run_project(project_id)

    body = web_client.get(f"/api/v1/projects/{project_id}").json()
    assert body["status"] == "queued"
    assert body["manifest"]["race_marker"] == "new"
    assert body["stem_generation"] == 0
    track_id = body["tracks"][0]["id"]
    assert not manager.project_stems.generation_stem_dir(project_id, track_id, 1).exists()
