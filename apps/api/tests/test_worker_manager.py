"""Worker startup migration for legacy project manifests."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from upmixer_web.api import create_app
from upmixer_web.settings import Settings
from upmixer_web.shared.models import ImportBatch, MediaAsset, Project, ProjectTrack


def test_start_removes_legacy_native_rate_and_requeues_false_projects(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'worker-manager.db'}",
        worker_count=1,
    )
    app = create_app(settings)
    sessions = app.state.sessions

    with sessions() as session:
        batch = ImportBatch(kind="track", title="Legacy projects")
        false_asset = MediaAsset(
            import_batch=batch,
            filename="false.wav",
            relative_path="false.wav",
            storage_key="assets/false.wav",
            sha256="0" * 64,
            size_bytes=1,
        )
        true_asset = MediaAsset(
            import_batch=batch,
            filename="true.wav",
            relative_path="true.wav",
            storage_key="assets/true.wav",
            sha256="1" * 64,
            size_bytes=1,
        )
        false_project = Project(
            import_batch=batch,
            name="Legacy false",
            manifest={
                "version": "1.0.0",
                "engine": {"mode": "stem", "stem_native_rate": False},
            },
            status="ready",
            progress=0.8,
            status_message="Old status",
            prepared_stems=["Vocals"],
            requested_stems=["Vocals"],
            revision=4,
            error="Old error",
        )
        true_project = Project(
            import_batch=batch,
            name="Legacy true",
            manifest={
                "version": "1.0.0",
                "engine": {"mode": "stem", "stem_native_rate": True},
            },
            status="ready",
            progress=1.0,
            status_message="Ready",
            prepared_stems=["Vocals"],
            requested_stems=["Vocals"],
            revision=7,
        )
        false_track = ProjectTrack(
            project=false_project,
            asset=false_asset,
            position=0,
            status="failed",
            progress=0.4,
            error="Old track error",
        )
        true_track = ProjectTrack(
            project=true_project,
            asset=true_asset,
            position=0,
            status="ready",
            progress=1.0,
        )
        session.add_all([
            batch,
            false_asset,
            true_asset,
            false_project,
            true_project,
            false_track,
            true_track,
        ])
        session.commit()
        false_id = false_project.id
        true_id = true_project.id

    manager = app.state.manager
    monkeypatch.setattr(manager, "_dispatch_loop", lambda: None)
    manager.start()
    manager.stop()

    with sessions() as session:
        migrated_false = session.get(Project, false_id)
        migrated_true = session.get(Project, true_id)
        assert migrated_false is not None
        assert migrated_true is not None
        assert "stem_native_rate" not in migrated_false.manifest["engine"]
        assert migrated_false.revision == 5
        assert migrated_false.status == "expanding"
        assert migrated_false.progress == 0.0
        assert migrated_false.error is None
        assert migrated_false.status_message == "Waiting to rebuild project stems"
        assert migrated_false.tracks[0].status == "queued"
        assert migrated_false.tracks[0].progress == 0.0
        assert migrated_false.tracks[0].error is None
        assert "stem_native_rate" not in migrated_true.manifest["engine"]
        assert migrated_true.revision == 7
        assert migrated_true.status == "ready"
        assert migrated_true.progress == 1.0
        assert migrated_true.tracks[0].status == "ready"
        assert migrated_true.tracks[0].progress == 1.0
