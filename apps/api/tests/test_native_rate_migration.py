"""One-time migration for projects prepared before native-rate separation."""

import json
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config


def _alembic_config(database_url: str) -> Config:
    package_dir = Path(__import__("upmixer_web").__file__).resolve().parent
    config = Config()
    config.attributes["database_url_configured"] = True
    config.set_main_option("script_location", str(package_dir / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_native_rate_migration_requeues_only_unproven_prepared_projects(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'native-rate-migration.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "e6f7a8b9c0d1")
    engine = sa.create_engine(database_url)

    projects = {
        "no-marker": ({"engine": {"mode": "stem"}}, ["Vocals"], "ready", 2, "ready"),
        "legacy-false": (
            {"engine": {"mode": "stem", "stem_native_rate": False}},
            ["Vocals"],
            "expansion_failed",
            4,
            "failed",
        ),
        "legacy-true": (
            {"engine": {"mode": "stem", "stem_native_rate": True}},
            ["Vocals"],
            "ready",
            6,
            "ready",
        ),
        "legacy-false-empty": (
            {"engine": {"mode": "stem", "stem_native_rate": False}},
            [],
            "queued",
            8,
            "queued",
        ),
        "deleting": ({"engine": {"mode": "stem"}}, ["Vocals"], "deleting", 10, "running"),
    }

    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO import_batches (id, kind, title, created_at)"
                " VALUES ('batch', 'track', 'Legacy', '2026-01-01')"
            )
        )
        for index, project_id in enumerate(projects):
            connection.execute(
                sa.text(
                    "INSERT INTO media_assets (id, import_id, filename, relative_path, storage_key,"
                    " sha256, size_bytes, position) VALUES (:asset_id, 'batch', :filename, :filename,"
                    " :storage_key, :sha256, 1, :position)"
                ),
                {
                    "asset_id": f"asset-{index}",
                    "filename": f"{project_id}.wav",
                    "storage_key": f"assets/{project_id}.wav",
                    "sha256": str(index) * 64,
                    "position": index,
                },
            )
            manifest, prepared_stems, status, revision, track_status = projects[project_id]
            connection.execute(
                sa.text(
                    "INSERT INTO projects (id, import_id, name, status, progress, status_message,"
                    " manifest, scene, view_state, progress_log, requested_stems, prepared_stems,"
                    " stem_generation, preview_quality, revision, error, created_at, updated_at)"
                    " VALUES (:id, 'batch', :name, :status, 0.8, 'Old status', :manifest, '{}', '{}',"
                    " '[]', :stems, :stems, 1, 'high', :revision, 'Old error',"
                    " '2026-01-01', '2026-01-01')"
                ),
                {
                    "id": project_id,
                    "name": project_id,
                    "status": status,
                    "manifest": json.dumps(manifest),
                    "stems": json.dumps(prepared_stems),
                    "revision": revision,
                },
            )
            connection.execute(
                sa.text(
                    "INSERT INTO project_tracks (id, project_id, asset_id, position, status, progress,"
                    " layout_overrides, scene_overrides, error) VALUES (:id, :project_id, :asset_id,"
                    " 0, :status, 0.4, '{}', '{}', 'Old track error')"
                ),
                {
                    "id": f"track-{index}",
                    "project_id": project_id,
                    "asset_id": f"asset-{index}",
                    "status": track_status,
                },
            )
    engine.dispose()

    command.upgrade(config, "head")

    engine = sa.create_engine(database_url)
    with engine.begin() as connection:
        rows = connection.execute(
            sa.text(
                "SELECT id, manifest, status, progress, status_message, prepared_stems,"
                " revision, error FROM projects ORDER BY id"
            )
        ).fetchall()
        tracks = connection.execute(
            sa.text("SELECT project_id, status, progress, error FROM project_tracks ORDER BY project_id")
        ).fetchall()
    engine.dispose()

    projects_by_id = {row.id: row for row in rows}
    tracks_by_project = {row.project_id: row for row in tracks}
    for project_id in projects:
        assert "stem_native_rate" not in json.loads(projects_by_id[project_id].manifest)["engine"]

    for project_id, revision in (("no-marker", 3), ("legacy-false", 5)):
        row = projects_by_id[project_id]
        assert row.status == "expanding"
        assert row.progress == 0.0
        assert row.status_message == "Waiting to rebuild project stems"
        assert row.revision == revision
        assert row.error is None
        track = tracks_by_project[project_id]
        assert (track.status, track.progress, track.error) == ("queued", 0.0, None)

    for project_id, revision, status, track_status in (
        ("legacy-true", 6, "ready", "ready"),
        ("legacy-false-empty", 8, "queued", "queued"),
        ("deleting", 10, "deleting", "running"),
    ):
        row = projects_by_id[project_id]
        assert (row.revision, row.status) == (revision, status)
        assert tracks_by_project[project_id].status == track_status
