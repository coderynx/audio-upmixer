"""Rebuild prepared projects for the always-native stem separation policy.

Projects created before native-rate separation became unconditional may have
prepared stems rendered at the delivery rate. Projects with no explicit
legacy ``True`` marker must regenerate those stems once; an explicit ``True``
marker proves that the cached stems already use the native rate.

Revision ID: f7a1b2c3d4e5
Revises: e6f7a8b9c0d1
"""

import json
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "f7a1b2c3d4e5"
down_revision: Union[str, None] = "e6f7a8b9c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_MISSING = object()


def _decode_json(value: Any) -> Any:
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def migrate_projects(connection) -> None:
    """Strip the old marker and requeue stale prepared project stems."""
    rows = connection.execute(
        sa.text("SELECT id, manifest, prepared_stems, status FROM projects")
    ).fetchall()
    for project_id, raw_manifest, raw_prepared_stems, status in rows:
        manifest = _decode_json(raw_manifest)
        if not isinstance(manifest, dict):
            continue
        engine = manifest.get("engine")
        if not isinstance(engine, dict):
            engine = {}
        legacy_native_rate = engine.get("stem_native_rate", _MISSING)
        prepared_stems = _decode_json(raw_prepared_stems)
        has_prepared_stems = bool(prepared_stems)
        if legacy_native_rate is _MISSING and not has_prepared_stems:
            continue

        if isinstance(manifest.get("engine"), dict):
            manifest["engine"].pop("stem_native_rate", None)
        encoded_manifest = json.dumps(manifest)
        if legacy_native_rate is True or not has_prepared_stems or status == "deleting":
            connection.execute(
                sa.text("UPDATE projects SET manifest = :manifest WHERE id = :id"),
                {"manifest": encoded_manifest, "id": project_id},
            )
            continue

        connection.execute(
            sa.text(
                "UPDATE projects SET manifest = :manifest, status = 'expanding',"
                " progress = 0.0, status_message = 'Waiting to rebuild project stems',"
                " error = NULL, revision = revision + 1 WHERE id = :id"
            ),
            {"manifest": encoded_manifest, "id": project_id},
        )
        connection.execute(
            sa.text(
                "UPDATE project_tracks SET status = 'queued', progress = 0.0,"
                " error = NULL WHERE project_id = :project_id"
            ),
            {"project_id": project_id},
        )


def upgrade() -> None:
    migrate_projects(op.get_bind())


def downgrade() -> None:
    pass
