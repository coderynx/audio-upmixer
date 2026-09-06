import pytest

pytest.importorskip("sqlalchemy")

from upmixer_web.features.projects.deletion import mark_project_deleting
from upmixer_web.shared.database import create_database_engine, create_session_factory, upgrade_database
from upmixer_web.shared.models import Project


def test_delete_race_does_not_overwrite_a_finished_project(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'delete-race.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = create_session_factory(engine)

    with factory() as session:
        project = Project(name="Delete race", manifest={}, status="preparing")
        session.add(project)
        session.commit()
        project_id = project.id

    with factory() as stale_session:
        cached = stale_session.get(Project, project_id)
        assert cached is not None
        with factory() as current_session:
            current = current_session.get(Project, project_id)
            assert current is not None
            current.status = "ready"
            current.stem_generation = 1
            current.prepared_stems = ["Vocals"]
            current_session.commit()

        assert mark_project_deleting(stale_session, cached) is True

    with factory() as session:
        current = session.get(Project, project_id)
        assert current is not None
        assert current.status == "ready"

    engine.dispose()
