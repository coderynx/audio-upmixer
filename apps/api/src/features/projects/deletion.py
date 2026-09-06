"""Project deletion state transition."""

from sqlalchemy import update
from sqlalchemy.orm import Session

from upmixer_web.shared.models import Project


def mark_project_deleting(session: Session, project: Project) -> bool:
    """Mark an in-flight project for worker-side teardown, or signal the
    caller to delete it immediately.

    Returns ``True`` when the project is idle and the caller should delete it
    right away (via ``WorkerManager.delete_now_project``); ``False`` when it
    is in-flight and has been flagged ``deleting`` for the worker to tear down.
    """
    if project.status in {"preparing", "expanding"}:
        updated = session.execute(
            update(Project)
            .where(Project.id == project.id, Project.status.in_(("preparing", "expanding")))
            .values(status="deleting", status_message="Stopping worker before deletion")
        )
        session.commit()
        return updated.rowcount == 0
    return True
