"""Save and load VERSO project files."""

from __future__ import annotations

from pathlib import Path

from verso.engine.model.project import Project


def save_project(project: Project, path: Path) -> None:
    """Serialise *project* to *path* as JSON.

    Args:
        project: The project to save.
        path: Destination file path.
    """
    raise NotImplementedError


def load_project(path: Path) -> Project:
    """Deserialise a project from *path*.

    Args:
        path: Path to a project JSON file.

    Returns:
        Reconstructed :class:`~verso.engine.model.project.Project`.
    """
    raise NotImplementedError
