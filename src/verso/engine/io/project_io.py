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


def import_project_styling(target: Project, source: Project) -> None:
    """Copy color and styling settings from *source* into *target* in place.

    Imports:
      - Per-channel ``color``, ``scale``, ``visible``, matched by position
        (index). Channel ``name`` is preserved on the target. Channels past
        the end of either list are left untouched.
      - Control-point style: ``cp_size``, ``cp_shape``, ``cp_color``.

    Args:
        target: Project to update in place.
        source: Project whose styling is read.
    """
    overlap = min(len(target.channels), len(source.channels))
    for i in range(overlap):
        src = source.channels[i]
        tgt = target.channels[i]
        tgt.color = src.color
        tgt.scale = src.scale
        tgt.visible = src.visible

    target.cp_size = source.cp_size
    target.cp_shape = source.cp_shape
    target.cp_color = source.cp_color
