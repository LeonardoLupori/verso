"""File dialogs that remember where the user last was.

Thin wrappers over the ``QFileDialog`` static helpers. Each one starts in the
folder last used for its *kind* (see the ``DIR_*`` keys in
:mod:`verso.engine.settings`) and records the folder the user ends up in, so
the next dialog of that kind opens there.

Kinds are kept separate on purpose: choosing an export destination should not
move where Open Project starts. When nothing is remembered — a fresh install,
or a folder that has since gone away — each wrapper falls back to the default
its call site used before, passed as ``fallback``.

Cancelling records nothing, and so does a folder that cannot be resolved, so a
dialog the user backs out of never changes the next one's starting point.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog

from verso.engine.settings import DIR_PROJECT, get_last_dir, set_last_dir

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


def start_dir(kind: str = DIR_PROJECT, fallback: str = "") -> str:
    """Return the directory a dialog of *kind* should open in.

    Args:
        kind: One of the ``DIR_*`` keys in :mod:`verso.engine.settings`.
        fallback: Used when nothing usable is remembered — the call site's
            historical default (often ``""`` or the user's home folder).

    Returns:
        A directory path, or *fallback*.
    """
    remembered = get_last_dir(kind)
    return str(remembered) if remembered is not None else fallback


def _remember(chosen: str, kind: str) -> None:
    """Record the folder of *chosen*, ignoring an empty (cancelled) result."""
    if chosen:
        set_last_dir(Path(chosen), kind)


def open_file(
    parent: QWidget | None,
    title: str,
    file_filter: str,
    *,
    kind: str = DIR_PROJECT,
    fallback: str = "",
) -> str:
    """Ask for one existing file. Returns ``""`` if the user cancelled."""
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(kind, fallback), file_filter)
    _remember(path, kind)
    return path


def open_files(
    parent: QWidget | None,
    title: str,
    file_filter: str,
    *,
    kind: str = DIR_PROJECT,
    fallback: str = "",
) -> list[str]:
    """Ask for any number of existing files. Returns ``[]`` if cancelled."""
    paths, _ = QFileDialog.getOpenFileNames(parent, title, start_dir(kind, fallback), file_filter)
    if paths:
        _remember(paths[0], kind)
    return paths


def save_file(
    parent: QWidget | None,
    title: str,
    suggested_name: str,
    file_filter: str,
    *,
    kind: str = DIR_PROJECT,
    fallback: str = "",
) -> str:
    """Ask where to write a file. Returns ``""`` if the user cancelled.

    ``suggested_name`` may be a bare filename or a full path. A bare name is
    joined onto the remembered folder, so the dialog opens there with the name
    pre-filled; a full path is already anchored and is offered unchanged.
    """
    suggested = suggested_name
    if suggested_name and not Path(suggested_name).is_absolute():
        directory = start_dir(kind, fallback)
        if directory:
            suggested = str(Path(directory) / suggested_name)
    path, _ = QFileDialog.getSaveFileName(parent, title, suggested, file_filter)
    _remember(path, kind)
    return path


def choose_dir(
    parent: QWidget | None,
    title: str,
    *,
    kind: str = DIR_PROJECT,
    fallback: str = "",
    current: str = "",
) -> str:
    """Ask for a directory. Returns ``""`` if the user cancelled.

    Args:
        current: A value the call site already has (typically what is showing in
            a location field). It wins over the remembered folder, since it is
            the more specific intent.
    """
    directory = QFileDialog.getExistingDirectory(
        parent, title, current or start_dir(kind, fallback)
    )
    _remember(directory, kind)
    return directory
