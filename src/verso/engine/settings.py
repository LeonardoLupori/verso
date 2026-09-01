"""Small per-user preferences store — things VERSO should remember between runs.

This is *not* project state: a project keeps everything about the data in its
own ``project-verso.json``. This file holds only convenience settings tied to
the person using the machine, currently the folder the last project was opened
from, so the Open dialog can start where the user last worked.

It lives beside the logs in the per-user config directory (never in the project
and never in the repository), and is pure stdlib + ``platformdirs`` so the
engine stays Qt-free.

Every read and write is best-effort: a missing, unreadable or corrupt file
yields defaults, and a failed write is logged and swallowed. Remembering a
folder is a convenience, and must never be able to stop a project from opening.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import platformdirs

_log = logging.getLogger(__name__)

#: Name of the settings file inside :func:`default_config_dir`.
SETTINGS_FILENAME = "settings.json"

#: Key holding the directory the last project was opened from.
LAST_PROJECT_DIR = "last_project_dir"

#: Recognised "kinds" of remembered folder. Each kind is stored under its own
#: key, so picking an export destination never moves where Open Project starts.
#: The values are the stored key names — keep them stable.
DIR_PROJECT = LAST_PROJECT_DIR
DIR_IMAGES = "last_images_dir"
DIR_EXPORT = "last_export_dir"
DIR_IMPORT = "last_import_dir"
DIR_ANNOTATION = "last_annotation_dir"


def default_config_dir() -> Path:
    """Return the per-user directory VERSO keeps settings in, creating it.

    Mirrors :func:`verso.engine.logconf.default_log_dir`: the correct per-OS
    location via :mod:`platformdirs`, overridable with ``VERSO_CONFIG_DIR`` for
    tests and support.

    Returns:
        The config directory as a :class:`~pathlib.Path` (guaranteed to exist).
    """
    override = os.environ.get("VERSO_CONFIG_DIR")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    # appauthor=False avoids a doubled "VERSO\VERSO" segment on Windows.
    return Path(platformdirs.user_config_dir("VERSO", appauthor=False, ensure_exists=True))


def settings_path() -> Path:
    """Return the full path of the settings file (which may not exist yet)."""
    return default_config_dir() / SETTINGS_FILENAME


def load_settings() -> dict[str, Any]:
    """Return the stored settings, or ``{}`` if there are none to read.

    Never raises: a missing file, an unreadable one, invalid JSON, or a JSON
    document that is not an object all come back as an empty dict.
    """
    path = settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _log.warning("Ignoring unreadable settings file %s", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        _log.warning("Ignoring settings file %s: expected a JSON object", path)
        return {}
    return data


def save_settings(settings: dict[str, Any]) -> None:
    """Write *settings*, replacing whatever was there. Never raises.

    The file is written to a temporary sibling and then moved into place, so an
    interrupted write cannot leave a half-written file behind.

    Args:
        settings: The complete settings mapping to store.
    """
    path = settings_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        _log.warning("Could not write settings to %s", path, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def update_setting(key: str, value: Any) -> None:
    """Set one key, leaving the rest of the stored settings untouched."""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)


def get_last_dir(kind: str = DIR_PROJECT) -> Path | None:
    """Return the folder last used for *kind*, if it is still usable.

    Returns ``None`` when nothing has been recorded yet, or when the recorded
    folder no longer exists — an unplugged external drive, a renamed share — so
    callers fall back to their normal default rather than opening a file dialog
    pointed at a dead path.

    Args:
        kind: One of the ``DIR_*`` keys (defaults to :data:`DIR_PROJECT`).
    """
    raw = load_settings().get(kind)
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_dir():
        _log.debug("Ignoring remembered %s folder (gone): %s", kind, path)
        return None
    return path


def set_last_dir(path: Path, kind: str = DIR_PROJECT) -> None:
    """Remember the folder holding *path* as the next default for *kind*.

    Args:
        path: A file that was just chosen — its parent folder is what gets
            stored — or a directory, which is stored as-is.
        kind: One of the ``DIR_*`` keys (defaults to :data:`DIR_PROJECT`).
    """
    directory = path if path.is_dir() else path.parent
    try:
        resolved = str(directory.resolve())
    except OSError:  # pragma: no cover - unresolvable path (dead network share)
        _log.debug("Not remembering unresolvable folder %s", directory)
        return
    update_setting(kind, resolved)
    _log.debug("Remembered %s folder %s", kind, resolved)


def get_last_project_dir() -> Path | None:
    """Return the folder the last project was opened from, if still usable."""
    return get_last_dir(DIR_PROJECT)


def set_last_project_dir(project_path: Path) -> None:
    """Remember the folder holding *project_path* for the next Open dialog."""
    set_last_dir(project_path, DIR_PROJECT)
