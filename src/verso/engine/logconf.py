"""Logging configuration for VERSO — the single source of truth for *where*
logs go and *how* logging is set up.

Pure stdlib + ``platformdirs`` (both Qt-free), so this module is safe to import
from the engine and from the standalone elastix subprocess as well as the GUI.
It configures only the ``verso`` package logger, never the root logger, so
third-party logging (brainglobe, itk, …) is left untouched. Application code
just calls ``logging.getLogger(__name__)``; handlers are attached here, once,
from the entry points (``gui/app.py``, ``__main__.py``, ``_elastix_worker.py``).
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import platformdirs

#: Root logger for the whole package. Every module uses ``getLogger(__name__)``,
#: which resolves under this one (``verso.gui.…`` / ``verso.engine.…``).
ROOT_LOGGER_NAME = "verso"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB per file
_BACKUP_COUNT = 5

#: Set once ``configure_logging`` has run so re-entry is a cheap no-op.
_configured = False
_log_file_path: Path | None = None


def default_log_dir() -> Path:
    """Return the per-user directory VERSO writes logs to, creating it.

    Uses :func:`platformdirs.user_log_dir` for the correct per-OS location
    (``%LOCALAPPDATA%\\VERSO\\Logs`` on Windows, ``~/Library/Logs/VERSO`` on
    macOS, ``~/.local/state/VERSO/log`` on Linux). The ``VERSO_LOG_DIR``
    environment variable overrides it (useful for tests and support).

    Returns:
        The log directory as a :class:`~pathlib.Path` (guaranteed to exist).
    """
    override = os.environ.get("VERSO_LOG_DIR")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    # appauthor=False avoids a doubled "VERSO\VERSO" segment on Windows.
    return Path(platformdirs.user_log_dir("VERSO", appauthor=False, ensure_exists=True))


def _resolve_level(level: int | str | None) -> int:
    """Resolve an effective log level from arg → ``VERSO_LOG_LEVEL`` → INFO."""
    if level is None:
        level = os.environ.get("VERSO_LOG_LEVEL")
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    return logging.getLevelNamesMapping().get(level.strip().upper(), logging.INFO)


def configure_logging(
    *,
    process_tag: str = "app",
    level: int | str | None = None,
    console: bool = True,
    log_dir: Path | None = None,
) -> Path:
    """Configure the ``verso`` logger. Idempotent — safe to call more than once.

    Args:
        process_tag: Names the log file (``verso-<tag>.log``). Each process uses
            a distinct tag so the main app and the elastix child never rotate
            the same file (concurrent ``RotatingFileHandler``s on one path
            corrupt rotation).
        level: Log level (int or name). Falls back to ``VERSO_LOG_LEVEL`` then
            ``INFO``.
        console: When true, also stream to ``sys.stderr``. Never ``stdout`` —
            the elastix child uses stdout as its READY/DONE IPC channel.
        log_dir: Override the log directory (defaults to :func:`default_log_dir`).

    Returns:
        The path of the log file handlers write to.
    """
    global _configured, _log_file_path

    if _configured and _log_file_path is not None:
        return _log_file_path

    directory = log_dir if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / f"verso-{process_tag}.log"

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    # Own handlers only; do not double up through the root logger.
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # In a windowed (pythonw) build sys.stderr can be None; only stream when a
    # real stderr exists. Never stdout — it is the elastix worker's IPC channel.
    if console and sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Route the stdlib ``warnings`` module (py.warnings logger) into logging too.
    logging.captureWarnings(True)

    _configured = True
    _log_file_path = log_file
    logger.debug(
        "Logging configured: file=%s level=%s", log_file, logging.getLevelName(logger.level)
    )
    return log_file


def log_file_path() -> Path | None:
    """Return the active log file path, or ``None`` if not yet configured."""
    return _log_file_path
