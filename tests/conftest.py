"""Shared test fixtures.

The autouse fixture below keeps logging hermetic: every test writes to a
throwaway ``VERSO_LOG_DIR`` and the ``verso`` logger's handlers/level are
snapshotted and restored, so a test that calls ``configure_logging`` never
leaks handlers into another test or writes to the real per-user log directory.
Importing engine modules does not configure logging on its own — only
``configure_logging`` does — so tests that never touch it are unaffected.
"""

from __future__ import annotations

import logging

import pytest

from verso.engine import logconf


@pytest.fixture(autouse=True)
def _isolate_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("VERSO_LOG_DIR", str(tmp_path / "logs"))

    verso_logger = logging.getLogger("verso")
    saved_handlers = verso_logger.handlers[:]
    saved_level = verso_logger.level
    saved_propagate = verso_logger.propagate

    logconf._configured = False
    logconf._log_file_path = None
    try:
        yield
    finally:
        for handler in verso_logger.handlers[:]:
            if handler not in saved_handlers:
                verso_logger.removeHandler(handler)
                handler.close()
        verso_logger.handlers[:] = saved_handlers
        verso_logger.setLevel(saved_level)
        verso_logger.propagate = saved_propagate
        logconf._configured = False
        logconf._log_file_path = None
