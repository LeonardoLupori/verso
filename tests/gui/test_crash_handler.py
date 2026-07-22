"""Tests for the global crash handler."""

from __future__ import annotations

import logging
import sys

from verso.gui import crash_handler


def test_install_excepthook_logs_traceback(tmp_path, monkeypatch):
    # Force the "no QApplication" path so no modal dialog is shown (a running
    # QApplication from another test could otherwise block on exec()).
    monkeypatch.setattr(crash_handler.QApplication, "instance", staticmethod(lambda: None))
    # Chain to the stdlib default rather than pytest-qt's event-loop hook (which
    # would fail the test); monkeypatch restores the original hook afterwards.
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    verso_logger = logging.getLogger("verso")
    verso_logger.addHandler(handler)
    verso_logger.setLevel(logging.DEBUG)

    crash_handler.install_excepthook(tmp_path / "verso-app.log")
    try:
        raise ValueError("boom-xyz")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    verso_logger.removeHandler(handler)

    logged = [r for r in records if r.levelno >= logging.CRITICAL]
    assert logged, "excepthook did not log a CRITICAL record"
    assert any(r.exc_info is not None for r in logged), "traceback was not attached"


def test_thread_excepthook_installs_without_error():
    saved = __import__("threading").excepthook
    try:
        crash_handler.install_thread_excepthook()
        assert __import__("threading").excepthook is not saved
    finally:
        __import__("threading").excepthook = saved
