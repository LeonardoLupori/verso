"""Tests for the logging configuration module."""

from __future__ import annotations

import logging
import sys

from verso.engine import logconf
from verso.engine.logconf import configure_logging, default_log_dir


def test_default_log_dir_respects_env(tmp_path, monkeypatch):
    target = tmp_path / "custom-logs"
    monkeypatch.setenv("VERSO_LOG_DIR", str(target))
    resolved = default_log_dir()
    assert resolved == target
    assert resolved.is_dir()


def test_configure_logging_writes_to_file(tmp_path):
    log_file = configure_logging(process_tag="app", log_dir=tmp_path, console=False)
    assert log_file == tmp_path / "verso-app.log"

    logging.getLogger("verso.testmodule").info("hello-marker")
    for handler in logging.getLogger("verso").handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello-marker" in log_file.read_text(encoding="utf-8")


def test_configure_logging_is_idempotent(tmp_path):
    first = configure_logging(process_tag="app", log_dir=tmp_path)
    count = len(logging.getLogger("verso").handlers)
    second = configure_logging(process_tag="app", log_dir=tmp_path)
    assert first == second
    # A second call must not stack a duplicate set of handlers.
    assert len(logging.getLogger("verso").handlers) == count


def test_distinct_file_per_process_tag(tmp_path):
    app_file = configure_logging(process_tag="app", log_dir=tmp_path)
    # Simulate a separate process configuring afresh.
    logconf._configured = False
    logconf._log_file_path = None
    child_file = configure_logging(process_tag="elastix", log_dir=tmp_path)

    assert app_file.name == "verso-app.log"
    assert child_file.name == "verso-elastix.log"
    assert app_file != child_file


def test_no_stdout_handler_even_with_console(tmp_path):
    # stdout is reserved for the elastix worker's IPC handshake; the console
    # handler must always target stderr, never stdout.
    configure_logging(process_tag="app", log_dir=tmp_path, console=True)
    streams = [getattr(h, "stream", None) for h in logging.getLogger("verso").handlers]
    assert sys.stdout not in streams


def test_level_from_argument(tmp_path):
    configure_logging(process_tag="app", log_dir=tmp_path, level="DEBUG", console=False)
    assert logging.getLogger("verso").level == logging.DEBUG
