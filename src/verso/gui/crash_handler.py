"""Global crash handling for the GUI: route otherwise-lost failures into the log.

A windowed build has no console, so an uncaught exception (main thread, a worker
thread, or Qt's own C++ side) would vanish silently. These installers make sure
every such failure is logged, and — for the main-thread case — shown to the user
with the log-file location so a bug report can carry the traceback.

Kept separate from ``engine/logconf`` because this half is Qt-specific; it is
called from ``gui/app.run`` after the ``QApplication`` exists.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from pathlib import Path
from types import TracebackType

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication, QMessageBox

_log = logging.getLogger("verso")
_qt_log = logging.getLogger("verso.qt")


def install_excepthook(log_path: Path) -> None:
    """Install a ``sys.excepthook`` that logs uncaught exceptions and alerts the user.

    Logs the full traceback, then — only if a ``QApplication`` is running — shows
    a critical dialog naming ``log_path`` so the user can find and send the log.
    The previous hook is still called so debuggers keep working.

    Args:
        log_path: The active log file, surfaced to the user in the dialog.
    """
    previous_hook = sys.excepthook

    def hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        # Never swallow the interrupt key.
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_tb)
            return

        _log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

        if QApplication.instance() is not None:
            details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("VERSO — Unexpected error")
            box.setText(
                "An unexpected error occurred.\n\n"
                "The full details have been written to the log file:\n"
                f"{log_path}"
            )
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()

        previous_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def install_thread_excepthook() -> None:
    """Install a ``threading.excepthook`` so worker-thread failures are logged.

    Insurance beyond each worker's own try/except: anything that escapes a
    Python thread's ``run`` is logged with its traceback instead of printed to a
    console nobody sees.
    """
    previous_hook = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        _log.critical(
            "Uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        previous_hook(args)

    threading.excepthook = hook


_QT_LEVEL = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def install_qt_message_handler() -> None:
    """Route Qt's own diagnostic messages into the ``verso.qt`` logger."""

    def handler(msg_type: QtMsgType, context: object, message: str) -> None:
        _qt_log.log(_QT_LEVEL.get(msg_type, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)
