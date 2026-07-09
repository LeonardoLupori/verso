from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import QObject, QThread, pyqtBoundSignal, pyqtSignal
from PyQt6.QtWidgets import QWidget


class DeepSliceWorker(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self) -> None: ...


class JobWorker(Protocol):
    @property
    def done(self) -> pyqtBoundSignal: ...

    def run(self) -> None: ...
    def moveToThread(self, thread: QThread) -> None: ...
    def deleteLater(self) -> None: ...


class BackgroundJob[W: JobWorker]:
    def __init__(self, parent: QWidget, worker: W) -> None:
        self.worker: W = worker
