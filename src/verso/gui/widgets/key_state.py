"""Application-level tracking of held modifier keys (Space and Shift).

A single ``QApplication`` event filter keeps ``SpaceState.held`` / ``ShiftState.held``
in sync regardless of which widget has keyboard focus, and notifies registered
listeners (image canvases) so they can update cursors and drag behaviour. Install
it once via :func:`ensure_key_state_filter`.

Listeners are expected to expose ``_on_space_changed()`` and ``_on_shift_changed()``
methods; they register by adding themselves to ``SpaceState.listeners`` /
``ShiftState.listeners`` and should discard themselves on destruction.
"""

from __future__ import annotations

from typing import ClassVar

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QAbstractButton, QApplication


class SpaceState:
    held: bool = False
    # Canvas instances that want to be notified on Space change.
    listeners: ClassVar[set] = set()


class ShiftState:
    held: bool = False
    # Canvas instances that want to be notified on Shift change.
    listeners: ClassVar[set] = set()


class _KeyStateFilter(QObject):
    """Application event filter that tracks whether Space/Shift are held.

    Tracking Shift here (rather than per-widget) keeps the prep-mode cursor color
    synced even when keyboard focus is on another widget (properties panel, main
    window, etc).
    """

    def eventFilter(self, _: QObject, event: QEvent) -> bool:
        if not isinstance(event, QKeyEvent):
            return False
        t = event.type()
        if t == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                SpaceState.held = True
                for canvas in list(SpaceState.listeners):
                    canvas._on_space_changed()
                # Consume the event when a button has focus so spacebar doesn't
                # re-trigger the last clicked button while panning.
                if isinstance(QApplication.focusWidget(), QAbstractButton):
                    return True
            elif event.key() == Qt.Key.Key_Shift:
                ShiftState.held = True
                for canvas in list(ShiftState.listeners):
                    canvas._on_shift_changed()
        elif t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                SpaceState.held = False
                for canvas in list(SpaceState.listeners):
                    canvas._on_space_changed()
            elif event.key() == Qt.Key.Key_Shift:
                ShiftState.held = False
                for canvas in list(ShiftState.listeners):
                    canvas._on_shift_changed()
        return False


def ensure_key_state_filter() -> None:
    """Install the singleton key-state event filter on the application once."""
    app = QApplication.instance()
    if app is not None and getattr(app, "_verso_key_state_filter", None) is None:
        filt = _KeyStateFilter()
        app._verso_key_state_filter = filt
        app.installEventFilter(filt)
