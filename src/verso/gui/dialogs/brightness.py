"""Floating "Adjust channels / brightness" dialog.

A non-modal tool window that hosts the per-channel brightness controls and
forwards their signals to ``MainWindow``. Opened from the *Images* menu.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QWidget

from verso.engine.model.project import ChannelSpec
from verso.gui.widgets.brightness_controls import BrightnessControls


class BrightnessDialog(QDialog):
    """Non-modal floating palette for adjusting channel brightness / color.

    The dialog is parented to ``MainWindow`` so Qt cleans it up at shutdown.
    Closing the window hides it (default non-modal QDialog behaviour); the
    same instance is reused across opens so the user's position and any
    in-progress edits are preserved during the session.
    """

    channels_changed = pyqtSignal(list)
    channels_committed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust channels / brightness")
        # Tool window: floats above main, no taskbar entry, slim title bar,
        # does not steal focus from the canvas while the user edits.
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowTitleHint
        )
        self.setModal(False)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._controls = BrightnessControls(self)
        self._controls.channels_changed.connect(self.channels_changed)
        self._controls.channels_committed.connect(self.channels_committed)
        layout.addWidget(self._controls)
        layout.addStretch()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._controls.set_channels(channels)
