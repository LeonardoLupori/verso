"""Floating "Channel settings" dialog.

A non-modal window that hosts the per-channel controls (visibility, name,
color, brightness) and forwards their signals to ``MainWindow``.
Opened from the *Image* menu.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from verso.engine.model.project import ChannelSpec
from verso.gui.widgets.brightness_controls import BrightnessControls


class BrightnessDialog(QDialog):
    """Non-modal floating palette for adjusting channel brightness / color.

    The dialog is parented to ``MainWindow`` so Qt cleans it up at shutdown,
    keeps it stacked above the main window, and lends it the VERSO window
    icon. Closing hides it (default non-modal QDialog behaviour); the same
    instance is reused across opens so the user's position and any
    in-progress edits are preserved during the session.

    Chrome and layout match the other dialogs (see ``export_images``,
    ``quantify``): plain window flags — the standard title bar and close
    button rather than the slimmer icon-less ``Qt.Tool`` chrome — a muted
    hint, and a closing ``QDialogButtonBox``. No group box, since the window
    title already says "Channel settings".
    """

    channels_changed = pyqtSignal(list)
    channels_committed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel settings")
        self.setModal(False)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._controls = BrightnessControls(self)
        self._controls.channels_changed.connect(self.channels_changed)
        self._controls.channels_committed.connect(self.channels_committed)
        layout.addWidget(self._controls)

        hint = QLabel("Applies to every section in the project. Double-click a name to rename it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(hint)

        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        # Not a dialog default button: otherwise Return from the name field or
        # the gamma spin-box would close the palette mid-edit.
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._controls.set_channels(channels)
