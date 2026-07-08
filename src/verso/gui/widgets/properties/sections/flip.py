"""Flip image section."""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton

from verso.gui.widgets.properties._common import colored_icon

_FLIP_BTN_STYLE = (
    "QPushButton:checked { background-color: #2a6db5;"
    " border: 1px solid #4a8fd5; border-radius: 3px; }"
)


class FlipBox(QGroupBox):
    flip_h_changed = pyqtSignal(bool)
    flip_v_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__("Flip image")
        layout = QHBoxLayout(self)

        self._flip_h = QPushButton()
        self._flip_h.setIcon(colored_icon("flip-horizontal-2.svg", "#ffffff"))
        self._flip_h.setIconSize(QSize(18, 18))
        self._flip_h.setCheckable(True)
        self._flip_h.setToolTip("Flip image horizontally")
        self._flip_h.setStyleSheet(_FLIP_BTN_STYLE)
        self._flip_h.toggled.connect(self.flip_h_changed)
        layout.addWidget(self._flip_h)

        self._flip_v = QPushButton()
        self._flip_v.setIcon(colored_icon("flip-vertical-2.svg", "#ffffff"))
        self._flip_v.setIconSize(QSize(18, 18))
        self._flip_v.setCheckable(True)
        self._flip_v.setToolTip("Flip image vertically")
        self._flip_v.setStyleSheet(_FLIP_BTN_STYLE)
        self._flip_v.toggled.connect(self.flip_v_changed)
        layout.addWidget(self._flip_v)

    def set_flip_h(self, value: bool) -> None:
        self._flip_h.blockSignals(True)
        self._flip_h.setChecked(value)
        self._flip_h.blockSignals(False)

    def set_flip_v(self, value: bool) -> None:
        self._flip_v.blockSignals(True)
        self._flip_v.setChecked(value)
        self._flip_v.blockSignals(False)
