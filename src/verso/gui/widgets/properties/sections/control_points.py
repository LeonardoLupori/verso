"""Control point style section (Warp view)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
)

from verso.gui.widgets.properties._common import color_swatch_style

_CP_SHAPES = ["Circle", "Cross", "Square", "Diamond"]


class ControlPointsBox(QGroupBox):
    style_changed = pyqtSignal(int, str, str)  # size, shape, color

    def __init__(self) -> None:
        super().__init__("Control points")
        layout = QFormLayout(self)

        self._size_spin = QSpinBox()
        self._size_spin.setRange(4, 30)
        self._size_spin.setValue(10)
        self._size_spin.setSuffix(" px")
        self._size_spin.valueChanged.connect(self._emit_style)
        layout.addRow("Size:", self._size_spin)

        self._shape_combo = QComboBox()
        self._shape_combo.addItems(_CP_SHAPES)
        self._shape_combo.setCurrentText("Cross")
        self._shape_combo.currentTextChanged.connect(self._emit_style)
        layout.addRow("Shape:", self._shape_combo)

        self._color_rgb: tuple[int, int, int] = (255, 245, 0)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick control point color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()
        layout.addRow("Color:", self._color_btn)

    def apply_style(self, size: int, shape: str, color: str) -> None:
        """Set CP style widgets silently (no signal emitted)."""
        for widget in (self._size_spin, self._shape_combo):
            widget.blockSignals(True)
        self._size_spin.setValue(size)
        self._shape_combo.setCurrentText(shape)
        if color.startswith("#") and len(color) == 7:
            self._color_rgb = (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        self._refresh_color_btn()
        for widget in (self._size_spin, self._shape_combo):
            widget.blockSignals(False)

    def _refresh_color_btn(self) -> None:
        self._color_btn.setStyleSheet(color_swatch_style(self._color_rgb))

    def _on_color(self) -> None:
        current = QColor(*self._color_rgb)
        color = QColorDialog.getColor(current, self, "Control point color")
        if color.isValid():
            self._color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_color_btn()
            self._emit_style()

    def _emit_style(self) -> None:
        r, g, b = self._color_rgb
        self.style_changed.emit(
            self._size_spin.value(),
            self._shape_combo.currentText(),
            f"#{r:02x}{g:02x}{b:02x}",
        )
