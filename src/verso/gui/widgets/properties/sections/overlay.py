"""Atlas overlay section (Align/Warp views)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSlider,
)

from verso.gui.widgets.properties._common import color_swatch_style, make_segmented_buttons


class OverlayBox(QGroupBox):
    opacity_changed = pyqtSignal(float)
    color_changed = pyqtSignal(tuple)  # (r, g, b) — outline color
    mode_changed = pyqtSignal(str)  # "annotation" | "outline" | "reference"

    def __init__(self) -> None:
        super().__init__("Overlay")
        layout = QFormLayout(self)

        specs = [
            ("annotation", "Annotation"),
            ("outline", "Outline"),
            ("reference", "Template"),
        ]
        mode_row, self._mode_btns, self._mode_group = make_segmented_buttons(
            self, specs, initial_key="annotation"
        )
        self._mode_group.buttonClicked.connect(self._on_mode_btn_clicked)
        layout.addRow(mode_row)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        layout.addRow("Opacity:", self._opacity_slider)

        self._color_rgb: tuple[int, int, int] = (255, 255, 255)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick outline color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_swatch()
        layout.addRow("Outline color:", self._color_btn)

    def emit_current_state(self) -> None:
        """Re-emit all overlay signals from the current widget state.

        Used when switching to this page so the canvas adopts this page's
        overlay settings instead of whatever the previous page left behind.
        """
        for mode, btn in self._mode_btns.items():
            if btn.isChecked():
                self.mode_changed.emit(mode)
                break
        self.opacity_changed.emit(self._opacity_slider.value() / 100.0)
        self.color_changed.emit(self._color_rgb)

    def set_mode(self, mode: str) -> None:
        for m, btn in self._mode_btns.items():
            checked = m == mode
            if btn.isChecked() != checked:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)

    def _refresh_color_swatch(self) -> None:
        self._color_btn.setStyleSheet(color_swatch_style(self._color_rgb))

    def _on_color(self) -> None:
        current = QColor(*self._color_rgb)
        color = QColorDialog.getColor(current, self, "Outline color")
        if color.isValid():
            self._color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_color_swatch()
            self.color_changed.emit(self._color_rgb)

    def _on_mode_btn_clicked(self, btn: QPushButton) -> None:
        for mode, b in self._mode_btns.items():
            if b is btn:
                self.mode_changed.emit(mode)
                return
