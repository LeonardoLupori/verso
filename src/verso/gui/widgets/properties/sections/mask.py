"""Slice mask section."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from verso.gui.widgets.properties._common import (
    color_swatch_style,
    colored_icon,
    eye_icon,
    make_eye_btn,
    make_segmented_buttons,
)


class MaskBox(QGroupBox):
    visibility_changed = pyqtSignal(bool)
    opacity_changed = pyqtSignal(float)
    color_changed = pyqtSignal(tuple)
    negative_changed = pyqtSignal(bool)
    draw_mode_changed = pyqtSignal(str)  # "freehand" | "brush"
    brush_size_changed = pyqtSignal(int)
    autodetect_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    erode_requested = pyqtSignal(int)
    expand_requested = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__("Slice mask")
        layout = QVBoxLayout(self)

        # Row 1: visibility toggle + color picker + negative checkbox
        self._eye_btn = make_eye_btn()
        self._eye_btn.setToolTip("Show / hide slice mask")
        self._eye_btn.toggled.connect(self.visibility_changed)
        self._color_rgb: tuple[int, int, int] = (255, 255, 255)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick mask color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()
        self._negative = QCheckBox("Show negative")
        self._negative.toggled.connect(self.negative_changed)
        row1 = QHBoxLayout()
        row1.addWidget(self._eye_btn)
        row1.addWidget(self._color_btn)
        row1.addStretch()
        row1.addWidget(self._negative)
        layout.addLayout(row1)

        # Row 2: opacity slider
        self._opacity_value = QLabel("0.40")
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(40)
        self._opacity_slider.setMinimumWidth(20)
        self._opacity_slider.valueChanged.connect(self._emit_opacity)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity"))
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        opacity_row.addWidget(self._opacity_value)
        layout.addLayout(opacity_row)

        layout.addSpacing(8)

        # Row 3: draw-mode selector (Freehand / Brush)
        mode_row, self._draw_mode_btns, self._draw_mode_group = make_segmented_buttons(
            self,
            [("freehand", "Freehand"), ("brush", "Brush")],
            tooltips=[
                "Freehand draw areas (F; hold shift to remove)",
                "Paint pixels (B; hold shift to remove)",
            ],
            initial_key="freehand",
        )
        self._draw_mode_group.buttonClicked.connect(self._on_draw_mode_btn_clicked)
        self._draw_mode_btns["freehand"].setIcon(colored_icon("lasso-select.svg", "#ffffff"))
        self._draw_mode_btns["brush"].setIcon(colored_icon("brush.svg", "#ffffff"))
        for _btn in self._draw_mode_btns.values():
            _btn.setIconSize(QSize(14, 14))
        layout.addLayout(mode_row)

        # Row 4: brush size slider
        self._brush_value = QLabel("20")
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(5, 200)
        self._brush_slider.setValue(20)
        self._brush_slider.setMinimumWidth(20)
        self._brush_slider.valueChanged.connect(self._emit_brush_size)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush"))
        brush_row.addWidget(self._brush_slider, stretch=1)
        brush_row.addWidget(self._brush_value)
        layout.addLayout(brush_row)

        # Row 5: erode + expand buttons + amount spinbox
        self._morph_spin = QSpinBox()
        self._morph_spin.setRange(1, 20)
        self._morph_spin.setValue(5)
        self._erode_btn = QPushButton("Erode")
        self._erode_btn.setToolTip("Erode the mask area by a set amount of pixels")
        self._erode_btn.clicked.connect(lambda: self.erode_requested.emit(self._morph_spin.value()))
        self._expand_btn = QPushButton("Expand")
        self._expand_btn.setToolTip("Expand the mask area by a set amount of pixels")
        self._expand_btn.clicked.connect(
            lambda: self.expand_requested.emit(self._morph_spin.value())
        )
        morph_row = QHBoxLayout()
        morph_row.addWidget(self._erode_btn, stretch=1)
        morph_row.addWidget(self._expand_btn, stretch=1)
        morph_row.addWidget(self._morph_spin)
        layout.addLayout(morph_row)

        layout.addSpacing(8)

        # Row 6: auto-detect + clear
        action_row = QHBoxLayout()
        self._autodetect_btn = QPushButton("Auto-detect")
        self._autodetect_btn.setToolTip("Apply adaptive threshold")
        self._autodetect_btn.clicked.connect(self.autodetect_requested)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setIcon(colored_icon("circle-x.svg", "#ffffff"))
        self._clear_btn.setIconSize(QSize(14, 14))
        self._clear_btn.setToolTip("Delete the current slice mask")
        self._clear_btn.clicked.connect(self.clear_requested)
        action_row.addWidget(self._autodetect_btn)
        action_row.addWidget(self._clear_btn)
        layout.addLayout(action_row)

    def set_visible_state(self, visible: bool) -> None:
        self._eye_btn.blockSignals(True)
        self._eye_btn.setChecked(visible)
        self._eye_btn.setIcon(eye_icon(visible))
        self._eye_btn.blockSignals(False)

    def set_negative(self, negative: bool) -> None:
        self._negative.blockSignals(True)
        self._negative.setChecked(negative)
        self._negative.blockSignals(False)

    def set_brush_size(self, size: int) -> None:
        self._brush_slider.setValue(size)

    def set_draw_mode(self, mode: str) -> None:
        """Sync the segmented Freehand/Brush buttons to *mode* without emitting.

        Lets a keyboard shortcut drive the mode while keeping this panel as the
        authoritative display of the current tool.
        """
        key = "brush" if mode == "brush" else "freehand"
        self._draw_mode_btns[key].setChecked(True)

    def _emit_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.opacity_changed.emit(opacity)

    def _emit_brush_size(self) -> None:
        size = self._brush_slider.value()
        self._brush_value.setText(str(size))
        self.brush_size_changed.emit(size)

    def _on_draw_mode_btn_clicked(self, btn: QPushButton) -> None:
        mode = "brush" if btn is self._draw_mode_btns["brush"] else "freehand"
        self.draw_mode_changed.emit(mode)

    def _refresh_color_btn(self) -> None:
        self._color_btn.setStyleSheet(color_swatch_style(self._color_rgb))

    def _on_color(self) -> None:
        current = QColor(*self._color_rgb)
        color = QColorDialog.getColor(current, self, "Mask color")
        if color.isValid():
            self._color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_color_btn()
            self.color_changed.emit(self._color_rgb)
