"""Hemisphere (L/R) section."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from verso.gui.widgets.properties._common import (
    color_swatch_style,
    colored_icon,
    make_eye_btn,
)


class HemisphereBox(QGroupBox):
    visibility_changed = pyqtSignal(bool)
    set_all_left_requested = pyqtSignal()
    set_all_right_requested = pyqtSignal()
    draw_mode_toggled = pyqtSignal(bool)
    apply_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    opacity_changed = pyqtSignal(float)
    left_color_changed = pyqtSignal(tuple)
    right_color_changed = pyqtSignal(tuple)

    def __init__(self) -> None:
        super().__init__("Hemisphere")
        layout = QVBoxLayout(self)

        # Row 1: visibility toggle + status label + L/R color pickers
        self._eye_btn = make_eye_btn()
        self._eye_btn.setToolTip("Show / hide L/R boundary")
        self._eye_btn.toggled.connect(self.visibility_changed)
        self._status = QLabel("Not set")
        self._status.setStyleSheet("color: #aaa; font-style: italic;")
        self._left_color_rgb: tuple[int, int, int] = (220, 60, 60)
        self._left_color_btn = QPushButton("L")
        self._left_color_btn.setFixedSize(20, 20)
        self._left_color_btn.setToolTip("Pick left hemisphere color")
        self._left_color_btn.clicked.connect(self._on_left_color)
        self._refresh_left_color_btn()
        self._right_color_rgb: tuple[int, int, int] = (60, 130, 220)
        self._right_color_btn = QPushButton("R")
        self._right_color_btn.setFixedSize(20, 20)
        self._right_color_btn.setToolTip("Pick right hemisphere color")
        self._right_color_btn.clicked.connect(self._on_right_color)
        self._refresh_right_color_btn()
        vis_row = QHBoxLayout()
        vis_row.addWidget(self._eye_btn)
        vis_row.addWidget(self._status, stretch=1)
        vis_row.addWidget(self._left_color_btn)
        vis_row.addWidget(self._right_color_btn)
        layout.addLayout(vis_row)

        # Row 2: opacity slider
        self._opacity_value = QLabel("0.25")
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(25)
        self._opacity_slider.setMinimumWidth(20)
        self._opacity_slider.valueChanged.connect(self._emit_opacity)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity"))
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        opacity_row.addWidget(self._opacity_value)
        layout.addLayout(opacity_row)

        # Row 3: all-left / all-right
        uniform_row = QHBoxLayout()
        self._btn_all_left = QPushButton("All left")
        self._btn_all_left.setToolTip("Label the entire section as left hemisphere")
        self._btn_all_left.clicked.connect(self.set_all_left_requested)
        uniform_row.addWidget(self._btn_all_left)
        self._btn_all_right = QPushButton("All right")
        self._btn_all_right.setToolTip("Label the entire section as right hemisphere")
        self._btn_all_right.clicked.connect(self.set_all_right_requested)
        uniform_row.addWidget(self._btn_all_right)
        layout.addLayout(uniform_row)

        # Row 4: draw line + clear
        draw_row = QHBoxLayout()
        self._btn_draw_line = QPushButton("Draw line")
        self._btn_draw_line.setCheckable(True)
        self._btn_draw_line.setToolTip("Draw a line to split left and right hemispheres")
        self._btn_draw_line.toggled.connect(self.draw_mode_toggled)
        draw_row.addWidget(self._btn_draw_line)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setIcon(colored_icon("circle-x.svg", "#ffffff"))
        self._btn_clear.setIconSize(QSize(14, 14))
        self._btn_clear.setToolTip("Remove the L/R label for this section")
        self._btn_clear.clicked.connect(self.clear_requested)
        draw_row.addWidget(self._btn_clear)
        layout.addLayout(draw_row)

        # Row 5: apply/cancel toolbar (hidden while draw mode is inactive)
        self._draw_toolbar = QWidget()
        tb = QHBoxLayout(self._draw_toolbar)
        tb.setContentsMargins(0, 0, 0, 0)
        self._btn_apply = QPushButton("✓ Apply")
        self._btn_apply.clicked.connect(self.apply_requested)
        self._btn_cancel = QPushButton("✕ Cancel")
        self._btn_cancel.clicked.connect(self.cancel_requested)
        tb.addWidget(self._btn_apply)
        tb.addWidget(self._btn_cancel)
        self._draw_toolbar.setVisible(False)
        layout.addWidget(self._draw_toolbar)

    def set_status(self, text: str) -> None:
        """Update the hemisphere status label."""
        self._status.setText(text)

    def set_draw_active(self, active: bool) -> None:
        """Swap the Hemisphere subpanel between idle and drawing layouts."""
        self._btn_draw_line.blockSignals(True)
        self._btn_draw_line.setChecked(active)
        self._btn_draw_line.blockSignals(False)
        self._btn_draw_line.setText("Drawing..." if active else "Draw line")
        self._draw_toolbar.setVisible(active)
        self._btn_all_left.setEnabled(not active)
        self._btn_all_right.setEnabled(not active)
        self._btn_clear.setEnabled(not active)

    def _emit_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.opacity_changed.emit(opacity)

    def _refresh_left_color_btn(self) -> None:
        self._left_color_btn.setStyleSheet(color_swatch_style(self._left_color_rgb))

    def _refresh_right_color_btn(self) -> None:
        self._right_color_btn.setStyleSheet(color_swatch_style(self._right_color_rgb))

    def _on_left_color(self) -> None:
        current = QColor(*self._left_color_rgb)
        color = QColorDialog.getColor(current, self, "Left hemisphere color")
        if color.isValid():
            self._left_color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_left_color_btn()
            self.left_color_changed.emit(self._left_color_rgb)

    def _on_right_color(self) -> None:
        current = QColor(*self._right_color_rgb)
        color = QColorDialog.getColor(current, self, "Right hemisphere color")
        if color.isValid():
            self._right_color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_right_color_btn()
            self.right_color_changed.emit(self._right_color_rgb)
