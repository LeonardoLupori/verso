"""Control point style section (Warp view)."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
)

from verso.gui.widgets.properties._common import color_swatch_style, colored_icon

_CP_SHAPES = ["Circle", "Cross", "Square", "Diamond"]


class ControlPointsBox(QGroupBox):
    style_changed = pyqtSignal(int, str, str)  # size, shape, color
    autogen_requested = pyqtSignal()  # "Auto-generate" clicked
    edit_params_requested = pyqtSignal()  # "Parameters…" clicked

    def __init__(self) -> None:
        super().__init__("Control points")

        self._size_spin = QSpinBox()
        self._size_spin.setRange(4, 30)
        self._size_spin.setValue(10)
        self._size_spin.setSuffix(" px")
        self._size_spin.valueChanged.connect(self._emit_style)
        # Let the spinbox shrink well below its natural width so it doesn't pin
        # the panel wide; it stays compact (no grid stretch) and a spacer pushes
        # the colour swatch to the right instead.
        self._size_spin.setMinimumWidth(48)
        self._size_spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._shape_combo = QComboBox()
        self._shape_combo.addItems(_CP_SHAPES)
        self._shape_combo.setCurrentText("Cross")
        self._shape_combo.currentTextChanged.connect(self._emit_style)

        self._color_rgb: tuple[int, int, int] = (255, 245, 0)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick control point color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        right_label = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        # Row 0: compact "Size:" spinbox on the left, colour swatch pinned right
        # with a stretch between them, so this row no longer forces a wide panel.
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_row.addWidget(QLabel("Size:"))
        size_row.addWidget(self._size_spin)
        size_row.addStretch(1)
        size_row.addWidget(QLabel("Color:"))
        size_row.addWidget(self._color_btn)
        layout.addLayout(size_row, 0, 0, 1, 4)

        layout.addWidget(QLabel("Shape:"), 1, 0, alignment=right_label)
        layout.addWidget(self._shape_combo, 1, 1, 1, 3)

        self._autogen_btn = QPushButton("Auto-generate")
        self._autogen_btn.setToolTip(
            "Automatically place control points by registering the atlas template "
            "to this section (elastix)."
        )
        self._autogen_btn.clicked.connect(self.autogen_requested)
        self._params_btn = QPushButton()
        self._params_btn.setIcon(colored_icon("settings.svg", "#ffffff"))
        self._params_btn.setIconSize(QSize(16, 16))
        self._params_btn.setToolTip("Edit the automatic registration parameters.")
        self._params_btn.clicked.connect(self.edit_params_requested)
        self._autogen_btn.setMinimumWidth(72)
        layout.addWidget(self._autogen_btn, 2, 0, 1, 3)
        layout.addWidget(self._params_btn, 2, 3)

        # Column 1 stretches so the shape combo and Auto-generate button fill the
        # width; column 3 stays tight so the settings icon button stays compact.
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 0)

    def set_autogen_enabled(self, enabled: bool) -> None:
        """Enable/disable the automatic control-point generation button."""
        self._autogen_btn.setEnabled(enabled)

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
