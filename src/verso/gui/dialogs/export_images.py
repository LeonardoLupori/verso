"""Modal dialog for the "Export → Images with atlas overlay" action.

Collects user choices (scope, burn vs separate, color/opacity, output size,
outline thickness) and exposes them as :class:`ExportOptions` for the
``MainWindow`` handler to consume.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from verso.engine.io.export_images import ExportOptions


class ExportImagesDialog(QDialog):
    """Collect parameters for an images-with-overlay export."""

    def __init__(
        self,
        n_selected: int,
        n_total: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export images with atlas overlay")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._color = QColor(255, 255, 255)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Scope -----------------------------------------------------
        scope_box = QGroupBox("Sections to export")
        scope_layout = QVBoxLayout(scope_box)
        self._radio_selected = QRadioButton(f"Selected sections in overview table({n_selected})")
        self._radio_all = QRadioButton(f"All sections ({n_total})")
        scope_group = QButtonGroup(self)
        scope_group.addButton(self._radio_selected)
        scope_group.addButton(self._radio_all)
        if n_selected > 0:
            self._radio_selected.setChecked(True)
        else:
            self._radio_selected.setEnabled(False)
            self._radio_all.setChecked(True)
        scope_layout.addWidget(self._radio_selected)
        scope_layout.addWidget(self._radio_all)
        layout.addWidget(scope_box)

        # --- Output style ----------------------------------------------
        style_box = QGroupBox("Output style")
        style_layout = QVBoxLayout(style_box)
        self._radio_burn = QRadioButton("Burn overlay into image")
        self._radio_separate = QRadioButton("Separate overlay PNG (transparent background)")
        self._radio_burn.setChecked(True)
        style_group = QButtonGroup(self)
        style_group.addButton(self._radio_burn)
        style_group.addButton(self._radio_separate)
        style_layout.addWidget(self._radio_burn)
        style_layout.addWidget(self._radio_separate)
        layout.addWidget(style_box)

        # --- Overlay appearance ----------------------------------------
        appearance_box = QGroupBox("Overlay appearance")
        appearance_layout = QFormLayout(appearance_box)

        color_row = QHBoxLayout()
        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(28, 22)
        self._color_swatch.setStyleSheet(self._swatch_qss(self._color))
        self._color_button = QPushButton("Choose…")
        self._color_button.clicked.connect(self._on_pick_color)
        color_row.addWidget(self._color_swatch)
        color_row.addWidget(self._color_button)
        color_row.addStretch()
        color_widget = QWidget()
        color_widget.setLayout(color_row)
        appearance_layout.addRow("Color:", color_widget)

        opacity_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_value = QLabel("100%")
        self._opacity_value.setMinimumWidth(36)
        self._opacity_slider.valueChanged.connect(lambda v: self._opacity_value.setText(f"{v}%"))
        opacity_row.addWidget(self._opacity_slider, 1)
        opacity_row.addWidget(self._opacity_value)
        opacity_widget = QWidget()
        opacity_widget.setLayout(opacity_row)
        appearance_layout.addRow("Opacity:", opacity_widget)

        self._thickness_spin = QSpinBox()
        self._thickness_spin.setRange(1, 8)
        self._thickness_spin.setValue(1)
        self._thickness_spin.setSuffix(" px")
        appearance_layout.addRow("Outline thickness:", self._thickness_spin)

        # Smoothing (Gaussian sigma along contour, in atlas-sampling pixels).
        smoothing_row = QHBoxLayout()
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self._smoothing_slider.setRange(0, 100)  # tenths of a unit: 0.0–10.0
        self._smoothing_slider.setValue(15)
        self._smoothing_value = QLabel("1.5")
        self._smoothing_value.setMinimumWidth(48)
        self._smoothing_slider.valueChanged.connect(
            lambda v: self._smoothing_value.setText(f"{v / 10:.1f}")
        )
        smoothing_row.addWidget(self._smoothing_slider, 1)
        smoothing_row.addWidget(self._smoothing_value)
        smoothing_widget = QWidget()
        smoothing_widget.setLayout(smoothing_row)
        appearance_layout.addRow("Smoothing:", smoothing_widget)

        layout.addWidget(appearance_box)

        # --- Output resolution -----------------------------------------
        size_box = QGroupBox("Output resolution")
        size_layout = QFormLayout(size_box)
        self._long_side_spin = QSpinBox()
        self._long_side_spin.setRange(500, 20000)
        self._long_side_spin.setSingleStep(500)
        self._long_side_spin.setValue(4000)
        self._long_side_spin.setSuffix(" px")
        size_layout.addRow("Long side:", self._long_side_spin)
        layout.addWidget(size_box)

        # --- Buttons ---------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    @staticmethod
    def _swatch_qss(color: QColor) -> str:
        return f"background: {color.name()}; border: 1px solid #555; border-radius: 3px;"

    def _on_pick_color(self) -> None:
        new = QColorDialog.getColor(self._color, self, "Overlay color")
        if new.isValid():
            self._color = new
            self._color_swatch.setStyleSheet(self._swatch_qss(new))

    # ------------------------------------------------------------------
    def export_all(self) -> bool:
        """True if the user chose to export every section in the project."""
        return self._radio_all.isChecked()

    def options(self) -> ExportOptions:
        return ExportOptions(
            burn_overlay=self._radio_burn.isChecked(),
            overlay_color=(
                self._color.red(),
                self._color.green(),
                self._color.blue(),
            ),
            overlay_opacity=self._opacity_slider.value() / 100.0,
            long_side=int(self._long_side_spin.value()),
            outline_thickness=int(self._thickness_spin.value()),
            contour_smoothing=self._smoothing_slider.value() / 10.0,
        )
