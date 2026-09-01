"""Modal dialog for the "Export → Images with atlas overlay" action.

Collects user choices (scope, burn vs separate, overlay style, color/opacity,
scale, smoothing, outline thickness, section colormap) and exposes them as
:class:`ExportOptions` for the ``MainWindow`` handler to consume.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from verso.engine.colormaps import NONE as COLORMAP_NONE
from verso.engine.colormaps import apply_colormap, colormap_names
from verso.engine.io.export_images import ExportOptions
from verso.engine.levels import Levels, apply_levels
from verso.gui.dialogs.export_advanced import ExportAdvancedDialog
from verso.gui.utils import ndarray_to_pixmap

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class _ColormapPreviewDialog(QDialog):
    """Shows one section rendered through the chosen colormap, at preview size."""

    def __init__(self, pixmap, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Colormap preview — {title}")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ExportImagesDialog(QDialog):
    """Collect parameters for an images-with-overlay export."""

    def __init__(
        self,
        n_selected: int,
        n_total: int,
        parent: QWidget | None = None,
        preview_source: Callable[[], np.ndarray] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export images with atlas overlay")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._color = QColor(255, 255, 255)
        # Renders one section as un-colormapped RGB at preview size. Called at
        # most once — the result is cached so switching colormaps is just a LUT.
        self._preview_source = preview_source
        self._preview_base: np.ndarray | None = None
        self._levels = Levels()

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

        # --- Overlay style ---------------------------------------------
        overlay_box = QGroupBox("Overlay style")
        overlay_layout = QVBoxLayout(overlay_box)
        self._radio_outline = QRadioButton("Region outlines")
        self._radio_filled = QRadioButton("Filled regions (atlas colors)")
        self._radio_outline.setChecked(True)
        overlay_group = QButtonGroup(self)
        overlay_group.addButton(self._radio_outline)
        overlay_group.addButton(self._radio_filled)
        self._radio_outline.toggled.connect(self._on_overlay_style_changed)
        overlay_layout.addWidget(self._radio_outline)
        overlay_layout.addWidget(self._radio_filled)
        layout.addWidget(overlay_box)

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

        # Smoothing: 0 (no blur) .. 100 (very smooth contours). Maps to an SDF
        # Gaussian sigma in atlas-voxel units inside the engine.
        smoothing_row = QHBoxLayout()
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self._smoothing_slider.setRange(0, 100)
        self._smoothing_slider.setValue(30)
        self._smoothing_value = QLabel("30")
        self._smoothing_value.setMinimumWidth(48)
        self._smoothing_slider.valueChanged.connect(lambda v: self._smoothing_value.setText(f"{v}"))
        smoothing_row.addWidget(self._smoothing_slider, 1)
        smoothing_row.addWidget(self._smoothing_value)
        smoothing_widget = QWidget()
        smoothing_widget.setLayout(smoothing_row)
        appearance_layout.addRow("Smoothing:", smoothing_widget)

        layout.addWidget(appearance_box)

        # --- Section colormap ------------------------------------------
        # Recolors the section image itself (not the overlay). The "none" entry
        # keeps the per-channel composite exactly as the GUI shows it, so its
        # label spells that out rather than reading as "no colors".
        colormap_box = QGroupBox("Section colormap")
        colormap_layout = QFormLayout(colormap_box)
        self._colormap_combo = QComboBox()
        for name in colormap_names():
            label = "Original channel colors" if name == COLORMAP_NONE else name
            self._colormap_combo.addItem(label, name)
        self._invert_check = QCheckBox("Invert colormap")
        self._colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        self._preview_button = QPushButton("Preview…")
        self._preview_button.clicked.connect(self._on_preview)
        colormap_row = QHBoxLayout()
        colormap_row.addWidget(self._colormap_combo, 1)
        colormap_row.addWidget(self._preview_button)
        colormap_widget = QWidget()
        colormap_widget.setLayout(colormap_row)
        colormap_layout.addRow("Colormap:", colormap_widget)
        colormap_layout.addRow("", self._invert_check)

        # Advanced levels (min/max/gamma), edited against a live preview and
        # summarised here so the choice stays visible from the main dialog.
        advanced_row = QHBoxLayout()
        self._advanced_button = QPushButton("Advanced…")
        self._advanced_button.setEnabled(preview_source is not None)
        self._advanced_button.clicked.connect(self._on_advanced)
        self._levels_label = QLabel()
        self._levels_label.setStyleSheet("color: #999;")
        advanced_row.addWidget(self._advanced_button)
        advanced_row.addWidget(self._levels_label, 1)
        advanced_widget = QWidget()
        advanced_widget.setLayout(advanced_row)
        colormap_layout.addRow("Levels:", advanced_widget)

        self._on_colormap_changed()
        self._update_levels_label()
        layout.addWidget(colormap_box)

        # --- Output resolution -----------------------------------------
        size_box = QGroupBox("Output resolution")
        size_layout = QFormLayout(size_box)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.0, 16.0)
        self._scale_spin.setSingleStep(1.0)
        self._scale_spin.setDecimals(1)
        self._scale_spin.setValue(4.0)
        self._scale_spin.setSuffix("×")
        size_layout.addRow("Scale (× atlas):", self._scale_spin)
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

    def _on_colormap_changed(self) -> None:
        """Gate invert + preview on an actual colormap being chosen."""
        active = self._colormap_combo.currentData() != COLORMAP_NONE
        self._invert_check.setEnabled(active)
        self._preview_button.setEnabled(active and self._preview_source is not None)

    def _ensure_preview_base(self) -> np.ndarray | None:
        """Return the cached un-adjusted preview render, loading it on first use.

        Returns ``None`` (after warning the user) if the section cannot be read,
        so callers can simply bail out.
        """
        if self._preview_base is not None:
            return self._preview_base
        if self._preview_source is None:
            return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._preview_base = self._preview_source()
        except Exception as exc:
            _log.exception("Colormap preview failed")
            QMessageBox.warning(
                self,
                "Preview",
                f"Could not render a preview section:\n\n{exc}",
            )
            return None
        finally:
            QApplication.restoreOverrideCursor()
        return self._preview_base

    def _on_preview(self) -> None:
        """Render the preview section as it will be exported and show it."""
        base = self._ensure_preview_base()
        if base is None:
            return

        name = str(self._colormap_combo.currentData())
        rgb = apply_colormap(
            apply_levels(base, self._levels), name, invert=self._invert_check.isChecked()
        )
        title = self._colormap_combo.currentText()
        if self._invert_check.isChecked():
            title += " (inverted)"
        _ColormapPreviewDialog(ndarray_to_pixmap(rgb), title, self).exec()

    def _on_advanced(self) -> None:
        """Edit the levels against a live preview, then keep them if accepted."""
        base = self._ensure_preview_base()
        if base is None:
            return

        dlg = ExportAdvancedDialog(
            base,
            self._levels,
            colormap=str(self._colormap_combo.currentData()),
            colormap_invert=self._invert_check.isChecked(),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._levels = dlg.levels()
            _log.debug("Export levels set to %s", self._levels)
            self._update_levels_label()

    def _update_levels_label(self) -> None:
        """Summarise the current levels beside the Advanced button."""
        if self._levels.is_identity:
            self._levels_label.setText("default")
        else:
            self._levels_label.setText(
                f"min {self._levels.black} · max {self._levels.white} · γ {self._levels.gamma:.2f}"
            )

    def _on_overlay_style_changed(self) -> None:
        """Disable line-only controls (color, thickness) for filled regions."""
        outline = self._radio_outline.isChecked()
        self._color_button.setEnabled(outline)
        self._thickness_spin.setEnabled(outline)

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
            scale=float(self._scale_spin.value()),
            smoothing=float(self._smoothing_slider.value()),
            overlay_style="outline" if self._radio_outline.isChecked() else "filled",
            outline_thickness=int(self._thickness_spin.value()),
            colormap=str(self._colormap_combo.currentData()),
            colormap_invert=self._invert_check.isChecked(),
            levels=self._levels,
        )
