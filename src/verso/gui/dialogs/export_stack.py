"""Modal dialog for the "Export → Aligned section stack" action.

Collects scope (selected vs all) and output scale, and exposes them as
:class:`ExportStackOptions` for the ``MainWindow`` handler to consume.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.io.export_stack import ExportStackOptions


class ExportStackDialog(QDialog):
    """Collect parameters for an aligned-stack export."""

    def __init__(
        self,
        n_selected: int,
        n_total: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export aligned section stack")
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        info = QLabel(
            "Each section is resampled onto a straight atlas slice — the warp and "
            "the affine rotation/shear/stretch are undone — and written as a "
            "multi-channel, multi-page TIFF aligned to the atlas."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- Scope -----------------------------------------------------
        scope_box = QGroupBox("Sections to export")
        scope_layout = QVBoxLayout(scope_box)
        self._radio_selected = QRadioButton(f"Selected sections in overview table ({n_selected})")
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

        # --- Background ------------------------------------------------
        bg_box = QGroupBox("Background outside slice mask")
        bg_layout = QVBoxLayout(bg_box)
        self._radio_bg_none = QRadioButton("Keep full section (black border)")
        self._radio_bg_black = QRadioButton("Mask to black")
        self._radio_bg_white = QRadioButton("Mask to white")
        self._radio_bg_none.setChecked(True)
        bg_group = QButtonGroup(self)
        bg_group.addButton(self._radio_bg_none)
        bg_group.addButton(self._radio_bg_black)
        bg_group.addButton(self._radio_bg_white)
        bg_layout.addWidget(self._radio_bg_none)
        bg_layout.addWidget(self._radio_bg_black)
        bg_layout.addWidget(self._radio_bg_white)
        layout.addWidget(bg_box)

        # --- Merge -----------------------------------------------------
        self._merge_check = QCheckBox(
            "Merge sections with the same slice index (max projection)"
        )
        layout.addWidget(self._merge_check)

        # --- Buttons ---------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def export_all(self) -> bool:
        """True if the user chose to export every section in the project."""
        return self._radio_all.isChecked()

    def options(self) -> ExportStackOptions:
        if self._radio_bg_black.isChecked():
            background: str | None = "black"
        elif self._radio_bg_white.isChecked():
            background = "white"
        else:
            background = None
        return ExportStackOptions(
            scale=float(self._scale_spin.value()),
            all_sections=self._radio_all.isChecked(),
            background=background,
            merge_by_slice_index=self._merge_check.isChecked(),
        )
