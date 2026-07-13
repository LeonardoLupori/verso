"""Modal dialog for the Export ▸ Quantify actions (intensity / dots / area).

Collects the shared precondition/aggregation/per-slice choices plus the
per-analysis options (channel selection, annotation, dot intensity) and exposes
them as a :class:`~verso.engine.quantification.QuantifyOptions` plus a few extra
accessors the controller reads.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from verso.engine.quantification import QuantifyOptions

_TITLES = {
    "intensity": "Quantify image intensity",
    "area": "Quantify area annotation",
    "dots": "Quantify dots annotation",
}


class QuantifyDialog(QDialog):
    """Collect parameters for an intensity / area / dots quantification."""

    def __init__(
        self,
        kind: str,
        channel_names: list[str],
        annotation_titles: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._channel_names = list(channel_names)
        self.setWindowTitle(_TITLES.get(kind, "Quantify"))
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Annotation (dots / area) ----------------------------------
        self._annotation_combo: QComboBox | None = None
        if kind in ("area", "dots"):
            box = QGroupBox("Point series" if kind == "dots" else "Area annotation")
            form = QFormLayout(box)
            self._annotation_combo = QComboBox()
            self._annotation_combo.addItems(annotation_titles or [])
            form.addRow("Annotation:", self._annotation_combo)
            layout.addWidget(box)

        # --- Channels (intensity / area) -------------------------------
        self._channel_checks: list[QCheckBox] = []
        if kind in ("intensity", "area"):
            layout.addWidget(self._build_channel_box("Channels to quantify"))

        # --- Dot mean intensity ----------------------------------------
        self._dot_intensity_check: QCheckBox | None = None
        self._dot_channel_checks: list[QCheckBox] = []
        self._diameter_spin: QDoubleSpinBox | None = None
        if kind == "dots":
            layout.addWidget(self._build_dot_intensity_box())

        # --- Precondition handling -------------------------------------
        gate_box = QGroupBox("If a step is missing")
        gate_layout = QVBoxLayout(gate_box)
        self._cb_no_mask = QCheckBox("Use sections without a slice mask (quantify whole frame)")
        self._cb_no_cp = QCheckBox("Use sections without warping control points (affine only)")
        gate_layout.addWidget(self._cb_no_mask)
        gate_layout.addWidget(self._cb_no_cp)
        gate_layout.addWidget(self._hint("Sections without an alignment always stop the run."))
        layout.addWidget(gate_box)

        # --- Output ----------------------------------------------------
        out_box = QGroupBox("Output")
        out_layout = QVBoxLayout(out_box)
        self._cb_per_slice = QCheckBox("Separate output per slice")
        self._cb_hemispheres = QCheckBox("Separate left/right hemispheres")
        self._cb_mid = QCheckBox("Also aggregate to mid ontology")
        self._cb_coarse = QCheckBox("Also aggregate to coarse ontology")
        out_layout.addWidget(self._cb_per_slice)
        out_layout.addWidget(self._cb_hemispheres)
        out_layout.addWidget(self._cb_mid)
        out_layout.addWidget(self._cb_coarse)
        out_layout.addWidget(
            self._hint("CSV files are written under the project's exports/ folder.")
        )
        layout.addWidget(out_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #888; font-size: 11px;")
        return label

    def _build_channel_box(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        vbox = QVBoxLayout(box)
        for name in self._channel_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self._channel_checks.append(cb)
            vbox.addWidget(cb)
        return box

    def _build_dot_intensity_box(self) -> QGroupBox:
        box = QGroupBox("Dot intensity (optional)")
        vbox = QVBoxLayout(box)
        self._dot_intensity_check = QCheckBox("Measure mean intensity around each dot")
        vbox.addWidget(self._dot_intensity_check)

        channels_widget = QWidget()
        ch_layout = QVBoxLayout(channels_widget)
        ch_layout.setContentsMargins(18, 0, 0, 0)
        for name in self._channel_names:
            cb = QCheckBox(name)
            self._dot_channel_checks.append(cb)
            ch_layout.addWidget(cb)
        vbox.addWidget(channels_widget)

        form = QFormLayout()
        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(1.0, 999.0)
        self._diameter_spin.setDecimals(0)
        self._diameter_spin.setValue(1.0)
        self._diameter_spin.setSuffix(" px")
        form.addRow("Disk diameter:", self._diameter_spin)
        form.setContentsMargins(18, 0, 0, 0)
        vbox.addLayout(form)

        channels_widget.setEnabled(False)
        self._diameter_spin.setEnabled(False)
        self._dot_intensity_check.toggled.connect(channels_widget.setEnabled)
        self._dot_intensity_check.toggled.connect(self._diameter_spin.setEnabled)
        return box

    # ------------------------------------------------------------------
    def _selected(self, checks: list[QCheckBox]) -> list[str]:
        return [cb.text() for cb in checks if cb.isChecked()]

    def quant_options(self) -> QuantifyOptions:
        """Build the :class:`QuantifyOptions` (``out_dir`` set by the caller)."""
        aggregate: list[str] = []
        if self._cb_mid.isChecked():
            aggregate.append("mid")
        if self._cb_coarse.isChecked():
            aggregate.append("coarse")

        channels: list[str] | None = None
        if self._channel_checks:
            selected = self._selected(self._channel_checks)
            # None means "all"; only narrow when the user deselected some.
            channels = None if len(selected) == len(self._channel_names) else selected

        return QuantifyOptions(
            include_unwarped_affine=self._cb_no_cp.isChecked(),
            include_unmasked_wholeframe=self._cb_no_mask.isChecked(),
            channels=channels,
            aggregate=tuple(aggregate),
            per_slice=self._cb_per_slice.isChecked(),
            split_hemispheres=self._cb_hemispheres.isChecked(),
        )

    def annotation(self) -> str | None:
        return self._annotation_combo.currentText() if self._annotation_combo else None

    def intensity_channels(self) -> list[str] | None:
        """Selected dot-intensity channels, or ``None`` if the option is off."""
        if self._dot_intensity_check and self._dot_intensity_check.isChecked():
            return self._selected(self._dot_channel_checks) or None
        return None

    def dot_diameter(self) -> float:
        return float(self._diameter_spin.value()) if self._diameter_spin else 1.0

    def validate(self) -> str | None:
        """Return an error message if the current selection is unusable."""
        if self._channel_checks and not self._selected(self._channel_checks):
            return "Select at least one channel to quantify."
        if self._annotation_combo is not None and not self._annotation_combo.currentText():
            return "No annotation of this type exists in the project."
        if (
            self._dot_intensity_check
            and self._dot_intensity_check.isChecked()
            and not self._selected(self._dot_channel_checks)
        ):
            return "Select at least one channel for dot intensity, or turn it off."
        return None
