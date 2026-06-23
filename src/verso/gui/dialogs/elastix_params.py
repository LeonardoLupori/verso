"""Dialog for editing automatic (elastix) control-point parameters."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.elastix import ElastixParams


class ElastixParamsDialog(QDialog):
    """Edit the per-project :class:`ElastixParams` used for auto control points."""

    def __init__(self, params: ElastixParams | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Automatic registration parameters")
        self.setMinimumWidth(400)
        params = params or ElastixParams()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Flexibility ------------------------------------------------
        flex_box = QGroupBox("Flexibility")
        flex_form = QFormLayout(flex_box)

        self._grid_spacing = QSpinBox()
        self._grid_spacing.setRange(8, 512)
        self._grid_spacing.setSingleStep(8)
        self._grid_spacing.setSuffix(" px")
        self._grid_spacing.setValue(params.grid_spacing)
        self._grid_spacing.setToolTip(
            "B-spline control-point spacing. Smaller = more local/flexible warp."
        )
        flex_form.addRow("Grid spacing:", self._grid_spacing)

        self._n_resolutions = QSpinBox()
        self._n_resolutions.setRange(1, 6)
        self._n_resolutions.setValue(params.n_resolutions)
        self._n_resolutions.setToolTip("Coarse-to-fine pyramid levels.")
        flex_form.addRow("Resolutions:", self._n_resolutions)

        self._max_iterations = QSpinBox()
        self._max_iterations.setRange(10, 2000)
        self._max_iterations.setSingleStep(10)
        self._max_iterations.setValue(params.max_iterations)
        self._max_iterations.setToolTip("Optimizer iterations per resolution level.")
        flex_form.addRow("Max iterations:", self._max_iterations)

        layout.addWidget(flex_box)

        # --- Sampling ---------------------------------------------------
        sampling_box = QGroupBox("Sampling")
        sampling_form = QFormLayout(sampling_box)

        self._n_samples = QSpinBox()
        self._n_samples.setRange(128, 16384)
        self._n_samples.setSingleStep(128)
        self._n_samples.setValue(params.n_samples)
        self._n_samples.setToolTip("Random spatial samples per iteration.")
        sampling_form.addRow("Spatial samples:", self._n_samples)

        self._registration_scale = QDoubleSpinBox()
        self._registration_scale.setRange(0.1, 1.0)
        self._registration_scale.setSingleStep(0.1)
        self._registration_scale.setDecimals(2)
        self._registration_scale.setValue(params.registration_scale)
        self._registration_scale.setToolTip(
            "Downsample factor for registration (1.0 = full working resolution)."
        )
        sampling_form.addRow("Registration scale:", self._registration_scale)

        layout.addWidget(sampling_box)

        # --- Mask -------------------------------------------------------
        mask_box = QGroupBox("Tissue mask")
        mask_form = QFormLayout(mask_box)

        self._mask_dilation_register = QSpinBox()
        self._mask_dilation_register.setRange(0, 200)
        self._mask_dilation_register.setSuffix(" px")
        self._mask_dilation_register.setValue(params.mask_dilation_register)
        self._mask_dilation_register.setToolTip(
            "Dilate the tissue mask by this radius before using it to gate the "
            "registration, so edge tissue still contributes."
        )
        mask_form.addRow("Registration dilation:", self._mask_dilation_register)

        self._mask_dilation_cp = QSpinBox()
        self._mask_dilation_cp.setRange(0, 400)
        self._mask_dilation_cp.setSuffix(" px")
        self._mask_dilation_cp.setValue(params.mask_dilation_cp)
        self._mask_dilation_cp.setToolTip(
            "Dilate the tissue mask by this (larger) radius to decide where new "
            "control points may be created."
        )
        mask_form.addRow("Control-point dilation:", self._mask_dilation_cp)

        layout.addWidget(mask_box)
        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> ElastixParams:
        return ElastixParams(
            grid_spacing=self._grid_spacing.value(),
            n_resolutions=self._n_resolutions.value(),
            max_iterations=self._max_iterations.value(),
            n_samples=self._n_samples.value(),
            registration_scale=float(self._registration_scale.value()),
            mask_dilation_register=self._mask_dilation_register.value(),
            mask_dilation_cp=self._mask_dilation_cp.value(),
        )
