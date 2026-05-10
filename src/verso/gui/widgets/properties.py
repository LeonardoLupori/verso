"""Context-sensitive right-side properties panel.

Contains a QStackedWidget with three pages (one per view mode).
MainWindow switches pages via set_mode().
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec, Section

_MASK_COLORS: dict[str, tuple[int, int, int]] = {
    "White": (255, 255, 255),
    "Cyan": (0, 210, 210),
    "Yellow": (255, 230, 0),
    "Magenta": (220, 0, 220),
    "Green": (80, 220, 80),
    "Orange": (255, 140, 0),
}


def _color_swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(18, 18)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class _ChannelRow(QWidget):
    """One row inside :class:`_BrightnessControls` — visibility, name, color, slider.

    Emits two signals:
      * :attr:`changed` — fires continuously while the slider is dragged
        (used for live canvas updates).
      * :attr:`committed` — fires once the user releases the slider, picks a
        color, or toggles visibility (used for expensive refreshes such as
        the filmstrip).
    """

    changed = pyqtSignal(int, object)    # index, ChannelSpec — live
    committed = pyqtSignal(int, object)  # index, ChannelSpec — on release

    def __init__(self, index: int, spec: ChannelSpec) -> None:
        super().__init__()
        self._index = index
        self._spec = ChannelSpec(
            name=spec.name,
            color=tuple(spec.color),
            scale=spec.scale,
            visible=spec.visible,
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._visible_btn = QToolButton()
        self._visible_btn.setCheckable(True)
        self._visible_btn.setChecked(self._spec.visible)
        self._visible_btn.setFixedSize(22, 22)
        self._visible_btn.setToolTip("Toggle channel visibility")
        self._visible_btn.toggled.connect(self._on_visible)
        self._refresh_visible_btn()
        layout.addWidget(self._visible_btn)

        self._name_label = QLabel(self._spec.name)
        self._name_label.setMinimumWidth(36)
        self._name_label.setMaximumWidth(60)
        self._name_label.setToolTip(self._spec.name)
        layout.addWidget(self._name_label)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick channel color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()
        layout.addWidget(self._color_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(int(round(max(1.0, min(100.0, self._spec.scale * 100.0)))))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self._slider, stretch=1)

    def spec(self) -> ChannelSpec:
        return ChannelSpec(
            name=self._spec.name,
            color=tuple(self._spec.color),
            scale=self._spec.scale,
            visible=self._spec.visible,
        )

    def update_values(self, spec: ChannelSpec) -> None:
        """Update widget state silently from an external ``ChannelSpec``.

        Used when the panel is re-synced from the project — keeps the row's
        widgets stable so an in-progress slider drag isn't interrupted.
        """
        self._spec = ChannelSpec(
            name=spec.name,
            color=tuple(spec.color),
            scale=spec.scale,
            visible=spec.visible,
        )
        target = int(round(max(1.0, min(100.0, self._spec.scale * 100.0))))
        if self._slider.value() != target:
            self._slider.blockSignals(True)
            self._slider.setValue(target)
            self._slider.blockSignals(False)
        if self._visible_btn.isChecked() != self._spec.visible:
            self._visible_btn.blockSignals(True)
            self._visible_btn.setChecked(self._spec.visible)
            self._visible_btn.blockSignals(False)
        self._refresh_visible_btn()
        self._refresh_color_btn()

    def _refresh_visible_btn(self) -> None:
        self._visible_btn.setText("◉" if self._spec.visible else "○")

    def _refresh_color_btn(self) -> None:
        r, g, b = self._spec.color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            " border-radius: 2px;"
        )

    def _on_visible(self, checked: bool) -> None:
        self._spec.visible = bool(checked)
        self._refresh_visible_btn()
        spec = self.spec()
        self.changed.emit(self._index, spec)
        self.committed.emit(self._index, spec)

    def _on_color(self) -> None:
        current = QColor(*self._spec.color)
        color = QColorDialog.getColor(current, self, f"Color for {self._spec.name}")
        if color.isValid():
            self._spec.color = (color.red(), color.green(), color.blue())
            self._refresh_color_btn()
            spec = self.spec()
            self.changed.emit(self._index, spec)
            self.committed.emit(self._index, spec)

    def _on_slider(self, value: int) -> None:
        self._spec.scale = value / 100.0
        spec = self.spec()
        self.changed.emit(self._index, spec)
        # Keyboard / programmatic changes don't go through sliderReleased,
        # so commit immediately when the slider isn't being dragged.
        if not self._slider.isSliderDown():
            self.committed.emit(self._index, spec)

    def _on_slider_released(self) -> None:
        self.committed.emit(self._index, self.spec())


class _BrightnessControls(QWidget):
    """Dynamic per-channel brightness/color/visibility controls.

    Hosts one :class:`_ChannelRow` per project-level
    :class:`~verso.engine.model.project.ChannelSpec`. Emits
    :attr:`channels_changed` whenever the user touches any control.
    """

    channels_changed = pyqtSignal(list)    # live, on every slider tick
    channels_committed = pyqtSignal(list)  # on slider release / discrete edits

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channels: list[ChannelSpec] = []
        self._rows: list[_ChannelRow] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._empty_label = QLabel("No channels")
        self._empty_label.setStyleSheet("color: #888; font-style: italic;")
        self._layout.addWidget(self._empty_label)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        new_specs = [
            ChannelSpec(
                name=c.name,
                color=tuple(c.color),
                scale=c.scale,
                visible=c.visible,
            )
            for c in channels
        ]
        # Fast path: same channel layout (count + names) → update values in
        # place. Avoids destroying the row that owns the slider currently
        # being dragged when the parent re-syncs us.
        same_structure = len(new_specs) == len(self._channels) and all(
            new_specs[i].name == self._channels[i].name
            for i in range(len(new_specs))
        )
        if same_structure and len(self._rows) == len(new_specs):
            self._channels = new_specs
            for i, row in enumerate(self._rows):
                row.update_values(new_specs[i])
            return

        self._channels = new_specs
        self._rebuild()

    def _rebuild(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        self._empty_label.setVisible(not self._channels)
        for i, spec in enumerate(self._channels):
            row = _ChannelRow(i, spec)
            row.changed.connect(self._on_row_changed)
            row.committed.connect(self._on_row_committed)
            self._rows.append(row)
            self._layout.addWidget(row)

    def _snapshot(self) -> list[ChannelSpec]:
        return [
            ChannelSpec(
                name=c.name,
                color=tuple(c.color),
                scale=c.scale,
                visible=c.visible,
            )
            for c in self._channels
        ]

    def _on_row_changed(self, idx: int, spec: ChannelSpec) -> None:
        if 0 <= idx < len(self._channels):
            self._channels[idx] = spec
        self.channels_changed.emit(self._snapshot())

    def _on_row_committed(self, idx: int, spec: ChannelSpec) -> None:
        if 0 <= idx < len(self._channels):
            self._channels[idx] = spec
        self.channels_committed.emit(self._snapshot())


class _OverviewProperties(QWidget):
    """Properties panel content for the Overview view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._title = QLabel("No section selected")
        self._title.setWordWrap(True)
        self._title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title)

        info_box = QGroupBox("Section info")
        info_layout = QFormLayout(info_box)
        self._lbl_file = QLabel("-")
        self._lbl_file.setWordWrap(True)
        self._lbl_serial = QLabel("-")
        self._lbl_scale = QLabel("-")
        info_layout.addRow("File:", self._lbl_file)
        info_layout.addRow("Serial #:", self._lbl_serial)
        info_layout.addRow("Scale:", self._lbl_scale)
        layout.addWidget(info_box)

        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._title.setText("No section selected")
            self._lbl_file.setText("-")
            self._lbl_serial.setText("-")
            self._lbl_scale.setText("-")
            return
        import os

        self._title.setText(os.path.basename(section.original_path))
        self._lbl_file.setText(section.original_path)
        self._lbl_serial.setText(str(section.serial_number))
        self._lbl_scale.setText(f"{section.scale:.4f}")


class _PrepProperties(QWidget):
    """Properties panel content for the Prep view."""

    flip_h_changed = pyqtSignal(bool)
    flip_v_changed = pyqtSignal(bool)
    channels_changed = pyqtSignal(list)
    channels_committed = pyqtSignal(list)
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_color_changed = pyqtSignal(tuple)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
    autodetect_all_requested = pyqtSignal()
    save_mask_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        flip_box = QGroupBox("Flip image")
        flip_layout = QVBoxLayout(flip_box)
        self._flip_h = QCheckBox("Flip horizontal")
        self._flip_h.toggled.connect(self.flip_h_changed)
        flip_layout.addWidget(self._flip_h)
        self._flip_v = QCheckBox("Flip vertical")
        self._flip_v.toggled.connect(self.flip_v_changed)
        flip_layout.addWidget(self._flip_v)
        layout.addWidget(flip_box)

        mask_box = QGroupBox("Mask visibility")
        mask_layout = QVBoxLayout(mask_box)
        self._show_slice = QCheckBox("Show slice mask")
        self._show_slice.setChecked(True)
        self._show_slice.toggled.connect(self.mask_visibility_changed)
        self._show_lr = QCheckBox("Show L/R boundary")
        self._show_lr.setChecked(True)
        self._show_lr.toggled.connect(self.lr_visibility_changed)
        self._negative = QCheckBox("Negative mask")
        self._negative.toggled.connect(self.mask_negative_changed)
        mask_layout.addWidget(self._show_slice)
        mask_layout.addWidget(self._show_lr)
        mask_layout.addWidget(self._negative)

        self._opacity_value = QLabel("0.40")
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(40)
        self._opacity_slider.valueChanged.connect(self._emit_mask_opacity)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity"))
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        opacity_row.addWidget(self._opacity_value)
        mask_layout.addLayout(opacity_row)

        self._mask_color_combo = QComboBox()
        for name, rgb in _MASK_COLORS.items():
            self._mask_color_combo.addItem(_color_swatch_icon(rgb), "", rgb)
            self._mask_color_combo.setItemData(
                self._mask_color_combo.count() - 1,
                name,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._mask_color_combo.currentIndexChanged.connect(self._emit_mask_color)
        self._mask_color_combo.setFixedSize(56, 26)
        self._mask_color_combo.setIconSize(QPixmap(18, 18).size())
        self._mask_color_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 16px; }"
            "QComboBox::drop-down { width: 16px; }"
        )
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color"))
        color_row.addWidget(self._mask_color_combo)
        color_row.addStretch()
        mask_layout.addLayout(color_row)
        layout.addWidget(mask_box)

        edit_box = QGroupBox("Mask editing")
        edit_layout = QVBoxLayout(edit_box)
        self._autodetect_btn = QPushButton("Auto-detect current")
        self._autodetect_btn.clicked.connect(self.autodetect_requested)
        edit_layout.addWidget(self._autodetect_btn)
        self._autodetect_all_btn = QPushButton("Auto-detect all")
        self._autodetect_all_btn.clicked.connect(self.autodetect_all_requested)
        edit_layout.addWidget(self._autodetect_all_btn)

        edit_row = QHBoxLayout()
        self._save_mask_btn = QPushButton("Save mask")
        self._save_mask_btn.clicked.connect(self.save_mask_requested)
        self._clear_mask_btn = QPushButton("Clear")
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested)
        edit_row.addWidget(self._save_mask_btn)
        edit_row.addWidget(self._clear_mask_btn)
        edit_layout.addLayout(edit_row)
        layout.addWidget(edit_box)

        info_box = QGroupBox("Section info")
        info_layout = QFormLayout(info_box)
        self._lbl_dims = QLabel("-")
        info_layout.addRow("Dimensions:", self._lbl_dims)
        layout.addWidget(info_box)

        brightness_box = QGroupBox("Adjust brightness")
        brightness_layout = QVBoxLayout(brightness_box)
        self._brightness = _BrightnessControls()
        self._brightness.channels_changed.connect(self.channels_changed)
        self._brightness.channels_committed.connect(self.channels_committed)
        brightness_layout.addWidget(self._brightness)
        layout.addWidget(brightness_box)

        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._flip_h.blockSignals(True)
            self._flip_h.setChecked(False)
            self._flip_h.blockSignals(False)
            self._flip_v.blockSignals(True)
            self._flip_v.setChecked(False)
            self._flip_v.blockSignals(False)
            self._lbl_dims.setText("-")
            return

        self._flip_h.blockSignals(True)
        self._flip_h.setChecked(section.preprocessing.flip_horizontal)
        self._flip_h.blockSignals(False)

        self._flip_v.blockSignals(True)
        self._flip_v.setChecked(section.preprocessing.flip_vertical)
        self._flip_v.blockSignals(False)

        self._lbl_dims.setText(self._section_dimensions(section))

    def set_mask_negative(self, negative: bool) -> None:
        self._negative.blockSignals(True)
        self._negative.setChecked(negative)
        self._negative.blockSignals(False)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._brightness.set_channels(channels)

    def _emit_mask_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.mask_opacity_changed.emit(opacity)

    def _emit_mask_color(self) -> None:
        self.mask_color_changed.emit(self._mask_color_combo.currentData())

    def _section_dimensions(self, section: Section) -> str:
        try:
            from verso.engine.io.image_io import registration_dimensions

            w, h = registration_dimensions(section)
            return f"{w} x {h}"
        except Exception:
            return "-"


_CP_SHAPES = ["Circle", "Cross", "Square", "Diamond"]
_CP_COLORS: dict[str, tuple[int, int, int]] = {
    "Orange": (255, 96, 0),
    "Cyan": (0, 255, 255),
    "Yellow": (255, 245, 0),
    "Red": (255, 32, 32),
    "White": (255, 255, 255),
    "Magenta": (255, 0, 255),
}


class _AlignProperties(QWidget):
    """Properties panel content for the Align/Warp view."""

    opacity_changed = pyqtSignal(float)
    ap_changed = pyqtSignal(float)
    cp_style_changed = pyqtSignal(int, str, str)  # size, shape, color
    channels_changed = pyqtSignal(list)
    channels_committed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atlas_name = "-"
        self._atlas_loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        atlas_box = QGroupBox("Atlas")
        atlas_layout = QFormLayout(atlas_box)
        self._atlas_label = QLabel("-")
        atlas_layout.addRow("Name:", self._atlas_label)
        layout.addWidget(atlas_box)

        overlay_box = QGroupBox("Overlay")
        overlay_layout = QFormLayout(overlay_box)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.opacity_changed.emit(v / 100.0)
        )
        overlay_layout.addRow("Opacity:", self._opacity_slider)
        layout.addWidget(overlay_box)

        proposal_box = QGroupBox("Proposal")
        proposal_layout = QFormLayout(proposal_box)
        self._proposal_source = QLabel("-")
        self._proposal_source.setWordWrap(True)
        self._proposal_confidence = QLabel("-")
        proposal_layout.addRow("Source:", self._proposal_source)
        proposal_layout.addRow("Confidence:", self._proposal_confidence)
        layout.addWidget(proposal_box)

        self._align_widget = QWidget()
        align_layout = QVBoxLayout(self._align_widget)
        align_layout.setContentsMargins(0, 0, 0, 0)
        align_layout.setSpacing(8)

        ap_box = QGroupBox("AP position")
        ap_box_layout = QVBoxLayout(ap_box)
        ap_box_layout.setSpacing(4)

        ap_form = QFormLayout()
        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(0.0, 20.0)
        self._ap_spin.setSingleStep(0.025)
        self._ap_spin.setSuffix(" mm")
        self._ap_spin.setDecimals(3)
        self._ap_spin.valueChanged.connect(self.ap_changed)
        ap_form.addRow("AP:", self._ap_spin)
        ap_box_layout.addLayout(ap_form)

        self._ap_plot = pg.PlotWidget(background="#1a1a1a")
        self._ap_plot.setFixedHeight(200)
        self._ap_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pi = self._ap_plot.getPlotItem()
        pi.hideAxis("top")
        pi.hideAxis("right")
        pi.getAxis("bottom").setLabel("AP (mm)", color="#aaa")
        pi.getAxis("left").setLabel("Section", color="#aaa")
        pi.getAxis("bottom").setTextPen(pg.mkPen("#aaa"))
        pi.getAxis("left").setTextPen(pg.mkPen("#aaa"))
        pi.invertY(True)
        pi.setMenuEnabled(False)
        ap_box_layout.addWidget(self._ap_plot)
        align_layout.addWidget(ap_box)

        layout.addWidget(self._align_widget)

        self._warp_widget = QWidget()
        warp_layout = QVBoxLayout(self._warp_widget)
        warp_layout.setContentsMargins(0, 0, 0, 0)
        warp_layout.setSpacing(8)

        cp_box = QGroupBox("Control points")
        cp_form = QFormLayout(cp_box)

        self._cp_size_spin = QSpinBox()
        self._cp_size_spin.setRange(4, 30)
        self._cp_size_spin.setValue(10)
        self._cp_size_spin.setSuffix(" px")
        self._cp_size_spin.valueChanged.connect(self._emit_cp_style)
        cp_form.addRow("Size:", self._cp_size_spin)

        self._cp_shape_combo = QComboBox()
        self._cp_shape_combo.addItems(_CP_SHAPES)
        self._cp_shape_combo.currentTextChanged.connect(self._emit_cp_style)
        self._cp_shape_combo.setCurrentText("Cross")
        cp_form.addRow("Shape:", self._cp_shape_combo)

        self._cp_color_combo = QComboBox()
        for name, rgb in _CP_COLORS.items():
            self._cp_color_combo.addItem(_color_swatch_icon(rgb), "", name)
            self._cp_color_combo.setItemData(
                self._cp_color_combo.count() - 1,
                name,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._cp_color_combo.currentIndexChanged.connect(self._emit_cp_style)
        self._cp_color_combo.setCurrentIndex(
            next(
                (i for i, k in enumerate(_CP_COLORS) if k == "Yellow"),
                0,
            )
        )
        self._cp_color_combo.setFixedSize(56, 26)
        self._cp_color_combo.setIconSize(QPixmap(18, 18).size())
        self._cp_color_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 16px; }"
            "QComboBox::drop-down { width: 16px; }"
        )
        cp_form.addRow("Color:", self._cp_color_combo)

        warp_layout.addWidget(cp_box)
        layout.addWidget(self._warp_widget)
        self._warp_widget.setVisible(False)

        brightness_box = QGroupBox("Adjust brightness")
        brightness_layout = QVBoxLayout(brightness_box)
        self._brightness = _BrightnessControls()
        self._brightness.channels_changed.connect(self.channels_changed)
        self._brightness.channels_committed.connect(self.channels_committed)
        brightness_layout.addWidget(self._brightness)
        layout.addWidget(brightness_box)

        layout.addStretch()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._brightness.set_channels(channels)

    def _emit_cp_style(self) -> None:
        self.cp_style_changed.emit(
            self._cp_size_spin.value(),
            self._cp_shape_combo.currentText(),
            self._cp_color_combo.currentData(),
        )

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._ap_spin.blockSignals(True)
            self._ap_spin.setValue(0.0)
            self._ap_spin.blockSignals(False)
            self._update_proposal_status(None)
            return
        ap = section.alignment.ap_position_mm or 0.0
        self._ap_spin.blockSignals(True)
        self._ap_spin.setValue(ap)
        self._ap_spin.blockSignals(False)
        self._update_proposal_status(section)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        self._ap_spin.blockSignals(True)
        self._ap_spin.setValue(ap_mm)
        self._ap_spin.blockSignals(False)

    def _update_proposal_status(self, section: Section | None) -> None:
        if section is None:
            self._proposal_source.setText("-")
            self._proposal_confidence.setText("-")
            return
        source = section.alignment.source
        labels = {
            "deepslice": "DeepSlice suggestion",
            "quicknii_default": "Default proposal",
            "manual": "Manual edit",
        }
        self._proposal_source.setText(labels.get(source, "-"))
        confidence = section.alignment.proposal_confidence
        self._proposal_confidence.setText("-" if confidence is None else f"{confidence:.3f}")

    def set_atlas_name(self, name: str) -> None:
        self._atlas_name = name
        self._refresh_atlas_label()

    def set_atlas_loading(self, loading: bool) -> None:
        self._atlas_loading = loading
        self._refresh_atlas_label()

    def _refresh_atlas_label(self) -> None:
        suffix = " (loading...)" if self._atlas_loading else ""
        self._atlas_label.setText(f"{self._atlas_name}{suffix}")

    def set_align_warp_mode(self, mode: str) -> None:
        is_align = mode == "align"
        self._align_widget.setVisible(is_align)
        self._warp_widget.setVisible(not is_align)

    def set_ap_range(self, min_mm: float, max_mm: float) -> None:
        self._ap_spin.blockSignals(True)
        self._ap_spin.setRange(min_mm, max_mm)
        self._ap_spin.blockSignals(False)

    def set_ap_step(self, step_mm: float) -> None:
        self._ap_spin.blockSignals(True)
        self._ap_spin.setSingleStep(max(step_mm, 0.001))
        self._ap_spin.setDecimals(3 if step_mm < 0.1 else 2)
        self._ap_spin.blockSignals(False)

    def update_ap_plot(self, sections: list, current_index: int) -> None:
        """Redraw the AP position strip chart."""
        from verso.engine.model.alignment import AlignmentStatus

        pi = self._ap_plot.getPlotItem()
        pi.clear()

        if not sections:
            return

        x_complete, y_complete = [], []
        x_progress, y_progress = [], []
        x_none, y_none = [], []

        for i, section in enumerate(sections):
            ap = section.alignment.ap_position_mm
            if ap is None or all(v == 0.0 for v in (section.alignment.anchoring or [])):
                continue
            s = section.alignment.status
            if s == AlignmentStatus.COMPLETE:
                x_complete.append(ap)
                y_complete.append(i)
            elif s == AlignmentStatus.IN_PROGRESS:
                x_progress.append(ap)
                y_progress.append(i)
            else:
                x_none.append(ap)
                y_none.append(i)

        def _add_scatter(xs, ys, color, size=6) -> None:
            if not xs:
                return
            pi.addItem(
                pg.ScatterPlotItem(
                    x=xs,
                    y=ys,
                    symbol="o",
                    size=size,
                    brush=pg.mkBrush(*color),
                    pen=pg.mkPen(None),
                )
            )

        _add_scatter(x_none, y_none, (130, 130, 130, 180))
        _add_scatter(x_progress, y_progress, (255, 193, 7, 220))
        _add_scatter(x_complete, y_complete, (76, 175, 80, 220))

        if 0 <= current_index < len(sections):
            section = sections[current_index]
            ap = section.alignment.ap_position_mm
            if ap is not None and any(v != 0.0 for v in (section.alignment.anchoring or [])):
                pi.addItem(
                    pg.ScatterPlotItem(
                        x=[ap],
                        y=[current_index],
                        symbol="o",
                        size=11,
                        brush=pg.mkBrush(255, 255, 255, 230),
                        pen=pg.mkPen(None),
                    )
                )


class PropertiesPanel(QWidget):
    """Outer container that switches between the three properties pages."""

    flip_h_changed = pyqtSignal(bool)
    flip_v_changed = pyqtSignal(bool)
    channels_changed = pyqtSignal(list)
    channels_committed = pyqtSignal(list)
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_color_changed = pyqtSignal(tuple)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
    autodetect_all_requested = pyqtSignal()
    save_mask_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    opacity_changed = pyqtSignal(float)
    ap_changed = pyqtSignal(float)
    cp_style_changed = pyqtSignal(int, str, str)  # size, shape, color

    _MODES = ("overview", "prep", "align")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._stack = QStackedWidget()
        self._overview_page = _OverviewProperties()
        self._prep_page = _PrepProperties()
        self._align_page = _AlignProperties()

        self._stack.addWidget(self._overview_page)
        self._stack.addWidget(self._prep_page)
        self._stack.addWidget(self._align_page)

        self._prep_page.flip_h_changed.connect(self.flip_h_changed)
        self._prep_page.flip_v_changed.connect(self.flip_v_changed)
        self._prep_page.channels_changed.connect(self.channels_changed)
        self._prep_page.channels_committed.connect(self.channels_committed)
        self._prep_page.mask_visibility_changed.connect(self.mask_visibility_changed)
        self._prep_page.lr_visibility_changed.connect(self.lr_visibility_changed)
        self._prep_page.mask_opacity_changed.connect(self.mask_opacity_changed)
        self._prep_page.mask_color_changed.connect(self.mask_color_changed)
        self._prep_page.mask_negative_changed.connect(self.mask_negative_changed)
        self._prep_page.autodetect_requested.connect(self.autodetect_requested)
        self._prep_page.autodetect_all_requested.connect(self.autodetect_all_requested)
        self._prep_page.save_mask_requested.connect(self.save_mask_requested)
        self._prep_page.clear_mask_requested.connect(self.clear_mask_requested)

        self._align_page.opacity_changed.connect(self.opacity_changed)
        self._align_page.ap_changed.connect(self.ap_changed)
        self._align_page.cp_style_changed.connect(self.cp_style_changed)
        self._align_page.channels_changed.connect(self.channels_changed)
        self._align_page.channels_committed.connect(self.channels_committed)

        layout.addWidget(self._stack)

    def set_mode(self, mode: str) -> None:
        # "warp" shares the align properties page
        page = "align" if mode == "warp" else mode
        self._stack.setCurrentIndex(self._MODES.index(page))

    def update_section(self, section, mode: str) -> None:
        if mode == "overview":
            self._overview_page.update_section(section)
        elif mode == "prep":
            self._prep_page.update_section(section)
        elif mode in ("align", "warp"):
            self._align_page.update_section(section)

    def set_atlas_name(self, name: str) -> None:
        self._align_page.set_atlas_name(name)

    def set_atlas_loading(self, loading: bool) -> None:
        self._align_page.set_atlas_loading(loading)

    def set_align_warp_mode(self, mode: str) -> None:
        self._align_page.set_align_warp_mode(mode)

    def set_ap_range(self, min_mm: float, max_mm: float) -> None:
        self._align_page.set_ap_range(min_mm, max_mm)

    def set_ap_step(self, step_mm: float) -> None:
        self._align_page.set_ap_step(step_mm)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        self._align_page.update_ap_from_anchoring(ap_mm)

    def update_ap_plot(self, sections: list, current_index: int) -> None:
        self._align_page.update_ap_plot(sections, current_index)

    def set_mask_negative(self, negative: bool) -> None:
        self._prep_page.set_mask_negative(negative)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._prep_page.set_channels(channels)
        self._align_page.set_channels(channels)
