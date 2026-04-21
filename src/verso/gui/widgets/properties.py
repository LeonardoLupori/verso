"""Context-sensitive right-side properties panel.

Contains a QStackedWidget with three pages (one per view mode).
MainWindow switches pages via set_mode().
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import Section


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
    channel_changed = pyqtSignal(int)
    channel_luminance_changed = pyqtSignal(float, float)
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
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

        flip_box = QGroupBox("Preprocessing")
        flip_layout = QVBoxLayout(flip_box)
        self._flip_h = QCheckBox("Flip horizontal")
        self._flip_h.toggled.connect(self.flip_h_changed)
        flip_layout.addWidget(self._flip_h)
        layout.addWidget(flip_box)

        chan_box = QGroupBox("Channel")
        chan_layout = QVBoxLayout(chan_box)
        self._channel_combo = QComboBox()
        self._channel_combo.currentIndexChanged.connect(self.channel_changed)
        chan_layout.addWidget(self._channel_combo)

        self._red_value = QLabel("1.00")
        self._red_slider = QSlider(Qt.Orientation.Horizontal)
        self._red_slider.setRange(1, 100)
        self._red_slider.setValue(100)
        self._red_slider.valueChanged.connect(self._emit_channel_luminance)
        red_row = QHBoxLayout()
        red_row.addWidget(QLabel("Red"))
        red_row.addWidget(self._red_slider, stretch=1)
        red_row.addWidget(self._red_value)
        chan_layout.addLayout(red_row)

        self._green_value = QLabel("1.00")
        self._green_slider = QSlider(Qt.Orientation.Horizontal)
        self._green_slider.setRange(1, 100)
        self._green_slider.setValue(100)
        self._green_slider.valueChanged.connect(self._emit_channel_luminance)
        green_row = QHBoxLayout()
        green_row.addWidget(QLabel("Green"))
        green_row.addWidget(self._green_slider, stretch=1)
        green_row.addWidget(self._green_value)
        chan_layout.addLayout(green_row)
        layout.addWidget(chan_box)

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
        layout.addWidget(mask_box)

        edit_box = QGroupBox("Mask editing")
        edit_layout = QVBoxLayout(edit_box)
        self._autodetect_btn = QPushButton("Auto-detect")
        self._autodetect_btn.clicked.connect(self.autodetect_requested)
        edit_layout.addWidget(self._autodetect_btn)

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
        self._lbl_channels = QLabel("-")
        info_layout.addRow("Dimensions:", self._lbl_dims)
        info_layout.addRow("Channels:", self._lbl_channels)
        layout.addWidget(info_box)

        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._flip_h.blockSignals(True)
            self._flip_h.setChecked(False)
            self._flip_h.blockSignals(False)
            self._channel_combo.clear()
            self._lbl_dims.setText("-")
            self._lbl_channels.setText("-")
            return

        self._flip_h.blockSignals(True)
        self._flip_h.setChecked(section.preprocessing.flip_horizontal)
        self._flip_h.blockSignals(False)

        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for ch in section.channels or ["Default"]:
            self._channel_combo.addItem(ch)
        self._channel_combo.blockSignals(False)

        self._lbl_dims.setText(self._section_dimensions(section))
        self._lbl_channels.setText(", ".join(section.channels) if section.channels else "-")

    def set_mask_negative(self, negative: bool) -> None:
        self._negative.blockSignals(True)
        self._negative.setChecked(negative)
        self._negative.blockSignals(False)

    def _emit_channel_luminance(self) -> None:
        red = self._red_slider.value() / 100.0
        green = self._green_slider.value() / 100.0
        self._red_value.setText(f"{red:.2f}")
        self._green_value.setText(f"{green:.2f}")
        self.channel_luminance_changed.emit(red, green)

    def _emit_mask_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.mask_opacity_changed.emit(opacity)

    def _section_dimensions(self, section: Section) -> str:
        try:
            from verso.engine.io.image_io import registration_dimensions

            w, h = registration_dimensions(section)
            return f"{w} x {h}"
        except Exception:
            return "-"


_CP_SHAPES = ["Circle", "Cross", "Square", "Diamond"]
_CP_COLORS = ["Orange", "Cyan", "Yellow", "Red", "White", "Magenta"]


class _AlignProperties(QWidget):
    """Properties panel content for the Align/Warp view."""

    opacity_changed = pyqtSignal(float)
    ap_changed = pyqtSignal(float)
    cp_style_changed = pyqtSignal(int, str, str)  # size, shape, color

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
        self._ap_spin.setSingleStep(0.05)
        self._ap_spin.setSuffix(" mm")
        self._ap_spin.setDecimals(2)
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
        cp_form.addRow("Shape:", self._cp_shape_combo)

        self._cp_color_combo = QComboBox()
        self._cp_color_combo.addItems(_CP_COLORS)
        self._cp_color_combo.currentTextChanged.connect(self._emit_cp_style)
        cp_form.addRow("Color:", self._cp_color_combo)

        warp_layout.addWidget(cp_box)
        layout.addWidget(self._warp_widget)
        self._warp_widget.setVisible(False)

        layout.addStretch()

    def _emit_cp_style(self) -> None:
        self.cp_style_changed.emit(
            self._cp_size_spin.value(),
            self._cp_shape_combo.currentText(),
            self._cp_color_combo.currentText(),
        )

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._ap_spin.blockSignals(True)
            self._ap_spin.setValue(0.0)
            self._ap_spin.blockSignals(False)
            return
        ap = section.alignment.ap_position_mm or 0.0
        self._ap_spin.blockSignals(True)
        self._ap_spin.setValue(ap)
        self._ap_spin.blockSignals(False)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        self._ap_spin.blockSignals(True)
        self._ap_spin.setValue(ap_mm)
        self._ap_spin.blockSignals(False)

    def set_atlas_name(self, name: str) -> None:
        self._atlas_label.setText(name)

    def set_atlas_loading(self, loading: bool) -> None:
        if loading:
            self._atlas_label.setText(self._atlas_label.text() + " (loading...)")

    def set_align_warp_mode(self, mode: str) -> None:
        is_align = mode == "align"
        self._align_widget.setVisible(is_align)
        self._warp_widget.setVisible(not is_align)

    def set_ap_range(self, min_mm: float, max_mm: float) -> None:
        self._ap_spin.blockSignals(True)
        self._ap_spin.setRange(min_mm, max_mm)
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
    channel_changed = pyqtSignal(int)
    channel_luminance_changed = pyqtSignal(float, float)
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
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
        self._prep_page.channel_changed.connect(self.channel_changed)
        self._prep_page.channel_luminance_changed.connect(self.channel_luminance_changed)
        self._prep_page.mask_visibility_changed.connect(self.mask_visibility_changed)
        self._prep_page.lr_visibility_changed.connect(self.lr_visibility_changed)
        self._prep_page.mask_opacity_changed.connect(self.mask_opacity_changed)
        self._prep_page.mask_negative_changed.connect(self.mask_negative_changed)
        self._prep_page.autodetect_requested.connect(self.autodetect_requested)
        self._prep_page.save_mask_requested.connect(self.save_mask_requested)
        self._prep_page.clear_mask_requested.connect(self.clear_mask_requested)

        self._align_page.opacity_changed.connect(self.opacity_changed)
        self._align_page.ap_changed.connect(self.ap_changed)
        self._align_page.cp_style_changed.connect(self.cp_style_changed)

        layout.addWidget(self._stack)

    def set_mode(self, mode: str) -> None:
        self._stack.setCurrentIndex(self._MODES.index(mode))

    def update_section(self, section, mode: str) -> None:
        if mode == "overview":
            self._overview_page.update_section(section)
        elif mode == "prep":
            self._prep_page.update_section(section)
        elif mode == "align":
            self._align_page.update_section(section)

    def set_atlas_name(self, name: str) -> None:
        self._align_page.set_atlas_name(name)

    def set_atlas_loading(self, loading: bool) -> None:
        self._align_page.set_atlas_loading(loading)

    def set_align_warp_mode(self, mode: str) -> None:
        self._align_page.set_align_warp_mode(mode)

    def set_ap_range(self, min_mm: float, max_mm: float) -> None:
        self._align_page.set_ap_range(min_mm, max_mm)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        self._align_page.update_ap_from_anchoring(ap_mm)

    def update_ap_plot(self, sections: list, current_index: int) -> None:
        self._align_page.update_ap_plot(sections, current_index)

    def set_mask_negative(self, negative: bool) -> None:
        self._prep_page.set_mask_negative(negative)
