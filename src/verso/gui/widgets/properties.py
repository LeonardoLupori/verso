"""Context-sensitive right-side properties panel.

Contains a QStackedWidget with three pages (one per view mode).
MainWindow switches pages via set_mode().
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
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

_ICONS_DIR = Path(__file__).parent.parent / "icons"


def _white_icon(name: str) -> QIcon:
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode())
    return QIcon(pixmap)


def _eye_icon(visible: bool) -> QIcon:
    name = "eye.svg" if visible else "eye-off.svg"
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode())
    return QIcon(pixmap)


def _make_eye_btn() -> QPushButton:
    btn = QPushButton()
    btn.setCheckable(True)
    btn.setChecked(True)
    btn.setFixedSize(24, 24)
    btn.setFlat(True)
    btn.setIcon(_eye_icon(True))
    btn.setIconSize(QSize(16, 16))
    btn.toggled.connect(lambda checked, b=btn: b.setIcon(_eye_icon(checked)))
    return btn


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
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_color_changed = pyqtSignal(tuple)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    # Hemisphere subpanel signals
    lr_set_all_left_requested = pyqtSignal()
    lr_set_all_right_requested = pyqtSignal()
    lr_draw_mode_toggled = pyqtSignal(bool)
    lr_apply_requested = pyqtSignal()
    lr_cancel_requested = pyqtSignal()
    lr_clear_requested = pyqtSignal()

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

        # --- Flip image -----------------------------------------------
        flip_box = QGroupBox("Flip image")
        flip_layout = QHBoxLayout(flip_box)
        _flip_btn_style = (
            "QPushButton:checked { background-color: #2a6db5;"
            " border: 1px solid #4a8fd5; border-radius: 3px; }"
        )
        self._flip_h = QPushButton()
        self._flip_h.setIcon(_white_icon("flip-horizontal-2.svg"))
        self._flip_h.setIconSize(QSize(18, 18))
        self._flip_h.setCheckable(True)
        self._flip_h.setToolTip("Flip image horizontally")
        self._flip_h.setStyleSheet(_flip_btn_style)
        self._flip_h.toggled.connect(self.flip_h_changed)
        flip_layout.addWidget(self._flip_h)
        self._flip_v = QPushButton()
        self._flip_v.setIcon(_white_icon("flip-vertical-2.svg"))
        self._flip_v.setIconSize(QSize(18, 18))
        self._flip_v.setCheckable(True)
        self._flip_v.setToolTip("Flip image vertically")
        self._flip_v.setStyleSheet(_flip_btn_style)
        self._flip_v.toggled.connect(self.flip_v_changed)
        flip_layout.addWidget(self._flip_v)
        layout.addWidget(flip_box)

        # --- Slice mask -----------------------------------------------
        mask_box = QGroupBox("Slice mask")
        mask_layout = QVBoxLayout(mask_box)

        # Row 1: visibility toggle + color picker
        self._mask_eye_btn = _make_eye_btn()
        self._mask_eye_btn.setToolTip("Show / hide slice mask")
        self._mask_eye_btn.toggled.connect(self.mask_visibility_changed)
        self._mask_color_rgb: tuple[int, int, int] = (255, 255, 255)
        self._mask_color_btn = QPushButton()
        self._mask_color_btn.setFixedSize(20, 20)
        self._mask_color_btn.setToolTip("Pick mask color")
        self._mask_color_btn.clicked.connect(self._on_mask_color)
        self._refresh_mask_color_btn()
        self._negative = QCheckBox("Show negative")
        self._negative.toggled.connect(self.mask_negative_changed)
        vis_color_row = QHBoxLayout()
        vis_color_row.addWidget(self._mask_eye_btn)
        vis_color_row.addWidget(self._mask_color_btn)
        vis_color_row.addStretch()
        vis_color_row.addWidget(self._negative)
        mask_layout.addLayout(vis_color_row)

        # Row 2: opacity slider
        self._opacity_value = QLabel("0.40")
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(40)
        self._opacity_slider.setMinimumWidth(20)
        self._opacity_slider.valueChanged.connect(self._emit_mask_opacity)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity"))
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        opacity_row.addWidget(self._opacity_value)
        mask_layout.addLayout(opacity_row)

        # Row 4: auto-detect + clear
        action_row = QHBoxLayout()
        self._autodetect_btn = QPushButton("Auto-detect")
        self._autodetect_btn.clicked.connect(self.autodetect_requested)
        self._clear_mask_btn = QPushButton("Clear")
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested)
        action_row.addWidget(self._autodetect_btn)
        action_row.addWidget(self._clear_mask_btn)
        mask_layout.addLayout(action_row)

        layout.addWidget(mask_box)

        # --- Hemisphere -----------------------------------------------
        hemi_box = QGroupBox("Hemisphere")
        hemi_layout = QVBoxLayout(hemi_box)

        # Row 1: visibility toggle + status label
        self._lr_eye_btn = _make_eye_btn()
        self._lr_eye_btn.setToolTip("Show / hide L/R boundary")
        self._lr_eye_btn.toggled.connect(self.lr_visibility_changed)
        self._hemi_status = QLabel("Not set")
        self._hemi_status.setStyleSheet("color: #aaa; font-style: italic;")
        hemi_vis_row = QHBoxLayout()
        hemi_vis_row.addWidget(self._lr_eye_btn)
        hemi_vis_row.addWidget(self._hemi_status, stretch=1)
        hemi_layout.addLayout(hemi_vis_row)

        # Row 2: all-left / all-right
        hemi_uniform_row = QHBoxLayout()
        self._btn_all_left = QPushButton("All left")
        self._btn_all_left.setToolTip("Label the entire section as left hemisphere")
        self._btn_all_left.clicked.connect(self.lr_set_all_left_requested)
        hemi_uniform_row.addWidget(self._btn_all_left)
        self._btn_all_right = QPushButton("All right")
        self._btn_all_right.setToolTip("Label the entire section as right hemisphere")
        self._btn_all_right.clicked.connect(self.lr_set_all_right_requested)
        hemi_uniform_row.addWidget(self._btn_all_right)
        hemi_layout.addLayout(hemi_uniform_row)

        # Row 3: draw separating line
        self._btn_draw_line = QPushButton("Draw separating line")
        self._btn_draw_line.setCheckable(True)
        self._btn_draw_line.setToolTip(
            "Draw a line to split the section into left and right hemispheres. "
            "L/R are determined by the line's direction — drag the start handle "
            "past the end handle to swap sides."
        )
        self._btn_draw_line.toggled.connect(self.lr_draw_mode_toggled)
        hemi_layout.addWidget(self._btn_draw_line)

        # Row 4: apply/cancel toolbar (hidden while draw mode is inactive)
        self._lr_draw_toolbar = QWidget()
        draw_tb = QHBoxLayout(self._lr_draw_toolbar)
        draw_tb.setContentsMargins(0, 0, 0, 0)
        self._btn_lr_apply = QPushButton("✓ Apply")
        self._btn_lr_apply.clicked.connect(self.lr_apply_requested)
        self._btn_lr_cancel = QPushButton("✕ Cancel")
        self._btn_lr_cancel.clicked.connect(self.lr_cancel_requested)
        draw_tb.addWidget(self._btn_lr_apply)
        draw_tb.addWidget(self._btn_lr_cancel)
        self._lr_draw_toolbar.setVisible(False)
        hemi_layout.addWidget(self._lr_draw_toolbar)

        # Row 5: clear
        self._btn_clear_lr = QPushButton("Clear")
        self._btn_clear_lr.setToolTip("Remove the L/R label for this section")
        self._btn_clear_lr.clicked.connect(self.lr_clear_requested)
        hemi_layout.addWidget(self._btn_clear_lr)

        layout.addWidget(hemi_box)

        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._flip_h.blockSignals(True)
            self._flip_h.setChecked(False)
            self._flip_h.blockSignals(False)
            self._flip_v.blockSignals(True)
            self._flip_v.setChecked(False)
            self._flip_v.blockSignals(False)
            return

        self._flip_h.blockSignals(True)
        self._flip_h.setChecked(section.preprocessing.flip_horizontal)
        self._flip_h.blockSignals(False)

        self._flip_v.blockSignals(True)
        self._flip_v.setChecked(section.preprocessing.flip_vertical)
        self._flip_v.blockSignals(False)

    def set_mask_negative(self, negative: bool) -> None:
        self._negative.blockSignals(True)
        self._negative.setChecked(negative)
        self._negative.blockSignals(False)

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_eye_btn.blockSignals(True)
        self._mask_eye_btn.setChecked(visible)
        self._mask_eye_btn.setIcon(_eye_icon(visible))
        self._mask_eye_btn.blockSignals(False)

    def set_lr_status(self, text: str) -> None:
        """Update the hemisphere status label
        (e.g. 'Not set', 'All left', 'All right', 'Line drawn')."""
        self._hemi_status.setText(text)

    def set_lr_draw_active(self, active: bool) -> None:
        """Swap the Hemisphere subpanel between idle and drawing layouts."""
        self._btn_draw_line.blockSignals(True)
        self._btn_draw_line.setChecked(active)
        self._btn_draw_line.blockSignals(False)
        self._btn_draw_line.setText(
            "Drawing — use Apply / Cancel" if active else "Draw separating line"
        )
        self._lr_draw_toolbar.setVisible(active)
        # Disable competing actions while editing the line.
        self._btn_all_left.setEnabled(not active)
        self._btn_all_right.setEnabled(not active)
        self._btn_clear_lr.setEnabled(not active)

    def _emit_mask_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.mask_opacity_changed.emit(opacity)

    def _refresh_mask_color_btn(self) -> None:
        r, g, b = self._mask_color_rgb
        self._mask_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            " border-radius: 2px; }"
        )

    def _on_mask_color(self) -> None:
        current = QColor(*self._mask_color_rgb)
        color = QColorDialog.getColor(current, self, "Mask color")
        if color.isValid():
            self._mask_color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_mask_color_btn()
            self.mask_color_changed.emit(self._mask_color_rgb)

_CP_SHAPES = ["Circle", "Cross", "Square", "Diamond"]


class _AlignProperties(QWidget):
    """Properties panel content for the Align/Warp view."""

    opacity_changed = pyqtSignal(float)
    overlay_color_changed = pyqtSignal(tuple)  # (r, g, b) — outline color
    overlay_mode_changed = pyqtSignal(str)     # "annotation" | "outline" | "reference"
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

        overlay_box = QGroupBox("Overlay")
        overlay_layout = QFormLayout(overlay_box)

        _overlay_specs = [
            ("annotation", "Annotation"),
            ("outline",    "Outline"),
            ("reference",  "Template"),
        ]
        self._overlay_mode_btns: dict[str, QPushButton] = {}
        self._overlay_btn_group = QButtonGroup(self)
        self._overlay_btn_group.setExclusive(True)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        for i, (mode, label) in enumerate(_overlay_specs):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(mode == "annotation")
            btn.setFixedHeight(24)
            n = len(_overlay_specs)
            if i == 0:
                radius = ("border-top-left-radius: 4px; border-bottom-left-radius: 4px;"
                          " border-top-right-radius: 0px; border-bottom-right-radius: 0px;")
                margin = ""
            elif i == n - 1:
                radius = ("border-top-right-radius: 4px; border-bottom-right-radius: 4px;"
                          " border-top-left-radius: 0px; border-bottom-left-radius: 0px;")
                margin = "margin-left: -1px;"
            else:
                radius = "border-radius: 0px;"
                margin = "margin-left: -1px;"
            btn.setStyleSheet(
                f"QPushButton {{ {radius} {margin} padding: 2px 6px; color: #ccc;"
                f" background: #3a3a3a; border: 1px solid #555; }}"
                "QPushButton:checked { background: #1e5a8a; color: #fff;"
                " border-color: #1e5a8a; }"
                f"QPushButton:hover:!checked {{ background: #4a4a4a; }}"
            )
            self._overlay_mode_btns[mode] = btn
            self._overlay_btn_group.addButton(btn)
            mode_row.addWidget(btn)
        self._overlay_btn_group.buttonClicked.connect(self._on_overlay_mode_btn_clicked)
        overlay_layout.addRow(mode_row)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.opacity_changed.emit(v / 100.0)
        )
        overlay_layout.addRow("Opacity:", self._opacity_slider)

        self._outline_color_rgb: tuple[int, int, int] = (255, 255, 255)
        self._outline_color_btn = QPushButton()
        self._outline_color_btn.setFixedSize(20, 20)
        self._outline_color_btn.setToolTip("Pick outline color")
        self._outline_color_btn.clicked.connect(self._on_outline_color)
        self._refresh_outline_color_swatch()
        overlay_layout.addRow("Outline color:", self._outline_color_btn)

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

        self._ap_plot = pg.PlotWidget(background="#1a1a1a")
        self._ap_plot.setFixedHeight(200)
        self._ap_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pi = self._ap_plot.getPlotItem()
        pi.hideAxis("top")
        pi.hideAxis("right")
        pi.getAxis("bottom").setLabel("Section", color="#aaa")
        pi.getAxis("left").setLabel("AP (mm)", color="#aaa")
        pi.getAxis("bottom").setTextPen(pg.mkPen("#aaa"))
        pi.getAxis("left").setTextPen(pg.mkPen("#aaa"))
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
        self._cp_shape_combo.setCurrentText("Cross")
        self._cp_shape_combo.currentTextChanged.connect(self._emit_cp_style)
        cp_form.addRow("Shape:", self._cp_shape_combo)

        self._cp_color_rgb: tuple[int, int, int] = (255, 245, 0)
        self._cp_color_btn = QPushButton()
        self._cp_color_btn.setFixedSize(20, 20)
        self._cp_color_btn.setToolTip("Pick control point color")
        self._cp_color_btn.clicked.connect(self._on_cp_color)
        self._refresh_cp_color_btn()
        cp_form.addRow("Color:", self._cp_color_btn)

        warp_layout.addWidget(cp_box)
        layout.addWidget(self._warp_widget)
        self._warp_widget.setVisible(False)

        layout.addStretch()

    def _refresh_outline_color_swatch(self) -> None:
        r, g, b = self._outline_color_rgb
        self._outline_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            " border-radius: 2px; }"
        )

    def _on_outline_color(self) -> None:
        current = QColor(*self._outline_color_rgb)
        color = QColorDialog.getColor(current, self, "Outline color")
        if color.isValid():
            self._outline_color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_outline_color_swatch()
            self.overlay_color_changed.emit(self._outline_color_rgb)

    def apply_cp_style(self, size: int, shape: str, color: str) -> None:
        """Set CP style widgets silently (no signal emitted)."""
        for widget in (self._cp_size_spin, self._cp_shape_combo):
            widget.blockSignals(True)
        self._cp_size_spin.setValue(size)
        self._cp_shape_combo.setCurrentText(shape)
        if color.startswith("#") and len(color) == 7:
            self._cp_color_rgb = (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        self._refresh_cp_color_btn()
        for widget in (self._cp_size_spin, self._cp_shape_combo):
            widget.blockSignals(False)

    def _refresh_cp_color_btn(self) -> None:
        r, g, b = self._cp_color_rgb
        self._cp_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            " border-radius: 2px; }"
        )

    def _on_cp_color(self) -> None:
        current = QColor(*self._cp_color_rgb)
        color = QColorDialog.getColor(current, self, "Control point color")
        if color.isValid():
            self._cp_color_rgb = (color.red(), color.green(), color.blue())
            self._refresh_cp_color_btn()
            self._emit_cp_style()

    def _emit_cp_style(self) -> None:
        r, g, b = self._cp_color_rgb
        self.cp_style_changed.emit(
            self._cp_size_spin.value(),
            self._cp_shape_combo.currentText(),
            f"#{r:02x}{g:02x}{b:02x}",
        )

    def _on_overlay_mode_btn_clicked(self, btn: QPushButton) -> None:
        for mode, b in self._overlay_mode_btns.items():
            if b is btn:
                self.overlay_mode_changed.emit(mode)
                return

    def set_overlay_mode(self, mode: str) -> None:
        for m, btn in self._overlay_mode_btns.items():
            checked = (m == mode)
            if btn.isChecked() != checked:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)

    def update_section(self, section: Section | None) -> None:
        self._update_proposal_status(section)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        pass

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

    def set_align_warp_mode(self, mode: str) -> None:
        is_align = mode == "align"
        self._align_widget.setVisible(is_align)
        self._warp_widget.setVisible(not is_align)

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
                x_complete.append(i)
                y_complete.append(ap)
            elif s == AlignmentStatus.IN_PROGRESS:
                x_progress.append(i)
                y_progress.append(ap)
            else:
                x_none.append(i)
                y_none.append(ap)

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
                        x=[current_index],
                        y=[ap],
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
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_color_changed = pyqtSignal(tuple)
    mask_negative_changed = pyqtSignal(bool)
    autodetect_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    # Hemisphere subpanel signals (re-exposed from _PrepProperties)
    lr_set_all_left_requested = pyqtSignal()
    lr_set_all_right_requested = pyqtSignal()
    lr_draw_mode_toggled = pyqtSignal(bool)
    lr_apply_requested = pyqtSignal()
    lr_cancel_requested = pyqtSignal()
    lr_clear_requested = pyqtSignal()
    opacity_changed = pyqtSignal(float)
    overlay_color_changed = pyqtSignal(tuple)  # (r, g, b) — outline color
    overlay_mode_changed = pyqtSignal(str)     # "annotation" | "outline" | "reference"
    cp_style_changed = pyqtSignal(int, str, str)  # size, shape, color

    _MODES = ("overview", "prep", "align")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Allow horizontal resize via the dock's splitter handle.  The
        # minimum keeps the inner sliders / combos legible; no maximum so
        # the user can widen the panel as much as they want.
        self.setMinimumWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._stack = QStackedWidget()
        self._overview_page = _OverviewProperties()
        self._prep_page = _PrepProperties()
        self._align_page = _AlignProperties()

        self._stack.addWidget(self._overview_page)
        self._stack.addWidget(self._prep_page)
        self._stack.addWidget(self._align_page)

        self._prep_page.flip_h_changed.connect(self.flip_h_changed)
        self._prep_page.flip_v_changed.connect(self.flip_v_changed)
        self._prep_page.mask_visibility_changed.connect(self.mask_visibility_changed)
        self._prep_page.lr_visibility_changed.connect(self.lr_visibility_changed)
        self._prep_page.mask_opacity_changed.connect(self.mask_opacity_changed)
        self._prep_page.mask_color_changed.connect(self.mask_color_changed)
        self._prep_page.mask_negative_changed.connect(self.mask_negative_changed)
        self._prep_page.autodetect_requested.connect(self.autodetect_requested)
        self._prep_page.clear_mask_requested.connect(self.clear_mask_requested)
        self._prep_page.lr_set_all_left_requested.connect(self.lr_set_all_left_requested)
        self._prep_page.lr_set_all_right_requested.connect(self.lr_set_all_right_requested)
        self._prep_page.lr_draw_mode_toggled.connect(self.lr_draw_mode_toggled)
        self._prep_page.lr_apply_requested.connect(self.lr_apply_requested)
        self._prep_page.lr_cancel_requested.connect(self.lr_cancel_requested)
        self._prep_page.lr_clear_requested.connect(self.lr_clear_requested)

        self._align_page.opacity_changed.connect(self.opacity_changed)
        self._align_page.overlay_color_changed.connect(self.overlay_color_changed)
        self._align_page.overlay_mode_changed.connect(self.overlay_mode_changed)
        self._align_page.cp_style_changed.connect(self.cp_style_changed)

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

    def set_align_warp_mode(self, mode: str) -> None:
        self._align_page.set_align_warp_mode(mode)

    def update_ap_from_anchoring(self, ap_mm: float) -> None:
        self._align_page.update_ap_from_anchoring(ap_mm)

    def update_ap_plot(self, sections: list, current_index: int) -> None:
        self._align_page.update_ap_plot(sections, current_index)

    def set_mask_negative(self, negative: bool) -> None:
        self._prep_page.set_mask_negative(negative)

    def set_mask_visible(self, visible: bool) -> None:
        self._prep_page.set_mask_visible(visible)

    def set_lr_status(self, text: str) -> None:
        self._prep_page.set_lr_status(text)

    def set_lr_draw_active(self, active: bool) -> None:
        self._prep_page.set_lr_draw_active(active)

    def apply_cp_style(self, size: int, shape: str, color: str) -> None:
        """Initialise CP style widgets from saved settings (no signal emitted)."""
        self._align_page.apply_cp_style(size, shape, color)

    def set_overlay_mode(self, mode: str) -> None:
        self._align_page.set_overlay_mode(mode)
