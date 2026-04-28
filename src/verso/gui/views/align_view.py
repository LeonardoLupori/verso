"""Align/Warp view — atlas registration and nonlinear refinement canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Section
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.navigator import NavigatorPanel

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume

# Scale increment per button click (2 %, matching QuickNII)
_SCALE_STEP = 1.02
# Translation per click in atlas voxels (~625 µm for Allen 25 µm atlas)
_MOVE_STEP = 5
# In-plane rotation per click, in degrees.
_ROTATE_STEP_DEG = 1.0


class AlignView(QWidget):
    """Canvas view for atlas alignment (Align mode) and warp (Warp mode)."""

    section_modified = pyqtSignal()
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "align" or "warp"
    reverse_requested = pyqtSignal()
    deepslice_requested = pyqtSignal()
    default_proposal_requested = pyqtSignal()

    # Pixel distance threshold (in normalised units × display size) for
    # picking an existing control point
    _CP_PICK_RADIUS = 16  # px

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section: Section | None = None
        self._raw_image = None
        self._atlas: AtlasVolume | None = None
        self._mode = "align"
        self._outline_mode = False
        # Warp interaction state
        self._cp_hovered: int = -1    # index of CP under cursor (-1 = none)
        self._cp_dragging: int = -1   # index of CP currently being dragged
        self._cp_drag_start_norm: tuple[float, float] | None = None
        self._cp_drag_start_dst: tuple[float, float] | None = None
        # CP style (synced from properties panel)
        self._cp_size = 10
        self._cp_shape = "Cross"
        self._cp_color = "Yellow"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_mode_bar())

        # Body: navigator + canvas side by side
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._navigator = NavigatorPanel()
        self._navigator.anchoring_changed.connect(self._on_navigator_changed)
        body.addWidget(self._navigator)

        canvas_col = QVBoxLayout()
        canvas_col.setContentsMargins(0, 0, 0, 0)
        canvas_col.setSpacing(0)

        self._canvas = ImageCanvas()
        self._canvas.overlay_panned.connect(self._on_overlay_panned)
        self._canvas.mouse_position_changed.connect(self._on_canvas_mouse_moved)
        self._canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self._canvas.canvas_drag_started.connect(self._on_canvas_drag_started)
        self._canvas.canvas_dragged.connect(self._on_canvas_dragged)
        self._canvas.canvas_drag_ended.connect(self._on_canvas_drag_ended)
        canvas_col.addWidget(self._canvas, stretch=1)

        # Region label bar
        self._region_bar = QLabel("")
        self._region_bar.setFixedHeight(38)
        self._region_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._region_bar.setStyleSheet(
            "background: #1a1a1a; color: #fff; font-size: 12px; font-weight: bold;"
            " border-top: 1px solid #333;"
        )
        canvas_col.addWidget(self._region_bar)

        body.addLayout(canvas_col, stretch=1)
        root.addLayout(body, stretch=1)

        # Delete CP shortcuts — QShortcut works even when pyqtgraph has focus
        from PyQt6.QtGui import QKeySequence, QShortcut
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._delete_hovered_cp)

    def _make_mode_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background: #252525;")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 2, 8, 2)
        h.setSpacing(4)

        # Align / Warp mode toggle
        self._mode_group = QButtonGroup()
        for label, mode in [("Align", "align"), ("Warp", "warp")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setStyleSheet(
                "QPushButton { border-radius: 4px; padding: 2px 12px; color: #ccc;"
                " background: #333; }"
                "QPushButton:checked { background: #1e5a8a; color: #fff; }"
                "QPushButton:hover { background: #444; }"
            )
            btn.setProperty("mode", mode)
            btn.clicked.connect(lambda checked, m=mode: self._set_mode(m))
            self._mode_group.addButton(btn)
            h.addWidget(btn)

        if self._mode_group.buttons():
            self._mode_group.buttons()[0].setChecked(True)

        h.addSpacing(8)

        # Outline toggle
        self._outline_btn = QPushButton("Outline")
        self._outline_btn.setCheckable(True)
        self._outline_btn.setFixedHeight(28)
        self._outline_btn.setToolTip("Show white region outlines instead of coloured fill")
        self._outline_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #333; border: 1px solid #555; }"
            "QPushButton:checked { background: #4a4a1a; color: #ff0; border-color: #888; }"
            "QPushButton:hover { background: #444; }"
        )
        self._outline_btn.toggled.connect(self._on_outline_toggled)
        h.addWidget(self._outline_btn)

        h.addSpacing(8)

        # Scale buttons
        _scale_css = (
            "QPushButton { border-radius: 3px; padding: 2px 7px; color: #ccc;"
            " background: #383838; border: 1px solid #555; font-size: 11px; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        scale_specs = [
            ("\u2194+", "Wider (2%)",    1.0 / _SCALE_STEP,  1.0),
            ("\u2194\u2212", "Narrower (2%)", _SCALE_STEP,   1.0),
            ("\u2195+", "Taller (2%)",   1.0,        1.0 / _SCALE_STEP),
            ("\u2195\u2212", "Shorter (2%)", 1.0,    _SCALE_STEP),
        ]
        self._scale_btns: list[QPushButton] = []
        for sym, tip, su, sv in scale_specs:
            btn = QPushButton(sym)
            btn.setFixedHeight(28)
            btn.setFixedWidth(32)
            btn.setToolTip(tip)
            btn.setStyleSheet(_scale_css)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=su, t=sv: self._scale_overlay(s, t))
            h.addWidget(btn)
            self._scale_btns.append(btn)

        h.addSpacing(8)

        # Translation buttons: ±AP (axis 1), ±DV (axis 2), ±LR (axis 0)
        move_specs = [
            ("AP\u2212", "Move posterior (AP\u2212)", 1, -_MOVE_STEP),
            ("AP+",      "Move anterior (AP+)",      1, +_MOVE_STEP),
            ("DV\u2212", "Move dorsal (DV\u2212)",   2, -_MOVE_STEP),
            ("DV+",      "Move ventral (DV+)",       2, +_MOVE_STEP),
            ("LR\u2212", "Move left (LR\u2212)",     0, -_MOVE_STEP),
            ("LR+",      "Move right (LR+)",         0, +_MOVE_STEP),
        ]
        self._move_btns: list[QPushButton] = []
        for i, (sym, tip, axis, step) in enumerate(move_specs):
            if i % 2 == 0 and i > 0:
                h.addSpacing(4)  # gap between axis pairs
            btn = QPushButton(sym)
            btn.setFixedHeight(28)
            btn.setFixedWidth(46)
            btn.setToolTip(tip)
            btn.setStyleSheet(_scale_css)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, a=axis, s=step: self._move_plane(a, s))
            h.addWidget(btn)
            self._move_btns.append(btn)

        h.addSpacing(8)

        # Rotation buttons for the three atlas-plane axes.
        rotate_specs = [
            ("⟲R", "Roll clockwise (1°)", "roll", -_ROTATE_STEP_DEG),
            ("R⟳", "Roll counter-clockwise (1°)", "roll", +_ROTATE_STEP_DEG),
            ("⟲DV", "Tilt DV negative (1°)", "tilt_dv", -_ROTATE_STEP_DEG),
            ("DV⟳", "Tilt DV positive (1°)", "tilt_dv", +_ROTATE_STEP_DEG),
            ("⟲AP", "Tilt AP negative (1°)", "tilt_ap", -_ROTATE_STEP_DEG),
            ("AP⟳", "Tilt AP positive (1°)", "tilt_ap", +_ROTATE_STEP_DEG),
        ]
        self._rotate_btns: list[QPushButton] = []
        for i, (sym, tip, axis, step) in enumerate(rotate_specs):
            if i % 2 == 0 and i > 0:
                h.addSpacing(4)
            btn = QPushButton(sym)
            btn.setFixedHeight(28)
            btn.setFixedWidth(48)
            btn.setToolTip(tip)
            btn.setStyleSheet(_scale_css)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, a=axis, s=step: self._rotate_plane(a, s))
            h.addWidget(btn)
            self._rotate_btns.append(btn)

        h.addStretch()

        # Series / Store / Clear
        self._deepslice_btn = QPushButton("Run DeepSlice")
        self._deepslice_btn.setFixedHeight(28)
        self._deepslice_btn.setToolTip("Generate editable affine suggestions with DeepSlice")
        self._deepslice_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #383838; border: 1px solid #555; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        self._deepslice_btn.setEnabled(False)
        self._deepslice_btn.clicked.connect(self.deepslice_requested)
        h.addWidget(self._deepslice_btn)

        self._default_btn = QPushButton("Default proposal")
        self._default_btn.setFixedHeight(28)
        self._default_btn.setToolTip("Revert editable suggestions to VERSO's default AP proposal")
        self._default_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #383838; border: 1px solid #555; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        self._default_btn.setEnabled(False)
        self._default_btn.clicked.connect(self.default_proposal_requested)
        h.addWidget(self._default_btn)

        self._reverse_btn = QPushButton("Reverse proposal")
        self._reverse_btn.setFixedHeight(28)
        self._reverse_btn.setToolTip(
            "Reverse the initial AP proposal before storing any alignment"
        )
        self._reverse_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #383838; border: 1px solid #555; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        self._reverse_btn.setEnabled(False)
        self._reverse_btn.clicked.connect(self.reverse_requested)
        h.addWidget(self._reverse_btn)

        self._store_btn = QPushButton("Store")
        self._store_btn.setFixedHeight(28)
        self._store_btn.setToolTip("Lock current atlas plane to this section")
        self._store_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #2a5a2a; }"
            "QPushButton:hover { background: #3a6a3a; }"
            "QPushButton:disabled { color: #666; background: #333; }"
        )
        self._store_btn.setEnabled(False)
        self._store_btn.clicked.connect(self._store_anchoring)
        h.addWidget(self._store_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setToolTip("Remove stored plane and revert to interpolated")
        self._clear_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #5a2a2a; }"
            "QPushButton:hover { background: #6a3a3a; }"
            "QPushButton:disabled { color: #666; background: #333; }"
        )
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_anchoring)
        h.addWidget(self._clear_btn)

        self._clear_cps_btn = QPushButton("Clear CPs")
        self._clear_cps_btn.setFixedHeight(28)
        self._clear_cps_btn.setToolTip("Remove all warp control points from this section")
        self._clear_cps_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #5a2a2a; }"
            "QPushButton:hover { background: #6a3a3a; }"
            "QPushButton:disabled { color: #666; background: #333; }"
        )
        self._clear_cps_btn.setEnabled(False)
        self._clear_cps_btn.setVisible(False)
        self._clear_cps_btn.clicked.connect(self._clear_all_cps)
        h.addWidget(self._clear_cps_btn)

        self._status_label = QLabel("No section loaded")
        self._status_label.setStyleSheet("color: #888; font-size: 11px; padding-left: 8px;")
        h.addWidget(self._status_label)

        return bar

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._cp_hovered = -1
        self._cp_dragging = -1
        is_align = (mode == "align")
        self._reverse_btn.setVisible(is_align)
        self._deepslice_btn.setVisible(is_align)
        self._default_btn.setVisible(is_align)
        self._store_btn.setVisible(is_align)
        self._clear_btn.setVisible(is_align)
        self._clear_cps_btn.setVisible(not is_align)
        self._update_clear_cps_enabled()
        self._update_overlay()
        self.mode_changed.emit(mode)

    def _on_outline_toggled(self, checked: bool) -> None:
        self._outline_mode = checked
        self._update_overlay()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def set_atlas(self, atlas: AtlasVolume | None) -> None:
        self._atlas = atlas
        self._navigator.set_atlas(atlas)
        self._update_overlay()

    def set_reverse_enabled(self, enabled: bool) -> None:
        """Enable the series reverse command when no alignment is stored."""
        self._reverse_btn.setEnabled(enabled)

    def set_deepslice_enabled(self, enabled: bool, running: bool = False) -> None:
        """Enable DeepSlice proposal controls."""
        self._deepslice_btn.setEnabled(enabled and not running)
        self._deepslice_btn.setText("DeepSlice running..." if running else "Run DeepSlice")
        self._default_btn.setEnabled(enabled and not running)

    def load_section(self, section: Section | None) -> None:
        self._section = section
        self._raw_image = None
        self._canvas.clear()
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._store_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        self._region_bar.setText("")
        for btn in self._scale_btns + self._move_btns + self._rotate_btns:
            btn.setEnabled(False)
        if section is None:
            self._status_label.setText("No section loaded")
            self._navigator.set_anchoring(None)
            return

        import os
        self._status_label.setText(os.path.basename(section.original_path))

        from PyQt6.QtWidgets import QMessageBox

        from verso.engine.io.image_io import ensure_working_copy
        try:
            self._raw_image = ensure_working_copy(section)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        self._display_image()
        self._update_overlay()
        self._store_btn.setEnabled(True)
        has_anchoring = bool(section.alignment.anchoring) and any(
            v != 0.0 for v in section.alignment.anchoring
        )
        self._clear_btn.setEnabled(has_anchoring)
        for btn in self._scale_btns + self._move_btns + self._rotate_btns:
            btn.setEnabled(True)

    def _display_image(self) -> None:
        if self._raw_image is None:
            return
        img = self._raw_image
        if self._section and self._section.preprocessing.flip_horizontal:
            img = np.fliplr(img)
        self._canvas.set_background(np.ascontiguousarray(img))

    def refresh_display(self) -> None:
        self._display_image()
        self._update_overlay()

    def update_overlay(self) -> None:
        self._update_overlay()

    def _update_overlay(self) -> None:
        if self._atlas is None or self._section is None or self._raw_image is None:
            self._canvas.set_overlay(None)
            self._canvas.clear_control_points()
            return

        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            h, w = self._raw_image.shape[:2]
            anchoring = self._atlas.default_anchoring(aspect_ratio=w / h)
            self._section.alignment.anchoring = anchoring

        self._navigator.set_anchoring(anchoring)

        h_bg, w_bg = self._raw_image.shape[:2]
        # Sample atlas at a capped resolution for speed; canvas stretches it to
        # fill the background exactly via setRect (no visual quality loss).
        ATLAS_MAX_SIDE = 512
        scale = min(1.0, ATLAS_MAX_SIDE / max(w_bg, h_bg))
        out_w = max(1, round(w_bg * scale))
        out_h = max(1, round(h_bg * scale))

        try:
            if self._outline_mode:
                rgba = self._atlas.slice_outline(anchoring, out_w, out_h)
            else:
                rgba = self._atlas.slice_annotation(anchoring, out_w, out_h)
        except Exception:
            self._canvas.set_overlay(None)
            return

        # Apply nonlinear warp when in warp mode and control points exist
        cps = self._section.warp.control_points
        if self._mode == "warp" and cps:
            from verso.engine.warping import warp_overlay
            src = np.array([[cp.src_x, cp.src_y] for cp in cps])
            dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
            try:
                rgba = warp_overlay(rgba, src, dst)
            except Exception:
                pass

        self._canvas.set_overlay(rgba, display_w=w_bg, display_h=h_bg)

        # Draw control points in warp mode
        if self._mode == "warp":
            dst_pts = [(cp.dst_x, cp.dst_y) for cp in cps]
            src_pts = [(cp.src_x, cp.src_y) for cp in cps]
            self._canvas.set_control_points(
                dst_pts, w_bg, h_bg, self._cp_hovered,
                cp_size=self._cp_size,
                cp_shape=self._cp_shape,
                cp_color=self._cp_color,
                src_pts=src_pts,
            )
        else:
            self._canvas.clear_control_points()

        self._update_clear_cps_enabled()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_navigator_changed(self, new_anchoring: list[float]) -> None:
        if self._section is None:
            return
        self._section.alignment.anchoring = new_anchoring
        if self._atlas is not None:
            self._section.alignment.ap_position_mm = self._atlas.ap_voxel_to_mm(new_anchoring[1])
        self._update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    def _on_overlay_panned(self, dx: float, dy: float) -> None:
        if self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        h_bg, w_bg = self._raw_image.shape[:2]
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        new_o = o - (dx / w_bg) * u - (dy / h_bg) * v
        new_anchoring = new_o.tolist() + anchoring[3:]
        self._section.alignment.anchoring = new_anchoring
        if self._atlas is not None:
            self._section.alignment.ap_position_mm = self._atlas.ap_voxel_to_mm(new_anchoring[1])
        self._update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    def _on_canvas_mouse_moved(self, x: float, y: float) -> None:
        """Update region label bar and CP hover state from cursor position."""
        # Update hovered CP even if atlas isn't loaded
        if self._mode == "warp" and self._cp_dragging < 0:
            new_hov = self._pick_cp(x, y)
            if new_hov != self._cp_hovered:
                self._cp_hovered = new_hov
                self._update_overlay()

        if self._atlas is None or self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        h_bg, w_bg = self._raw_image.shape[:2]
        if x < 0 or y < 0 or x >= w_bg or y >= h_bg:
            self._region_bar.setText("")
            self._region_bar.setStyleSheet(
                "background: #1a1a1a; color: #fff; font-size: 12px; font-weight: bold;"
                " border-top: 1px solid #333;"
            )
            return
        s, t = x / w_bg, y / h_bg
        name, (r, g, b) = self._atlas.get_region_info(anchoring, s, t)
        # Darken the region colour slightly so white text stays legible
        br = int(r * 0.55)
        bg = int(g * 0.55)
        bb = int(b * 0.55)
        self._region_bar.setText(name)
        self._region_bar.setStyleSheet(
            f"background: rgb({br},{bg},{bb}); color: #fff; font-size: 20px;"
            " font-weight: bold; border-top: 1px solid #333;"
        )

    def _scale_overlay(self, scale_u: float, scale_v: float) -> None:
        if self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        from verso.engine.registration import scale_anchoring
        new_anchoring = scale_anchoring(anchoring, scale_u, scale_v)
        self._section.alignment.anchoring = new_anchoring
        if self._atlas is not None:
            self._section.alignment.ap_position_mm = self._atlas.ap_voxel_to_mm(new_anchoring[1])
        self._update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    def _move_plane(self, axis: int, delta: float) -> None:
        """Translate the cut-plane origin along one atlas axis by delta voxels.

        Axis convention (atlas voxel space):
            0 = LR (left→right), 1 = AP (posterior→anterior), 2 = DV (dorsal→ventral)
        """
        if self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        new_anchoring = list(anchoring)
        new_anchoring[axis] += delta
        self._section.alignment.anchoring = new_anchoring
        if self._atlas is not None:
            self._section.alignment.ap_position_mm = self._atlas.ap_voxel_to_mm(new_anchoring[1])
        self._update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    def _rotate_plane(self, axis: str, degrees: float) -> None:
        """Rotate the atlas plane around one of the toolbar rotation axes."""
        if self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return

        import math

        import numpy as np

        from verso.engine.registration import anchoring_to_vectors, vectors_to_anchoring

        def rot_around(vec, rot_axis, deg):
            angle = math.radians(deg)
            c, s = math.cos(angle), math.sin(angle)
            k = rot_axis / np.linalg.norm(rot_axis)
            return c * vec + s * np.cross(k, vec) + (1.0 - c) * np.dot(k, vec) * k

        o, u, v = anchoring_to_vectors(anchoring)
        center = o + u / 2.0 + v / 2.0
        u_n = u / np.linalg.norm(u)
        v_n = v / np.linalg.norm(v)
        n = np.cross(u_n, v_n)

        if axis == "roll":
            u_new = rot_around(u, n, degrees)
            v_new = rot_around(v, n, degrees)
        elif axis == "tilt_dv":
            u_new = u
            v_new = rot_around(v, u_n, degrees)
        elif axis == "tilt_ap":
            u_new = rot_around(u, v_n, degrees)
            v_new = v
        else:
            return

        new_o = center - u_new / 2.0 - v_new / 2.0
        new_anchoring = vectors_to_anchoring(new_o, u_new, v_new)
        self._section.alignment.anchoring = new_anchoring
        if self._atlas is not None:
            self._section.alignment.ap_position_mm = self._atlas.ap_voxel_to_mm(new_anchoring[1])
        self._update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Store / Clear
    # ------------------------------------------------------------------

    def _store_anchoring(self) -> None:
        if self._section is None or self._atlas is None:
            return
        if not self._section.alignment.anchoring or all(
            v == 0.0 for v in self._section.alignment.anchoring
        ):
            h, w = self._raw_image.shape[:2]
            self._section.alignment.anchoring = self._atlas.default_anchoring(w / h)
        self._section.alignment.status = AlignmentStatus.COMPLETE
        self._clear_btn.setEnabled(True)
        self.section_modified.emit()
        self.alignments_updated.emit()

    def _clear_anchoring(self) -> None:
        if self._section is None:
            return
        self._section.alignment.anchoring = [0.0] * 9
        self._section.alignment.ap_position_mm = None
        self._section.alignment.status = AlignmentStatus.NOT_STARTED
        self._section.alignment.source = None
        self._section.alignment.proposal_anchoring = None
        self._section.alignment.proposal_confidence = None
        self._section.alignment.proposal_run_id = None
        self.alignments_updated.emit()
        self._clear_btn.setEnabled(True)

    def _clear_all_cps(self) -> None:
        if self._section is None:
            return
        self._section.warp.control_points.clear()
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._update_clear_cps_enabled()
        self._update_overlay()
        self.section_modified.emit()

    def _update_clear_cps_enabled(self) -> None:
        has_cps = (
            self._section is not None
            and bool(self._section.warp.control_points)
        )
        self._clear_cps_btn.setEnabled(has_cps)

    def set_cp_style(self, size: int, shape: str, color: str) -> None:
        """Update control-point visual style and redraw."""
        self._cp_size = size
        self._cp_shape = shape
        self._cp_color = color
        if self._mode == "warp":
            self._update_overlay()

    def apply_warp(self) -> None:
        if self._section is None or not self._section.warp.control_points:
            return
        self.section_modified.emit()

    # ------------------------------------------------------------------
    # Warp control-point helpers
    # ------------------------------------------------------------------

    def _norm_pos(self, x: float, y: float) -> tuple[float, float] | None:
        """Convert image pixel coords to normalised [0,1].  Returns None if OOB."""
        if self._raw_image is None:
            return None
        h, w = self._raw_image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x / w, y / h

    def _clamped_norm_pos(self, x: float, y: float) -> tuple[float, float] | None:
        """Convert image pixel coords to normalised [0,1], clamping at image edges."""
        if self._raw_image is None:
            return None
        h, w = self._raw_image.shape[:2]
        x = min(max(x, 0.0), float(w))
        y = min(max(y, 0.0), float(h))
        return x / w, y / h

    def _move_dragged_cp_to(self, x: float, y: float) -> None:
        """Move the dragged CP using VisuAlign-style press-to-current displacement."""
        if self._section is None or self._cp_dragging < 0:
            return
        cur = self._clamped_norm_pos(x, y)
        if cur is None:
            return
        cp = self._section.warp.control_points[self._cp_dragging]
        if self._cp_drag_start_norm is None or self._cp_drag_start_dst is None:
            cp.dst_x, cp.dst_y = cur
            return

        sx, sy = self._cp_drag_start_norm
        bx, by = self._cp_drag_start_dst
        cp.dst_x = min(max(bx + cur[0] - sx, 0.0), 1.0)
        cp.dst_y = min(max(by + cur[1] - sy, 0.0), 1.0)

    def _pick_cp(self, x: float, y: float) -> int:
        """Return index of nearest CP within pick radius, or -1."""
        if self._section is None or self._raw_image is None:
            return -1
        cps = self._section.warp.control_points
        if not cps:
            return -1
        h, w = self._raw_image.shape[:2]
        px, py = x / w, y / h
        best, best_d2 = -1, (self._CP_PICK_RADIUS / w) ** 2 + (self._CP_PICK_RADIUS / h) ** 2
        for i, cp in enumerate(cps):
            d2 = (cp.dst_x - px) ** 2 + (cp.dst_y - py) ** 2
            if d2 < best_d2:
                best_d2, best = d2, i
        return best

    def _current_src_at(self, s: float, t: float) -> tuple[float, float]:
        """Atlas normalised coords for section position (s, t) given current warp."""
        cps = self._section.warp.control_points if self._section else []
        if not cps:
            return s, t
        from verso.engine.warping import find_atlas_position
        src = np.array([[cp.src_x, cp.src_y] for cp in cps])
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
        return find_atlas_position(s, t, src, dst)

    def _update_control_point_display(self) -> None:
        """Redraw CP markers without resampling the atlas overlay."""
        if self._mode != "warp" or self._section is None or self._raw_image is None:
            self._canvas.clear_control_points()
            return
        h_bg, w_bg = self._raw_image.shape[:2]
        cps = self._section.warp.control_points
        self._canvas.set_control_points(
            [(cp.dst_x, cp.dst_y) for cp in cps],
            w_bg,
            h_bg,
            self._cp_hovered,
            cp_size=self._cp_size,
            cp_shape=self._cp_shape,
            cp_color=self._cp_color,
            src_pts=[(cp.src_x, cp.src_y) for cp in cps],
        )

    # ------------------------------------------------------------------
    # Warp canvas event handlers
    # ------------------------------------------------------------------

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        """Single click in warp mode: add a control point (not near existing one)."""
        if self._mode != "warp" or self._section is None:
            return
        norm = self._norm_pos(x, y)
        if norm is None:
            return
        s, t = norm
        if self._pick_cp(x, y) >= 0:
            return  # clicked on existing CP — drag handles the move
        u, v = self._current_src_at(s, t)
        from verso.engine.model.alignment import ControlPoint
        self._section.warp.control_points.append(ControlPoint(u, v, s, t))
        self._cp_hovered = len(self._section.warp.control_points) - 1
        self._update_overlay()
        self.section_modified.emit()

    def _on_canvas_drag_started(self, x: float, y: float) -> None:
        """Drag start in warp mode: pick an existing CP to move."""
        if self._mode != "warp":
            return
        self._cp_dragging = self._pick_cp(x, y)
        self._cp_drag_start_norm = self._clamped_norm_pos(x, y)
        self._cp_drag_start_dst = None
        if self._section is not None and self._cp_dragging >= 0:
            cp = self._section.warp.control_points[self._cp_dragging]
            self._cp_drag_start_dst = (cp.dst_x, cp.dst_y)

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        """Drag update: move the dragged CP's dst position."""
        if self._mode != "warp" or self._cp_dragging < 0 or self._section is None:
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        self._update_control_point_display()

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        """Drag end: finalise CP position."""
        if self._mode != "warp" or self._cp_dragging < 0 or self._section is None:
            self._cp_dragging = -1
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._update_overlay()
        self.section_modified.emit()

    def _delete_hovered_cp(self) -> None:
        """Delete the hovered control point (triggered by Delete/Backspace shortcut)."""
        if self._mode != "warp" or self._section is None:
            return
        cps = self._section.warp.control_points
        if 0 <= self._cp_hovered < len(cps):
            cps.pop(self._cp_hovered)
            self._cp_hovered = -1
            self._update_overlay()
            self.section_modified.emit()
