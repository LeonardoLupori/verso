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


class AlignView(QWidget):
    """Canvas view for atlas alignment (Align mode) and warp (Warp mode)."""

    section_modified = pyqtSignal()
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "align" or "warp"

    # Pixel distance threshold (in normalised units × display size) for
    # picking an existing control point
    _CP_PICK_RADIUS = 12  # px

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section: Section | None = None
        self._raw_image = None
        self._atlas: "AtlasVolume | None" = None
        self._mode = "align"
        self._outline_mode = False
        # Warp interaction state
        self._cp_hovered: int = -1    # index of CP under cursor (-1 = none)
        self._cp_dragging: int = -1   # index of CP currently being dragged
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
        self._region_bar.setFixedHeight(28)
        self._region_bar.setStyleSheet(
            "background: #1a1a1a; color: #fff; font-size: 12px; font-weight: bold;"
            " padding-left: 8px; border-top: 1px solid #333;"
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
                "QPushButton { border-radius: 4px; padding: 2px 12px; color: #ccc; background: #333; }"
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

        h.addStretch()

        # Store / Clear
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

        self._status_label = QLabel("No section loaded")
        self._status_label.setStyleSheet("color: #888; font-size: 11px; padding-left: 8px;")
        h.addWidget(self._status_label)

        return bar

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._cp_hovered = -1
        self._cp_dragging = -1
        is_align = (mode == "align")
        for btn in self._scale_btns:
            btn.setVisible(is_align)
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

    def set_atlas(self, atlas: "AtlasVolume | None") -> None:
        self._atlas = atlas
        self._navigator.set_atlas(atlas)
        self._update_overlay()

    def load_section(self, section: Section | None) -> None:
        self._section = section
        self._raw_image = None
        self._canvas.clear()
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._store_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        self._region_bar.setText("")
        for btn in self._scale_btns:
            btn.setEnabled(False)
        if section is None:
            self._status_label.setText("No section loaded")
            self._navigator.set_anchoring(None)
            return

        import os
        self._status_label.setText(os.path.basename(section.original_path))

        from verso.engine.io.image_io import ensure_working_copy
        from PyQt6.QtWidgets import QMessageBox
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
        for btn in self._scale_btns:
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
        scale = min(1.0, 512 / max(w_bg, h_bg))
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
            self._canvas.set_control_points(
                dst_pts, w_bg, h_bg, self._cp_hovered
            )
        else:
            self._canvas.clear_control_points()

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
                " padding-left: 8px; border-top: 1px solid #333;"
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
            f"background: rgb({br},{bg},{bb}); color: #fff; font-size: 12px;"
            " font-weight: bold; padding-left: 8px; border-top: 1px solid #333;"
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
        self._section.alignment.status = AlignmentStatus.NOT_STARTED
        self._clear_btn.setEnabled(False)
        self.alignments_updated.emit()

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

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        """Drag update: move the dragged CP's dst position."""
        if self._mode != "warp" or self._cp_dragging < 0 or self._section is None:
            return
        norm = self._norm_pos(x, y)
        if norm is None:
            return
        cp = self._section.warp.control_points[self._cp_dragging]
        cp.dst_x, cp.dst_y = norm
        self._cp_hovered = self._cp_dragging
        self._update_overlay()

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        """Drag end: finalise CP position."""
        if self._mode != "warp" or self._cp_dragging < 0 or self._section is None:
            self._cp_dragging = -1
            return
        norm = self._norm_pos(x, y)
        if norm is not None:
            cp = self._section.warp.control_points[self._cp_dragging]
            cp.dst_x, cp.dst_y = norm
        self._cp_hovered = self._cp_dragging
        self._cp_dragging = -1
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
