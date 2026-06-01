"""Warp view — nonlinear refinement on top of an existing affine alignment.

Composes the shared :class:`SectionCanvasPanel` (created once by
``MainWindow`` and reparented into whichever view is active) and contributes
only the warp-specific toolbar and control-point interaction.

Edits made to control points are drafts: they live in memory only and are
discarded on slice / view change.  The shared Save / Clear bar in the
Warp properties page commits or wipes them.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus, WarpState
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.gui.state import AppState


class WarpView(QWidget):
    """Canvas view for nonlinear warp via per-section control points."""

    dirty_changed = pyqtSignal(bool)
    # Emitted whenever the control-point set changes (add / delete) so the
    # filmstrip status dot can refresh even when the dirty flag doesn't flip.
    cp_changed = pyqtSignal()

    # Pixel distance threshold for picking an existing control point.
    _CP_PICK_RADIUS = 16  # px

    def __init__(
        self,
        panel: SectionCanvasPanel,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._state = state

        self._active = False

        # CP interaction state
        self._cp_hovered: int = -1
        self._cp_dragging: int = -1
        self._cp_drag_start_norm: tuple[float, float] | None = None
        self._cp_drag_start_dst: tuple[float, float] | None = None
        self._cp_size = 10
        self._cp_shape = "Cross"
        self._cp_color = "Yellow"

        # Real-time warp throttle: fires update_overlay at ~30fps during CP drag
        self._warp_timer = QTimer(self)
        self._warp_timer.setInterval(33)
        self._warp_timer.timeout.connect(self._panel.update_overlay)

        # Snapshot of section.warp at section-load time, used to report whether
        # there's persisted state to Clear.  CP edits are no longer discarded on
        # navigation — they persist on the Section and dirtiness lives in the
        # edit registry.
        self._baseline_warp: WarpState | None = None
        self._dirty = False

        self._build_ui()
        self._wire_panel()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(make_view_status_bar(self._panel.make_status_label()))

        self._panel_slot = QWidget()
        slot_layout = QHBoxLayout(self._panel_slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        slot_layout.setSpacing(0)
        root.addWidget(self._panel_slot, stretch=1)

        # Delete CP shortcuts — work even when pyqtgraph has focus
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._delete_hovered_cp)

    def _wire_panel(self) -> None:
        self._panel.canvas_clicked.connect(self._on_canvas_clicked)
        self._panel.canvas_drag_started.connect(self._on_canvas_drag_started)
        self._panel.canvas_dragged.connect(self._on_canvas_dragged)
        self._panel.canvas_drag_ended.connect(self._on_canvas_drag_ended)
        self._panel.mouse_position_changed.connect(self._on_canvas_mouse_moved)
        self._panel.overlay_updated.connect(self._on_overlay_updated)
        self._panel.section_loaded.connect(self._on_section_loaded)

    # ------------------------------------------------------------------
    # Activation — claim the shared panel and install warp hooks
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Reparent the shared panel into this view and install warp hooks."""
        self._active = True
        self._panel_slot.layout().addWidget(self._panel)
        self._panel.canvas.set_interaction_mode("warp")
        self._panel.overlay_post_processor = self._warp_overlay
        self._panel.cursor_to_atlas_mapper = self._cursor_to_src
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._panel.update_overlay()
        # Re-snapshot in case the section was loaded before activate.
        section = self._panel.section
        if section is not None:
            self._baseline_warp = copy.deepcopy(section.warp)
            self._set_dirty(self._state.is_dirty(section.id, "warp"))

    def deactivate(self) -> None:
        """Release warp hooks so other views see a clean panel."""
        self._active = False
        self._panel.overlay_post_processor = None
        self._panel.cursor_to_atlas_mapper = None
        self._warp_timer.stop()
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._panel.canvas.clear_control_points()

    # ------------------------------------------------------------------
    # External API — preserved from the old AlignView
    # ------------------------------------------------------------------

    def set_cp_style(self, size: int, shape: str, color: str) -> None:
        self._cp_size = size
        self._cp_shape = shape
        self._cp_color = color
        if self._active:
            self._draw_control_points()

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _warp_overlay(self, rgba: np.ndarray) -> np.ndarray:
        section = self._panel.section
        if section is None:
            return rgba
        cps = section.warp.control_points
        if not cps:
            return rgba
        from verso.engine.warping import warp_overlay
        src = np.array([[cp.src_x, cp.src_y] for cp in cps])
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
        try:
            return warp_overlay(rgba, src, dst)
        except Exception:
            return rgba

    def _cursor_to_src(self, s: float, t: float) -> tuple[float, float]:
        section = self._panel.section
        cps = section.warp.control_points if section else []
        if not cps:
            return s, t
        from verso.engine.warping import find_atlas_position
        src = np.array([[cp.src_x, cp.src_y] for cp in cps])
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
        return find_atlas_position(s, t, src, dst)

    # ------------------------------------------------------------------
    # Panel events
    # ------------------------------------------------------------------

    def _on_section_loaded(self, section) -> None:
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        if section is None:
            self._baseline_warp = None
            self._set_dirty(False)
            return
        # Persisted CP edits survive navigation; mirror the registry dirty state.
        # When the slice is still dirty, recover the genuine last-saved baseline
        # from the stash rather than re-snapshotting the (dirty) section.
        if self._state.is_dirty(section.id, "warp"):
            stashed = self._state.get_baseline(section.id, "warp")
            self._baseline_warp = (
                stashed if stashed is not None else copy.deepcopy(section.warp)
            )
            self._set_dirty(True)
        else:
            self._baseline_warp = copy.deepcopy(section.warp)
            self._state.pop_baseline(section.id, "warp")
            self._set_dirty(False)

    def _on_overlay_updated(self, _anchoring, _display_w, _display_h) -> None:
        if not self._active:
            return
        self._draw_control_points()

    def _on_canvas_mouse_moved(self, x: float, y: float) -> None:
        if not self._active:
            return
        if self._cp_dragging >= 0:
            return
        new_hov = self._pick_cp(x, y)
        if new_hov != self._cp_hovered:
            self._cp_hovered = new_hov
            self._draw_control_points()

    # ------------------------------------------------------------------
    # CP drawing
    # ------------------------------------------------------------------

    def _draw_control_points(self) -> None:
        section = self._panel.section
        raw = self._panel.raw_image
        if section is None or raw is None:
            self._panel.canvas.clear_control_points()
            return
        h_bg, w_bg = raw.shape[:2]
        cps = section.warp.control_points
        self._panel.canvas.set_control_points(
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
    # CP picking / movement helpers
    # ------------------------------------------------------------------

    def _norm_pos(self, x: float, y: float) -> tuple[float, float] | None:
        raw = self._panel.raw_image
        if raw is None:
            return None
        h, w = raw.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x / w, y / h

    def _clamped_norm_pos(self, x: float, y: float) -> tuple[float, float] | None:
        raw = self._panel.raw_image
        if raw is None:
            return None
        h, w = raw.shape[:2]
        x = min(max(x, 0.0), float(w))
        y = min(max(y, 0.0), float(h))
        return x / w, y / h

    def _pick_cp(self, x: float, y: float) -> int:
        section = self._panel.section
        raw = self._panel.raw_image
        if section is None or raw is None:
            return -1
        cps = section.warp.control_points
        if not cps:
            return -1
        h, w = raw.shape[:2]
        px, py = x / w, y / h
        best, best_d2 = -1, (self._CP_PICK_RADIUS / w) ** 2 + (self._CP_PICK_RADIUS / h) ** 2
        for i, cp in enumerate(cps):
            d2 = (cp.dst_x - px) ** 2 + (cp.dst_y - py) ** 2
            if d2 < best_d2:
                best_d2, best = d2, i
        return best

    def _move_dragged_cp_to(self, x: float, y: float) -> None:
        section = self._panel.section
        if section is None or self._cp_dragging < 0:
            return
        if self._cp_dragging >= len(section.warp.control_points):
            self._cp_dragging = -1
            return
        cur = self._clamped_norm_pos(x, y)
        if cur is None:
            return
        cp = section.warp.control_points[self._cp_dragging]
        if self._cp_drag_start_norm is None or self._cp_drag_start_dst is None:
            cp.dst_x, cp.dst_y = cur
            return
        sx, sy = self._cp_drag_start_norm
        bx, by = self._cp_drag_start_dst
        cp.dst_x = min(max(bx + cur[0] - sx, 0.0), 1.0)
        cp.dst_y = min(max(by + cur[1] - sy, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Canvas click / drag handlers
    # ------------------------------------------------------------------

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        section = self._panel.section
        if section is None:
            return
        norm = self._norm_pos(x, y)
        if norm is None:
            return
        s, t = norm
        if self._pick_cp(x, y) >= 0:
            return
        u, v = self._cursor_to_src(s, t)
        from verso.engine.model.alignment import ControlPoint
        section.warp.control_points.append(ControlPoint(u, v, s, t))
        self._cp_hovered = len(section.warp.control_points) - 1
        self._panel.update_overlay()
        self._set_dirty(True)
        self.cp_changed.emit()

    def _on_canvas_drag_started(self, x: float, y: float) -> None:
        self._cp_dragging = self._pick_cp(x, y)
        self._cp_drag_start_norm = self._clamped_norm_pos(x, y)
        self._cp_drag_start_dst = None
        section = self._panel.section
        if section is not None and self._cp_dragging >= 0:
            cp = section.warp.control_points[self._cp_dragging]
            self._cp_drag_start_dst = (cp.dst_x, cp.dst_y)

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        if self._cp_dragging < 0 or self._panel.section is None:
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        if not self._warp_timer.isActive():
            self._warp_timer.start()

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        self._warp_timer.stop()
        if self._cp_dragging < 0 or self._panel.section is None:
            self._cp_dragging = -1
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._panel.update_overlay()
        self._set_dirty(True)

    def _delete_hovered_cp(self) -> None:
        section = self._panel.section
        if section is None:
            return
        cps = section.warp.control_points
        if 0 <= self._cp_hovered < len(cps):
            if self._cp_hovered == self._cp_dragging:
                self._cp_dragging = -1
            cps.pop(self._cp_hovered)
            self._cp_hovered = -1
            self._panel.update_overlay()
            self._set_dirty(True)
            self.cp_changed.emit()

    # ------------------------------------------------------------------
    # Draft / save / clear / discard
    # ------------------------------------------------------------------

    def is_dirty(self) -> bool:
        return self._dirty

    def has_persisted_state(self) -> bool:
        baseline = self._baseline_warp
        if baseline is None:
            return False
        return bool(baseline.control_points)

    def save(self) -> bool:
        """Commit the current control-point list as this slice's warp."""
        section = self._panel.section
        if section is None:
            return False
        self._baseline_warp = copy.deepcopy(section.warp)
        self._state.pop_baseline(section.id, "warp")
        self._set_dirty(False)
        return True

    def revert(self) -> bool:
        """Discard unsaved control-point edits, restoring the last-saved warp."""
        section = self._panel.section
        if section is None or self._baseline_warp is None:
            return False
        section.warp = copy.deepcopy(self._baseline_warp)
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._state.pop_baseline(section.id, "warp")
        self._set_dirty(False)
        if self._active:
            self._panel.update_overlay()
        return True

    def clear(self) -> bool:
        """Wipe this slice's warp control points."""
        section = self._panel.section
        if section is None:
            return False
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_norm = None
        self._cp_drag_start_dst = None
        self._baseline_warp = copy.deepcopy(section.warp)
        self._state.pop_baseline(section.id, "warp")
        self._set_dirty(False)
        if self._active:
            self._panel.update_overlay()
        return True

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        if dirty:
            section = self._panel.section
            if section is not None and self._baseline_warp is not None:
                self._state.set_baseline(
                    section.id, "warp", copy.deepcopy(self._baseline_warp)
                )
        self._dirty = dirty
        self.dirty_changed.emit(dirty)
