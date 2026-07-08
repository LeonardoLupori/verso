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

from verso.engine.drafts import commit_warp
from verso.engine.model.alignment import AlignmentStatus, ControlPoint, WarpState
from verso.gui.utils import require
from verso.gui.views.draft_canvas_view import DraftCanvasView
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.engine.model.project import Section
    from verso.gui.state import AppState


class WarpView(DraftCanvasView):
    """Canvas view for nonlinear warp via per-section control points."""

    STEP = "warp"

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
        super().__init__(state, parent)
        self._panel = panel

        self._active = False

        # CP interaction state
        self._cp_hovered: int = -1
        self._cp_dragging: int = -1
        self._cp_drag_start_px: tuple[float, float] | None = None
        self._cp_drag_start_dst: tuple[float, float] | None = None
        self._cp_size = 10
        self._cp_shape = "Cross"
        self._cp_color = "Yellow"

        # Real-time warp throttle: fires update_overlay at ~30fps during CP drag
        self._warp_timer = QTimer(self)
        self._warp_timer.setInterval(33)
        self._warp_timer.timeout.connect(self._panel.update_overlay)

        # Dirty flag, last-saved baseline, and the undo stack live in the base
        # (DraftCanvasView) keyed by (section.id, "warp").
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

        # Ctrl+Z undo — scoped to this view and its children (the reparented
        # canvas panel) so it only fires while the Warp view is active.
        undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo.activated.connect(self.undo)
        self._undo_shortcut = undo

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
        require(self._panel_slot.layout()).addWidget(self._panel)
        self._panel.canvas.set_interaction_mode("warp")
        self._panel.overlay_post_processor = self._warp_overlay
        self._panel.cursor_to_atlas_mapper = self._cursor_to_src
        self._reset_undo()  # also clears the CP interaction state + warp timer
        self._panel.update_overlay()
        # Re-sync the baseline in case the section was loaded before activate.
        # A no-op while dirty, so the stashed last-saved warp survives.  The
        # save bar's dirty state is refreshed by the window on view entry.
        section = self._panel.section
        if section is not None:
            self._state.sync_baseline(section.id, "warp", copy.deepcopy(section.warp))

    def deactivate(self) -> None:
        """Release warp hooks so other views see a clean panel."""
        self._active = False
        self._panel.overlay_post_processor = None
        self._panel.cursor_to_atlas_mapper = None
        self._warp_timer.stop()
        self._reset_cp_interaction()
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

    def apply_auto_control_points(self, cps: list[ControlPoint]) -> None:
        """Replace this slice's auto-generated control points with ``cps``.

        Manual control points are preserved; only ``auto=True`` points are
        discarded and replaced. Leaves the result as an unsaved draft.
        """
        section = self._panel.section
        if section is None:
            return
        self._push_undo()
        manual = [cp for cp in section.warp.control_points if not cp.auto]
        section.warp.control_points = manual + list(cps)
        self._cp_hovered = -1
        self._cp_dragging = -1
        if self._active:
            self._panel.update_overlay()
        self._set_dirty(True)
        self.cp_changed.emit()

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
        raw = self._panel.raw_image
        if raw is None:
            return rgba
        from verso.engine.warping import warp_overlay

        work_h, work_w = raw.shape[:2]
        src = np.array([[cp.src_x, cp.src_y] for cp in cps])
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
        try:
            return warp_overlay(rgba, src, dst, work_w, work_h)
        except Exception:
            return rgba

    def _cursor_to_src(self, s: float, t: float) -> tuple[float, float]:
        section = self._panel.section
        cps = section.warp.control_points if section else []
        if not cps:
            return s, t
        raw = self._panel.raw_image
        if raw is None:
            return s, t
        from verso.engine.warping import find_atlas_position

        work_h, work_w = raw.shape[:2]
        src = np.array([[cp.src_x, cp.src_y] for cp in cps])
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps])
        return find_atlas_position(s, t, src, dst, work_w, work_h)

    # ------------------------------------------------------------------
    # Panel events
    # ------------------------------------------------------------------

    def _on_section_loaded(self, section) -> None:
        # The canvas panel is shared across views and emits section_loaded to
        # every connected view.  Ignore it while another view owns the panel:
        # reacting here would clobber this view's dirty flag / baseline with the
        # currently-loaded section.  activate() re-syncs baseline + dirty state
        # from the registry on entry.
        if not self._active:
            return
        self._reset_undo()  # also clears the CP interaction state + warp timer
        if section is None:
            return
        # Persisted CP edits survive navigation; the section's dirty state and
        # last-saved baseline live in AppState.  Re-sync the baseline (a no-op
        # while dirty, so the stash survives).  The window refreshes the save
        # bar for the new section.
        self._state.sync_baseline(section.id, "warp", copy.deepcopy(section.warp))

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
            [(cp.dst_x / w_bg, cp.dst_y / h_bg) for cp in cps],
            w_bg,
            h_bg,
            self._cp_hovered,
            cp_size=self._cp_size,
            cp_shape=self._cp_shape,
            cp_color=self._cp_color,
            src_pts=[(cp.src_x / w_bg, cp.src_y / h_bg) for cp in cps],
            auto_flags=[cp.auto for cp in cps],
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

    def _clamped_pixel_pos(self, x: float, y: float) -> tuple[float, float] | None:
        raw = self._panel.raw_image
        if raw is None:
            return None
        h, w = raw.shape[:2]
        return min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))

    def _pick_cp(self, x: float, y: float) -> int:
        section = self._panel.section
        if section is None:
            return -1
        cps = section.warp.control_points
        if not cps:
            return -1
        best, best_d2 = -1, float(self._CP_PICK_RADIUS**2)
        for i, cp in enumerate(cps):
            d2 = (cp.dst_x - x) ** 2 + (cp.dst_y - y) ** 2
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
        cur = self._clamped_pixel_pos(x, y)
        if cur is None:
            return
        cp = section.warp.control_points[self._cp_dragging]
        raw = self._panel.raw_image
        if self._cp_drag_start_px is None or self._cp_drag_start_dst is None or raw is None:
            cp.dst_x, cp.dst_y = cur
            return
        sx, sy = self._cp_drag_start_px
        bx, by = self._cp_drag_start_dst
        h, w = raw.shape[:2]
        cp.dst_x = min(max(bx + cur[0] - sx, 0.0), float(w))
        cp.dst_y = min(max(by + cur[1] - sy, 0.0), float(h))

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
        raw = self._panel.raw_image
        if raw is None:
            return
        work_h, work_w = raw.shape[:2]
        self._push_undo()
        section.warp.control_points.append(ControlPoint(u * work_w, v * work_h, x, y))
        self._cp_hovered = len(section.warp.control_points) - 1
        self._panel.update_overlay()
        self._set_dirty(True)
        self.cp_changed.emit()

    def _on_canvas_drag_started(self, x: float, y: float) -> None:
        self._cp_dragging = self._pick_cp(x, y)
        self._cp_drag_start_px = self._clamped_pixel_pos(x, y)
        self._cp_drag_start_dst = None
        section = self._panel.section
        if section is not None and self._cp_dragging >= 0:
            # Snapshot once before the move so the whole drag undoes in one step.
            self._push_undo()
            cp = section.warp.control_points[self._cp_dragging]
            self._cp_drag_start_dst = (cp.dst_x, cp.dst_y)
            # Sample the outline cheaper while re-warping every frame; drag_ended
            # restores full resolution.
            self._panel.set_overlay_fast(True)

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        if self._cp_dragging < 0 or self._panel.section is None:
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        if not self._warp_timer.isActive():
            self._warp_timer.start()

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        self._warp_timer.stop()
        self._panel.set_overlay_fast(False)
        if self._cp_dragging < 0 or self._panel.section is None:
            self._cp_dragging = -1
            return
        self._move_dragged_cp_to(x, y)
        self._cp_hovered = self._cp_dragging
        self._cp_dragging = -1
        self._cp_drag_start_px = None
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
            self._push_undo()
            cps.pop(self._cp_hovered)
            self._cp_hovered = -1
            self._panel.update_overlay()
            self._set_dirty(True)
            self.cp_changed.emit()

    # ------------------------------------------------------------------
    # Draft-view hooks (see DraftCanvasView)
    # ------------------------------------------------------------------

    def _current_section(self) -> Section | None:
        return self._panel.section

    def has_persisted_state(self) -> bool:
        baseline = self._saved_state()
        return baseline is not None and bool(baseline.control_points)

    def _capture_edit(self) -> list[ControlPoint]:
        return copy.deepcopy(self._panel.section.warp.control_points)

    def _restore(self, snapshot: list[ControlPoint]) -> None:
        self._panel.section.warp.control_points = snapshot
        self._panel.update_overlay()

    def _matches_saved(self, snapshot: list[ControlPoint]) -> bool:
        baseline = self._saved_state()
        base_cps = baseline.control_points if baseline is not None else []
        return snapshot == base_cps

    def _saved_copy(self) -> WarpState:
        return copy.deepcopy(self._panel.section.warp)

    def _commit(self) -> bool:
        """Commit the current control-point list as this slice's warp.

        Saving is also how the user *accepts* an auto-generated (elastix) warp:
        ``commit_warp`` promotes ``warp.status`` to COMPLETE (turning the
        proposal green) and the affine plane the warp sits on too, and resets an
        empty warp to NOT_STARTED so it reads gray.
        """
        commit_warp(self._panel.section)
        return True

    def _apply_saved(self, baseline: WarpState) -> None:
        self._panel.section.warp = copy.deepcopy(baseline)
        self._reset_cp_interaction()
        if self._active:
            self._panel.update_overlay()

    def _wipe(self) -> None:
        section = self._panel.section
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED
        self._reset_cp_interaction()
        if self._active:
            self._panel.update_overlay()

    def _end_edit_gesture(self) -> None:
        self._warp_timer.stop()
        self._reset_cp_interaction()

    def _after_undo_restore(self) -> None:
        self.cp_changed.emit()

    def _reset_cp_interaction(self) -> None:
        """Clear transient control-point hover / drag state."""
        self._cp_hovered = -1
        self._cp_dragging = -1
        self._cp_drag_start_px = None
        self._cp_drag_start_dst = None
