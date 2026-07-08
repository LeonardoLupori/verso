"""Align view — atlas registration canvas.

Composes the shared :class:`SectionCanvasPanel` (created by ``MainWindow``
and reparented into whichever view is active) and contributes a thin
status bar plus the orthogonal :class:`NavigatorPanel` on the left, which
carries the per-view translate / rotate buttons next to each slice.

Edits made via the navigator are drafts: they live in memory only and are
discarded on slice / view change.  The shared Save / Clear bar in the
Align properties page commits or wipes them.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from verso.engine.drafts import commit_alignment
from verso.engine.model.alignment import Alignment
from verso.gui.utils import require
from verso.gui.views.base_canvas_view import BaseCanvasView
from verso.gui.widgets.navigator import NavigatorPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from PyQt6.QtCore import pyqtBoundSignal  # noqa: F401

    from verso.engine.model.project import Section
    from verso.gui.state import AppState

from PyQt6.QtCore import Qt, QTimer, pyqtSignal

# Maximum in-plane spin (degrees) from axis-aligned allowed via the align handle.
_MAX_INPLANE_DEG = 45.0


class AlignView(BaseCanvasView):
    """Canvas view for atlas alignment (affine anchoring)."""

    STEP = "align"

    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()

    def __init__(
        self,
        panel: SectionCanvasPanel,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, parent)
        self._panel = panel
        self._reverse_axis = False
        self._interpolation_axis = 1
        # Dirty flag, last-saved baseline, and the undo stack live in the base
        # (BaseCanvasView) keyed by (section.id, "align").  A continuous run of
        # pan events is coalesced into a single undo step via the idle timer.
        self._active = False
        self._pan_run_active = False
        self._pan_coalesce_timer = QTimer(self)
        self._pan_coalesce_timer.setSingleShot(True)
        self._pan_coalesce_timer.setInterval(300)
        self._pan_coalesce_timer.timeout.connect(self._end_pan_run)

        self._build_ui()
        self._wire_panel()
        self._wire_shortcuts()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_status_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._navigator = NavigatorPanel()
        self._navigator.anchoring_changed.connect(self._on_navigator_changed)
        self._navigator.scale_requested.connect(self._scale_overlay)
        body.addWidget(self._navigator)

        self._panel_slot = QWidget()
        slot_layout = QHBoxLayout(self._panel_slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        slot_layout.setSpacing(0)
        body.addWidget(self._panel_slot, stretch=1)

        root.addLayout(body, stretch=1)

    def _make_status_bar(self) -> QWidget:
        """Thin bar that just shows the current section filename."""
        return make_view_status_bar(self._panel.make_status_label())

    def _wire_panel(self) -> None:
        self._panel.overlay_panned.connect(self._on_overlay_panned)
        self._panel.overlay_rotated.connect(self._on_overlay_rotated)
        self._panel.overlay_scaled.connect(self._on_overlay_scaled)
        self._panel.overlay_updated.connect(self._on_overlay_updated)
        self._panel.section_loaded.connect(self._on_section_loaded)
        self._panel.atlas_changed.connect(lambda _atlas: self._navigator.set_atlas(_atlas))

    def _wire_shortcuts(self) -> None:
        """Install the Ctrl+Z undo shortcut.

        Scoped to this view and its children (the reparented canvas panel and
        navigator) so it only fires while the Align view is active.
        """
        undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo.activated.connect(self.undo)
        self._undo_shortcut = undo

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Reparent the shared panel into this view."""
        self._active = True
        require(self._panel_slot.layout()).addWidget(self._panel)
        self._panel.canvas.set_interaction_mode("align")
        self._panel.canvas.clear_control_points()
        # Align view has no overlay post-processing
        self._panel.overlay_post_processor = None
        self._panel.cursor_to_atlas_mapper = None
        self._panel.update_overlay()
        # Re-sync the baseline in case the section was loaded before activate.
        # A no-op while dirty, so the stashed last-saved plane survives.  The
        # save bar's dirty state is refreshed by the window on view entry.
        self._reset_undo()
        section = self._panel.section
        if section is not None:
            self._state.sync_baseline(section.id, "align", copy.deepcopy(section.alignment))

    def deactivate(self) -> None:
        """Release any state set on the panel."""
        # Currently align installs no panel-level hooks, so nothing to clear.
        self._active = False

    # ------------------------------------------------------------------
    # External API used by MainWindow
    # ------------------------------------------------------------------

    def set_reverse_axis(self, reverse: bool) -> None:
        """Invert slicing-axis movement and tilt directions when the series is reversed."""
        self._reverse_axis = reverse
        self._navigator.set_reverse_axis(reverse)

    def set_interpolation_axis(self, axis: int) -> None:
        """Update the anchoring voxel axis used to compute ``position_mm``."""
        self._interpolation_axis = int(axis)
        self._navigator.set_interpolation_axis(axis)

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
        self._navigator.set_stretch_enabled(section is not None)
        self._reset_undo()
        if section is None:
            self._navigator.set_anchoring(None)
            return
        # Persisted edits are not discarded on navigation; the section's dirty
        # state and last-saved baseline live in AppState.  Re-sync the baseline
        # (a no-op while dirty, so the stash survives).  The window refreshes the
        # save bar for the new section.
        self._state.sync_baseline(section.id, "align", copy.deepcopy(section.alignment))

    def _on_overlay_updated(self, anchoring, _display_w, _display_h) -> None:
        self._navigator.set_anchoring(anchoring)

    # ------------------------------------------------------------------
    # Navigator handler
    # ------------------------------------------------------------------

    def _on_navigator_changed(self, new_anchoring: list[float]) -> None:
        section = self._panel.section
        if section is None:
            return
        self._end_pan_run()
        self._push_undo()
        section.alignment.current_anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    def _sync_position_from_anchoring(self, anchoring: list[float]) -> None:
        section = self._panel.section
        atlas = self._panel.atlas
        if section is None or atlas is None:
            return
        center = atlas.cut_center(anchoring)
        section.alignment.position_mm = atlas.voxel_to_mm(center[self._interpolation_axis])

    # ------------------------------------------------------------------
    # Overlay pan (space + drag)
    # ------------------------------------------------------------------

    def _on_overlay_panned(self, dx: float, dy: float) -> None:
        section = self._panel.section
        raw = self._panel.raw_image
        if section is None or raw is None:
            return
        anchoring = section.alignment.current_anchoring
        if not section.alignment.is_anchored:
            return
        # Coalesce the whole drag gesture into a single undo step: snapshot
        # once when the run starts, then keep the idle timer alive so the run
        # only ends after the user stops dragging.
        if not self._pan_run_active:
            self._push_undo()
            self._pan_run_active = True
            # Canvas gestures must move keyboard focus into this view's subtree
            # so the Ctrl+Z shortcut (WidgetWithChildrenShortcut) still fires.
            self._panel.canvas.view.setFocus()
            # Pan re-renders the outline every move event; sample it cheaper for
            # the duration of the gesture (restored in _end_pan_run).
            self._panel.set_overlay_fast(True)
        self._pan_coalesce_timer.start()
        h_bg, w_bg = raw.shape[:2]
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        new_o = o - (dx / w_bg) * u - (dy / h_bg) * v
        new_anchoring = new_o.tolist() + anchoring[3:]
        section.alignment.current_anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Overlay rotate (handle ring drag)
    # ------------------------------------------------------------------

    def _on_overlay_rotated(self, d_deg: float) -> None:
        section = self._panel.section
        if section is None or self._panel.raw_image is None:
            return
        anchoring = section.alignment.current_anchoring
        if not section.alignment.is_anchored:
            return

        from verso.engine.anchoring import clamp_inplane_rotation, rotate_anchoring

        # In-plane spin about the section centre. A clockwise drag on the ring
        # spins the overlay clockwise (canvas y points down, so a positive
        # screen-space sweep maps straight onto the rotation). Cap it so the
        # overlay never spins more than ±_MAX_INPLANE_DEG from axis-aligned.
        angle_rad = clamp_inplane_rotation(
            anchoring, float(np.radians(d_deg)), self._interpolation_axis, _MAX_INPLANE_DEG
        )
        if angle_rad == 0.0:
            return  # already at the rotation limit — ignore further outward spin

        # Coalesce the whole ring drag into a single undo step, mirroring the pan
        # gesture: snapshot once, keep the idle timer alive, sample the outline
        # cheaper while dragging.
        if not self._pan_run_active:
            self._push_undo()
            self._pan_run_active = True
            # See _on_overlay_panned: grab focus so Ctrl+Z keeps working.
            self._panel.canvas.view.setFocus()
            self._panel.set_overlay_fast(True)
        self._pan_coalesce_timer.start()

        new_anchoring = rotate_anchoring(anchoring, angle_rad)
        section.alignment.current_anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Overlay stretch (handle grip drag)
    # ------------------------------------------------------------------

    def _on_overlay_scaled(self, scale_s: float, scale_t: float) -> None:
        section = self._panel.section
        if section is None or self._panel.raw_image is None:
            return
        anchoring = section.alignment.current_anchoring
        if not section.alignment.is_anchored:
            return
        # Coalesce the whole grip drag into a single undo step, mirroring pan/rotate.
        if not self._pan_run_active:
            self._push_undo()
            self._pan_run_active = True
            # See _on_overlay_panned: grab focus so Ctrl+Z keeps working.
            self._panel.canvas.view.setFocus()
            self._panel.set_overlay_fast(True)
        self._pan_coalesce_timer.start()

        from verso.engine.anchoring import scale_anchoring

        new_anchoring = scale_anchoring(anchoring, scale_s, scale_t)
        section.alignment.current_anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Affine controls
    # ------------------------------------------------------------------

    def _scale_overlay(self, scale_u: float, scale_v: float) -> None:
        section = self._panel.section
        if section is None or self._panel.raw_image is None:
            return
        anchoring = section.alignment.current_anchoring
        if not section.alignment.is_anchored:
            return
        self._end_pan_run()
        self._push_undo()
        from verso.engine.anchoring import scale_anchoring

        new_anchoring = scale_anchoring(anchoring, scale_u, scale_v)
        section.alignment.current_anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Draft-view hooks (see BaseCanvasView)
    # ------------------------------------------------------------------

    def _current_section(self) -> Section | None:
        return self._panel.section

    def has_persisted_state(self) -> bool:
        section = self._panel.section
        if section is None:
            return False
        # ``stored_anchoring`` only changes on save/clear, so it reflects the
        # last-saved plane even mid-edit — no baseline lookup needed.
        from verso.engine.anchoring import is_anchored

        return is_anchored(section.alignment.stored_anchoring)

    def _capture_edit(self) -> list[float]:
        return list(self._panel.section.alignment.current_anchoring)

    def _restore(self, snapshot: list[float]) -> None:
        section = self._panel.section
        section.alignment.current_anchoring = snapshot
        self._sync_position_from_anchoring(snapshot)
        self._panel.update_overlay()

    def _matches_saved(self, snapshot: list[float]) -> bool:
        baseline = self._saved_state()
        base_anchoring = baseline.current_anchoring if baseline is not None else None
        return snapshot == base_anchoring

    def _saved_copy(self) -> Alignment:
        return copy.deepcopy(self._panel.section.alignment)

    def _commit(self) -> bool:
        """Seed a default plane if untouched, then promote to stored + COMPLETE."""
        section = self._panel.section
        atlas = self._panel.atlas
        raw = self._panel.raw_image
        if atlas is None:
            return False
        # If the user never touched the navigator but hits Save, seed a default
        # plane so there's something to store.
        if not section.alignment.is_anchored:
            if raw is None:
                return False
            h, w = raw.shape[:2]
            section.alignment.current_anchoring = atlas.default_anchoring(
                axis=self._interpolation_axis,
                aspect_ratio=w / h,
            )
            self._sync_position_from_anchoring(section.alignment.current_anchoring)
        commit_alignment(section)
        return True

    def _apply_saved(self, baseline: Alignment) -> None:
        section = self._panel.section
        section.alignment = copy.deepcopy(baseline)
        self._sync_position_from_anchoring(section.alignment.current_anchoring)
        self._panel.update_overlay()

    def _wipe(self) -> None:
        """Wipe the alignment (and the slice's warp, which depended on it)."""
        from verso.engine.drafts import reset_alignment

        reset_alignment(self._panel.section)

    def _end_edit_gesture(self) -> None:
        self._end_pan_run()

    def _after_undo_restore(self) -> None:
        self.anchoring_changed.emit(list(self._panel.section.alignment.current_anchoring))

    def _after_save(self) -> None:
        self.alignments_updated.emit()

    def _after_revert(self) -> None:
        self.anchoring_changed.emit(list(self._panel.section.alignment.current_anchoring))
        self.alignments_updated.emit()

    def _after_clear(self) -> None:
        self.alignments_updated.emit()

    # ------------------------------------------------------------------
    # Pan-gesture coalescing
    # ------------------------------------------------------------------

    def _end_pan_run(self) -> None:
        """Close the current pan gesture so the next drag starts a fresh undo step."""
        was_active = self._pan_run_active
        self._pan_run_active = False
        self._pan_coalesce_timer.stop()
        if was_active:
            # Restore full display resolution now that the gesture has settled.
            self._panel.set_overlay_fast(False)
            if self._active:
                self._panel.update_overlay()
