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

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.gui.widgets.navigator import NavigatorPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from PyQt6.QtCore import pyqtBoundSignal  # noqa: F401

    from verso.gui.state import AppState

from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class AlignView(QWidget):
    """Canvas view for atlas alignment (affine anchoring)."""

    dirty_changed = pyqtSignal(bool)
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()

    def __init__(
        self,
        panel: SectionCanvasPanel,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._state = state
        self._reverse_axis = False
        self._interpolation_axis = 1
        # Snapshot of section.alignment taken at section-load time; used as the
        # undo floor and to report the last-saved plane.  Navigator edits are no
        # longer discarded on navigation — they persist on the Section and the
        # section's dirty state is tracked in the edit registry.
        self._baseline_alignment: Alignment | None = None
        self._dirty = False

        # In-memory undo history of anchoring snapshots for the active slice.
        # Reset whenever the baseline is re-snapshotted (section / view change,
        # save, clear, discard).  A continuous run of pan events is coalesced
        # into a single undo step via the idle timer below.
        self._undo_stack: list[list[float]] = []
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
        self._panel_slot.layout().addWidget(self._panel)
        self._panel.canvas.set_interaction_mode("align")
        self._panel.canvas.clear_control_points()
        # Align view has no overlay post-processing
        self._panel.overlay_post_processor = None
        self._panel.cursor_to_atlas_mapper = None
        self._panel.update_overlay()
        # Re-snapshot in case the section was already loaded before activate.
        self._reset_undo()
        section = self._panel.section
        if section is not None:
            if self._state.is_dirty(section.id, "align"):
                stashed = self._state.get_baseline(section.id, "align")
                self._baseline_alignment = (
                    stashed
                    if stashed is not None
                    else copy.deepcopy(section.alignment)
                )
                self._set_dirty(True)
            else:
                self._baseline_alignment = copy.deepcopy(section.alignment)
                self._state.pop_baseline(section.id, "align")
                self._set_dirty(False)

    def deactivate(self) -> None:
        """Release any state set on the panel."""
        # Currently align installs no panel-level hooks, so nothing to clear.
        pass

    # ------------------------------------------------------------------
    # External API used by MainWindow
    # ------------------------------------------------------------------

    def set_reverse_axis(self, reverse: bool) -> None:
        """Invert slicing-axis movement and tilt directions when the series is reversed."""
        self._reverse_axis = reverse
        self._navigator.set_reverse_axis(reverse)

    def set_interpolation_axis(self, axis: int) -> None:
        """Update the QuickNII voxel axis used to compute ``position_mm``."""
        self._interpolation_axis = int(axis)
        self._navigator.set_interpolation_axis(axis)

    # ------------------------------------------------------------------
    # Panel events
    # ------------------------------------------------------------------

    def _on_section_loaded(self, section) -> None:
        self._navigator.set_stretch_enabled(section is not None)
        self._reset_undo()
        if section is None:
            self._baseline_alignment = None
            self._navigator.set_anchoring(None)
            self._set_dirty(False)
            return
        # Persisted edits are not discarded on navigation; reflect the section's
        # registry dirty state instead of forcing clean.  When still dirty,
        # recover the genuine last-saved baseline from the stash.
        if self._state.is_dirty(section.id, "align"):
            stashed = self._state.get_baseline(section.id, "align")
            self._baseline_alignment = (
                stashed if stashed is not None else copy.deepcopy(section.alignment)
            )
            self._set_dirty(True)
        else:
            self._baseline_alignment = copy.deepcopy(section.alignment)
            self._state.pop_baseline(section.id, "align")
            self._set_dirty(False)

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
        section.alignment.anchoring = new_anchoring
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
        anchoring = section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        # Coalesce the whole drag gesture into a single undo step: snapshot
        # once when the run starts, then keep the idle timer alive so the run
        # only ends after the user stops dragging.
        if not self._pan_run_active:
            self._push_undo()
            self._pan_run_active = True
        self._pan_coalesce_timer.start()
        h_bg, w_bg = raw.shape[:2]
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        new_o = o - (dx / w_bg) * u - (dy / h_bg) * v
        new_anchoring = new_o.tolist() + anchoring[3:]
        section.alignment.anchoring = new_anchoring
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
        anchoring = section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        self._end_pan_run()
        self._push_undo()
        from verso.engine.registration import scale_anchoring
        new_anchoring = scale_anchoring(anchoring, scale_u, scale_v)
        section.alignment.anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Draft / save / clear / discard
    # ------------------------------------------------------------------

    def is_dirty(self) -> bool:
        return self._dirty

    def has_persisted_state(self) -> bool:
        baseline = self._baseline_alignment
        if baseline is None:
            return False
        stored = baseline.stored_anchoring
        return bool(stored) and any(v != 0.0 for v in stored)

    def save(self) -> bool:
        """Promote the current anchoring to ``stored_anchoring`` + COMPLETE."""
        section = self._panel.section
        atlas = self._panel.atlas
        raw = self._panel.raw_image
        if section is None or atlas is None:
            return False
        # If the user never touched the navigator but hits Save, seed a default
        # plane so there's something to store.
        if not section.alignment.anchoring or all(
            v == 0.0 for v in section.alignment.anchoring
        ):
            if raw is None:
                return False
            h, w = raw.shape[:2]
            section.alignment.anchoring = atlas.default_anchoring(
                axis=self._interpolation_axis,
                aspect_ratio=w / h,
            )
            self._sync_position_from_anchoring(section.alignment.anchoring)
        section.alignment.stored_anchoring = list(section.alignment.anchoring)
        section.alignment.status = AlignmentStatus.COMPLETE
        self._baseline_alignment = copy.deepcopy(section.alignment)
        self._state.pop_baseline(section.id, "align")
        self._reset_undo()
        self._set_dirty(False)
        self.alignments_updated.emit()
        return True

    def revert(self) -> bool:
        """Discard unsaved anchoring edits, restoring the last-saved alignment."""
        section = self._panel.section
        if section is None or self._baseline_alignment is None:
            return False
        section.alignment = copy.deepcopy(self._baseline_alignment)
        self._state.pop_baseline(section.id, "align")
        self._reset_undo()
        self._set_dirty(False)
        self._sync_position_from_anchoring(section.alignment.anchoring)
        self._panel.update_overlay()
        self.anchoring_changed.emit(list(section.alignment.anchoring))
        self.alignments_updated.emit()
        return True

    def clear(self) -> bool:
        """Wipe the alignment (and the slice's warp, which depended on it)."""
        section = self._panel.section
        if section is None:
            return False
        section.alignment.anchoring = [0.0] * 9
        section.alignment.position_mm = None
        section.alignment.status = AlignmentStatus.NOT_STARTED
        section.alignment.source = None
        section.alignment.stored_anchoring = None
        section.alignment.proposal_anchoring = None
        section.alignment.proposal_confidence = None
        section.alignment.proposal_run_id = None
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED
        self._baseline_alignment = copy.deepcopy(section.alignment)
        self._state.pop_baseline(section.id, "align")
        self._reset_undo()
        self._set_dirty(False)
        self.alignments_updated.emit()
        return True

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Restore the previous anchoring from the undo history (Ctrl+Z)."""
        section = self._panel.section
        if section is None or not self._undo_stack:
            return
        self._end_pan_run()
        previous = self._undo_stack.pop()
        section.alignment.anchoring = previous
        self._sync_position_from_anchoring(previous)
        self._panel.update_overlay()
        baseline = self._baseline_alignment
        base_anchoring = baseline.anchoring if baseline is not None else None
        self._set_dirty(previous != base_anchoring)
        self.anchoring_changed.emit(previous)

    def _push_undo(self) -> None:
        """Snapshot the current anchoring before a mutating edit."""
        section = self._panel.section
        if section is None:
            return
        self._undo_stack.append(list(section.alignment.anchoring))
        # Bound the history so a long editing session can't grow without limit.
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _reset_undo(self) -> None:
        """Clear the undo history (called whenever the baseline is re-snapshotted)."""
        self._end_pan_run()
        self._undo_stack.clear()

    def _end_pan_run(self) -> None:
        """Close the current pan gesture so the next drag starts a fresh undo step."""
        self._pan_run_active = False
        self._pan_coalesce_timer.stop()

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        if dirty:
            section = self._panel.section
            if section is not None and self._baseline_alignment is not None:
                self._state.set_baseline(
                    section.id, "align", copy.deepcopy(self._baseline_alignment)
                )
        self._dirty = dirty
        self.dirty_changed.emit(dirty)
