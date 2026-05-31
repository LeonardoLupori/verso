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

from PyQt6.QtCore import Qt, pyqtSignal


def _is_set(anchoring: list[float] | None) -> bool:
    """True when *anchoring* exists and is not the all-zero placeholder."""
    return bool(anchoring) and any(v != 0.0 for v in anchoring)


class AlignView(QWidget):
    """Canvas view for atlas alignment (affine anchoring)."""

    dirty_changed = pyqtSignal(bool)
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()

    def __init__(self, panel: SectionCanvasPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._reverse_axis = False
        self._interpolation_axis = 1
        # Snapshot of section.alignment taken at section-load time; restored
        # by discard() so unsaved navigator edits roll back when the user
        # switches slice or view.
        self._baseline_alignment: Alignment | None = None
        self._dirty = False

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
        """Install the Ctrl+Z shortcut that reverts the anchoring.

        Uses a WindowShortcut gated on visibility (matching PrepView) so it
        fires no matter which widget in the window holds focus — the canvas,
        the navigator, or the properties dock — but only while the Align view
        is the active page.
        """
        undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo.setContext(Qt.ShortcutContext.WindowShortcut)
        undo.activated.connect(self._on_undo_shortcut)
        self._undo_shortcut = undo

    def _on_undo_shortcut(self) -> None:
        if self.isVisible():
            self.revert_anchoring()

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
        section = self._panel.section
        if section is not None:
            self._baseline_alignment = copy.deepcopy(section.alignment)
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
        if section is None:
            self._baseline_alignment = None
            self._navigator.set_anchoring(None)
            self._set_dirty(False)
            return
        self._baseline_alignment = copy.deepcopy(section.alignment)
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
        from verso.engine.registration import scale_anchoring
        new_anchoring = scale_anchoring(anchoring, scale_u, scale_v)
        section.alignment.anchoring = new_anchoring
        self._sync_position_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self._set_dirty(True)
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Undo (Ctrl+Z) — revert to the saved or proposed anchoring
    # ------------------------------------------------------------------

    def revert_anchoring(self) -> bool:
        """Snap the overlay back to the saved anchoring, else the proposal.

        Triggered by Ctrl+Z.  Restores ``stored_anchoring`` (the last Save) if
        present; otherwise the auto-suggested ``proposal_anchoring``; otherwise
        the load-time baseline (which, for an un-saved section, is the default
        QuickNII proposal — those never populate ``proposal_anchoring``).
        Returns False (no-op) when none exists or nothing would change.
        """
        section = self._panel.section
        if section is None:
            return False
        alignment = section.alignment

        target: list[float] | None = None
        if _is_set(alignment.stored_anchoring):
            target = list(alignment.stored_anchoring)
        elif _is_set(alignment.proposal_anchoring):
            target = list(alignment.proposal_anchoring)
        elif self._baseline_alignment is not None and _is_set(
            self._baseline_alignment.anchoring
        ):
            target = list(self._baseline_alignment.anchoring)

        if target is None or target == list(alignment.anchoring):
            return False

        alignment.anchoring = target
        self._sync_position_from_anchoring(target)
        self._panel.update_overlay()
        # Reverting to the saved plane is a clean state; reverting to a proposal
        # (no save yet) leaves a draft the user can still Save or Clear.
        base = self._baseline_alignment.anchoring if self._baseline_alignment else None
        self._set_dirty(target != base)
        self.anchoring_changed.emit(target)
        return True

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
        self._set_dirty(False)
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
        self._set_dirty(False)
        self.alignments_updated.emit()
        return True

    def discard(self) -> None:
        """Restore the section's alignment from the baseline snapshot."""
        section = self._panel.section
        if section is None or self._baseline_alignment is None:
            self._set_dirty(False)
            return
        section.alignment = copy.deepcopy(self._baseline_alignment)
        self._set_dirty(False)
        self._panel.update_overlay()
        self.anchoring_changed.emit(section.alignment.anchoring)

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self.dirty_changed.emit(dirty)
