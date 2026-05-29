"""Align view — atlas registration canvas.

Composes the shared :class:`SectionCanvasPanel` (created by ``MainWindow``
and reparented into whichever view is active) and contributes a thin
status bar plus the orthogonal :class:`NavigatorPanel` on the left, which
carries the per-view translate / rotate buttons next to each slice. The
store / revert / clear / clear-all buttons live in the properties panel
(:class:`AlignActionsBox`) and are wired in via :meth:`bind_actions`.

Warp-mode interaction lives in a sibling view, ``WarpView``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus
from verso.gui.widgets.navigator import NavigatorPanel
from verso.gui.widgets.properties.sections import AlignActionsBox
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel

if TYPE_CHECKING:
    from PyQt6.QtCore import pyqtBoundSignal  # noqa: F401

from PyQt6.QtCore import pyqtSignal


class AlignView(QWidget):
    """Canvas view for atlas alignment (affine anchoring)."""

    section_modified = pyqtSignal()
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()
    clear_all_alignments_requested = pyqtSignal()

    def __init__(self, panel: SectionCanvasPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._reverse_ap = False
        self._actions: AlignActionsBox | None = None

        self._build_ui()
        self._wire_panel()

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
        container = QWidget()
        container.setFixedHeight(36)
        container.setStyleSheet("background: #252525;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)
        layout.addWidget(self._panel.make_status_label())
        layout.addStretch(1)
        return container

    def _wire_panel(self) -> None:
        self._panel.overlay_panned.connect(self._on_overlay_panned)
        self._panel.overlay_updated.connect(self._on_overlay_updated)
        self._panel.section_loaded.connect(self._on_section_loaded)
        self._panel.atlas_changed.connect(lambda _atlas: self._navigator.set_atlas(_atlas))

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

    def deactivate(self) -> None:
        """Release any state set on the panel."""
        # Currently align installs no panel-level hooks, so nothing to clear.
        pass

    # ------------------------------------------------------------------
    # External API used by MainWindow
    # ------------------------------------------------------------------

    def set_reverse_ap(self, reverse: bool) -> None:
        """Invert AP movement and tilt directions when the series is AP-reversed."""
        self._reverse_ap = reverse
        self._navigator.set_reverse_ap(reverse)

    def bind_actions(self, actions: AlignActionsBox) -> None:
        """Wire the properties-panel action buttons to this view's handlers."""
        self._actions = actions
        actions.store_requested.connect(self._store_anchoring)
        actions.revert_requested.connect(self._revert_to_stored)
        actions.clear_requested.connect(self._clear_anchoring)
        actions.clear_all_requested.connect(self.clear_all_alignments_requested)

    def set_clear_all_enabled(self, enabled: bool) -> None:
        if self._actions is not None:
            self._actions.set_clear_all_enabled(enabled)

    # ------------------------------------------------------------------
    # Panel events
    # ------------------------------------------------------------------

    def _on_section_loaded(self, section) -> None:
        self._navigator.set_stretch_enabled(section is not None)
        if self._actions is not None:
            self._actions.set_store_enabled(section is not None)
        if section is None:
            if self._actions is not None:
                self._actions.set_clear_enabled(False)
                self._actions.set_revert_enabled(False)
            self._navigator.set_anchoring(None)
            return
        has_anchoring = bool(section.alignment.anchoring) and any(
            v != 0.0 for v in section.alignment.anchoring
        )
        if self._actions is not None:
            self._actions.set_clear_enabled(has_anchoring)
        self._update_revert_enabled()

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
        self._sync_ap_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    def _sync_ap_from_anchoring(self, anchoring: list[float]) -> None:
        section = self._panel.section
        atlas = self._panel.atlas
        if section is None or atlas is None:
            return
        center = atlas.cut_center(anchoring)
        section.alignment.ap_position_mm = atlas.ap_voxel_to_mm(center[atlas.ap_axis])

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
        self._sync_ap_from_anchoring(new_anchoring)
        self._panel.update_overlay()
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
        self._sync_ap_from_anchoring(new_anchoring)
        self._panel.update_overlay()
        self.anchoring_changed.emit(new_anchoring)

    # ------------------------------------------------------------------
    # Store / Revert / Clear
    # ------------------------------------------------------------------

    def _update_revert_enabled(self) -> None:
        section = self._panel.section
        has_stored = (
            section is not None
            and section.alignment.stored_anchoring is not None
            and any(v != 0.0 for v in section.alignment.stored_anchoring)
        )
        if self._actions is not None:
            self._actions.set_revert_enabled(has_stored)

    def _revert_to_stored(self) -> None:
        section = self._panel.section
        if section is None:
            return
        stored = section.alignment.stored_anchoring
        if not stored or all(v == 0.0 for v in stored):
            return
        section.alignment.anchoring = list(stored)
        self._sync_ap_from_anchoring(stored)
        self._panel.update_overlay()
        self.anchoring_changed.emit(stored)
        self.section_modified.emit()

    def _store_anchoring(self) -> None:
        section = self._panel.section
        atlas = self._panel.atlas
        raw = self._panel.raw_image
        if section is None or atlas is None:
            return
        if not section.alignment.anchoring or all(
            v == 0.0 for v in section.alignment.anchoring
        ):
            if raw is None:
                return
            h, w = raw.shape[:2]
            section.alignment.anchoring = atlas.default_anchoring(w / h)
        section.alignment.stored_anchoring = list(section.alignment.anchoring)
        section.alignment.status = AlignmentStatus.COMPLETE
        if self._actions is not None:
            self._actions.set_clear_enabled(True)
        self._update_revert_enabled()
        self.section_modified.emit()
        self.alignments_updated.emit()

    def _clear_anchoring(self) -> None:
        section = self._panel.section
        if section is None:
            return
        section.alignment.anchoring = [0.0] * 9
        section.alignment.ap_position_mm = None
        section.alignment.status = AlignmentStatus.NOT_STARTED
        section.alignment.source = None
        section.alignment.stored_anchoring = None
        section.alignment.proposal_anchoring = None
        section.alignment.proposal_confidence = None
        section.alignment.proposal_run_id = None
        section.warp.control_points.clear()
        self.alignments_updated.emit()
        if self._actions is not None:
            self._actions.set_clear_enabled(True)
        self._update_revert_enabled()
