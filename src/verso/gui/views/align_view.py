"""Align view — atlas registration canvas.

Composes the shared :class:`SectionCanvasPanel` (created by ``MainWindow``
and reparented into whichever view is active) and contributes the affine
toolbar (scale / store / revert / clear / proposals) plus the
orthogonal :class:`NavigatorPanel` on the left, which carries the per-view
translate / rotate buttons next to each slice.

Warp-mode interaction lives in a sibling view, ``WarpView``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus
from verso.gui.widgets.navigator import NavigatorPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel

if TYPE_CHECKING:
    from PyQt6.QtCore import pyqtBoundSignal  # noqa: F401

from PyQt6.QtCore import pyqtSignal

# Scale increment per button click (2 %, matching QuickNII)
_SCALE_STEP = 1.02


class AlignView(QWidget):
    """Canvas view for atlas alignment (affine anchoring)."""

    section_modified = pyqtSignal()
    anchoring_changed = pyqtSignal(list)
    alignments_updated = pyqtSignal()
    reverse_requested = pyqtSignal()
    deepslice_requested = pyqtSignal()
    default_proposal_requested = pyqtSignal()
    clear_all_alignments_requested = pyqtSignal()

    def __init__(self, panel: SectionCanvasPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._reverse_ap = False

        self._build_ui()
        self._wire_panel()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_toolbar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._navigator = NavigatorPanel()
        self._navigator.anchoring_changed.connect(self._on_navigator_changed)
        body.addWidget(self._navigator)

        self._panel_slot = QWidget()
        slot_layout = QHBoxLayout(self._panel_slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        slot_layout.setSpacing(0)
        body.addWidget(self._panel_slot, stretch=1)

        root.addLayout(body, stretch=1)

    def _make_toolbar(self) -> QWidget:
        """Build the align toolbar.

        Uses a real :class:`QToolBar` so Qt's built-in overflow extension
        (the ``>>`` button) kicks in when the window is too narrow to fit
        every button — without this, the sum of ``setFixedWidth`` buttons
        pins the window's minimum width at ~1400 px.

        The status label is placed *outside* the toolbar so the section
        filename remains visible even when buttons spill into the extension
        popup.
        """
        small_btn_qss = (
            "QPushButton { border-radius: 3px; padding: 2px 7px; color: #ccc;"
            " background: #383838; border: 1px solid #555; font-size: 11px; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        plain_btn_qss = (
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #383838; border: 1px solid #555; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )
        red_btn_qss = (
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #5a2a2a; }"
            "QPushButton:hover { background: #6a3a3a; }"
            "QPushButton:disabled { color: #666; background: #333; }"
        )
        green_btn_qss = (
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #2a5a2a; }"
            "QPushButton:hover { background: #3a6a3a; }"
            "QPushButton:disabled { color: #666; background: #333; }"
        )
        yellow_btn_qss = (
            "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
            " background: #3a3a1a; border: 1px solid #666; }"
            "QPushButton:hover { background: #4a4a2a; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
        )

        tb = QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(tb.iconSize())  # no-op, but ensures size-hint is computed
        tb.setStyleSheet(
            "QToolBar { background: #252525; spacing: 4px; padding: 2px 6px;"
            " border: none; }"
            "QToolBar::separator { background: #444; width: 1px;"
            " margin: 6px 4px; }"
        )
        # Let the toolbar shrink horizontally; Qt will surface a `>>` extension
        # button for any items that don't fit.
        tb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        tb.addWidget(self._panel.make_outline_button())
        tb.addSeparator()

        scale_specs = [
            ("↔+", "Wider (2%)",    1.0 / _SCALE_STEP,  1.0),
            ("↔−", "Narrower (2%)", _SCALE_STEP,   1.0),
            ("↕+", "Taller (2%)",   1.0,        1.0 / _SCALE_STEP),
            ("↕−", "Shorter (2%)", 1.0,    _SCALE_STEP),
        ]
        self._scale_btns: list[QPushButton] = []
        for sym, tip, su, sv in scale_specs:
            btn = QPushButton(sym)
            btn.setFixedHeight(28)
            btn.setFixedWidth(32)
            btn.setToolTip(tip)
            btn.setStyleSheet(small_btn_qss)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=su, t=sv: self._scale_overlay(s, t))
            tb.addWidget(btn)
            self._scale_btns.append(btn)

        tb.addSeparator()

        self._deepslice_btn = QPushButton("Run DeepSlice")
        self._deepslice_btn.setFixedHeight(28)
        self._deepslice_btn.setToolTip("Generate editable affine suggestions with DeepSlice")
        self._deepslice_btn.setStyleSheet(plain_btn_qss)
        self._deepslice_btn.setEnabled(False)
        self._deepslice_btn.clicked.connect(self.deepslice_requested)
        tb.addWidget(self._deepslice_btn)

        self._default_btn = QPushButton("Default proposal")
        self._default_btn.setFixedHeight(28)
        self._default_btn.setToolTip("Revert editable suggestions to VERSO's default AP proposal")
        self._default_btn.setStyleSheet(plain_btn_qss)
        self._default_btn.setEnabled(False)
        self._default_btn.clicked.connect(self.default_proposal_requested)
        tb.addWidget(self._default_btn)

        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.setFixedHeight(28)
        self._clear_all_btn.setToolTip(
            "Clear every stored alignment and restore the default AP proposal"
        )
        self._clear_all_btn.setStyleSheet(red_btn_qss)
        self._clear_all_btn.setEnabled(False)
        self._clear_all_btn.clicked.connect(self.clear_all_alignments_requested)
        tb.addWidget(self._clear_all_btn)

        self._reverse_btn = QPushButton("Reverse proposal")
        self._reverse_btn.setFixedHeight(28)
        self._reverse_btn.setToolTip(
            "Reverse the initial AP proposal before storing any alignment"
        )
        self._reverse_btn.setStyleSheet(plain_btn_qss)
        self._reverse_btn.setEnabled(False)
        self._reverse_btn.clicked.connect(self.reverse_requested)
        tb.addWidget(self._reverse_btn)

        self._store_btn = QPushButton("Store")
        self._store_btn.setFixedHeight(28)
        self._store_btn.setToolTip("Lock current atlas plane to this section")
        self._store_btn.setStyleSheet(green_btn_qss)
        self._store_btn.setEnabled(False)
        self._store_btn.clicked.connect(self._store_anchoring)
        tb.addWidget(self._store_btn)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setFixedHeight(28)
        self._revert_btn.setToolTip("Restore the last stored plane, discarding unsaved edits")
        self._revert_btn.setStyleSheet(yellow_btn_qss)
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self._revert_to_stored)
        tb.addWidget(self._revert_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setToolTip("Remove stored plane and revert to interpolated")
        self._clear_btn.setStyleSheet(red_btn_qss)
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_anchoring)
        tb.addWidget(self._clear_btn)

        # Wrap the toolbar + a status label so the filename stays visible even
        # when buttons overflow into the extension popup.
        container = QWidget()
        container.setFixedHeight(36)
        container.setStyleSheet("background: #252525;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(0)
        layout.addWidget(tb, stretch=1)
        layout.addWidget(self._panel.make_status_label())
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

    def set_reverse_enabled(self, enabled: bool) -> None:
        self._reverse_btn.setEnabled(enabled)

    def set_deepslice_enabled(self, enabled: bool, running: bool = False) -> None:
        self._deepslice_btn.setEnabled(enabled and not running)
        self._deepslice_btn.setText("DeepSlice running..." if running else "Run DeepSlice")
        self._default_btn.setEnabled(enabled and not running)
        self._clear_all_btn.setEnabled(enabled and not running)

    # ------------------------------------------------------------------
    # Panel events
    # ------------------------------------------------------------------

    def _on_section_loaded(self, section) -> None:
        for btn in self._scale_btns:
            btn.setEnabled(section is not None)
        self._store_btn.setEnabled(section is not None)
        if section is None:
            self._clear_btn.setEnabled(False)
            self._revert_btn.setEnabled(False)
            self._navigator.set_anchoring(None)
            return
        has_anchoring = bool(section.alignment.anchoring) and any(
            v != 0.0 for v in section.alignment.anchoring
        )
        self._clear_btn.setEnabled(has_anchoring)
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
        self._revert_btn.setEnabled(has_stored)

    def _revert_to_stored(self) -> None:
        section = self._panel.section
        if section is None:
            return
        stored = section.alignment.stored_anchoring
        if not stored or all(v == 0.0 for v in stored):
            return
        from verso.engine.registration import flip_anchoring_horizontal, flip_anchoring_vertical
        display = list(stored)
        if section.preprocessing.flip_horizontal:
            display = flip_anchoring_horizontal(display)
        if section.preprocessing.flip_vertical:
            display = flip_anchoring_vertical(display)
        section.alignment.anchoring = display
        self._sync_ap_from_anchoring(display)
        self._panel.update_overlay()
        self.anchoring_changed.emit(display)
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
        from verso.engine.registration import flip_anchoring_horizontal, flip_anchoring_vertical
        stored = list(section.alignment.anchoring)
        if section.preprocessing.flip_horizontal:
            stored = flip_anchoring_horizontal(stored)
        if section.preprocessing.flip_vertical:
            stored = flip_anchoring_vertical(stored)
        section.alignment.stored_anchoring = stored
        section.alignment.status = AlignmentStatus.COMPLETE
        self._clear_btn.setEnabled(True)
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
        self._clear_btn.setEnabled(True)
        self._update_revert_enabled()
