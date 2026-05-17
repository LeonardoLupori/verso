"""Shared canvas panel reused by AlignView and WarpView.

Owns the ImageCanvas, the region label bar, and the section/atlas/channels
state plus the background-image and atlas-overlay pipelines.  Each view sets
its own ``overlay_post_processor`` and ``cursor_to_atlas_mapper`` hooks to
extend the pipeline (Warp injects ``warp_overlay`` and inverse-maps the
cursor for the region label).

A single instance lives in MainWindow and is reparented into whichever view
is currently active, so zoom/pan and channel cache survive mode switches.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import Section
from verso.gui.widgets.canvas import ImageCanvas

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume


_OUTLINE_BTN_QSS = (
    "QPushButton { border-radius: 4px; padding: 2px 10px; color: #ccc;"
    " background: #333; border: 1px solid #555; }"
    "QPushButton:checked { background: #4a4a1a; color: #ff0; border-color: #888; }"
    "QPushButton:hover { background: #444; }"
)

_REGION_BAR_IDLE_QSS = (
    "background: #1a1a1a; color: #fff; font-size: 12px; font-weight: bold;"
    " border-top: 1px solid #333;"
)


class SectionCanvasPanel(QWidget):
    """Canvas + region label bar shared between Align and Warp views."""

    # Re-exposed canvas signals
    mouse_position_changed = pyqtSignal(float, float)
    canvas_clicked = pyqtSignal(float, float)
    canvas_drag_started = pyqtSignal(float, float)
    canvas_dragged = pyqtSignal(float, float)
    canvas_drag_ended = pyqtSignal(float, float)
    overlay_panned = pyqtSignal(float, float)

    # Lifecycle / state-change notifications for the active view
    section_loaded = pyqtSignal(object)            # Section | None
    atlas_changed = pyqtSignal(object)             # AtlasVolume | None
    overlay_updated = pyqtSignal(list, int, int)   # anchoring, display_w, display_h
    outline_mode_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Shared state
        self._section: Section | None = None
        self._raw_image: np.ndarray | None = None
        self._atlas: AtlasVolume | None = None
        self._channels: list = []
        self._channel_layers: list[np.ndarray | None] = []
        self._cached_channel_specs: list[tuple] = []
        self._layer_image_key: tuple = ()
        self._outline_mode = False

        # Hooks set by the active view
        self.overlay_post_processor: Callable[[np.ndarray], np.ndarray] | None = None
        self.cursor_to_atlas_mapper: Callable[[float, float], tuple[float, float]] | None = None

        # Buttons/labels created via the make_* helpers — kept in sync here.
        self._outline_buttons: list[QPushButton] = []
        self._status_labels: list[QLabel] = []

        self._build_ui()
        self._wire_canvas()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self._region_bar = QLabel("")
        self._region_bar.setFixedHeight(38)
        self._region_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._region_bar.setStyleSheet(_REGION_BAR_IDLE_QSS)
        layout.addWidget(self._region_bar)

    def _wire_canvas(self) -> None:
        self.canvas.mouse_position_changed.connect(self._on_canvas_mouse_moved)
        self.canvas.canvas_clicked.connect(self.canvas_clicked)
        self.canvas.canvas_drag_started.connect(self.canvas_drag_started)
        self.canvas.canvas_dragged.connect(self.canvas_dragged)
        self.canvas.canvas_drag_ended.connect(self.canvas_drag_ended)
        self.canvas.overlay_panned.connect(self.overlay_panned)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def section(self) -> Section | None:
        return self._section

    @property
    def raw_image(self) -> np.ndarray | None:
        return self._raw_image

    @property
    def atlas(self) -> AtlasVolume | None:
        return self._atlas

    @property
    def channels(self) -> list:
        return self._channels

    @property
    def outline_mode(self) -> bool:
        return self._outline_mode

    # ------------------------------------------------------------------
    # State mutators
    # ------------------------------------------------------------------

    def set_atlas(self, atlas: AtlasVolume | None) -> None:
        self._atlas = atlas
        self.atlas_changed.emit(atlas)
        self.update_overlay()

    def set_channels(self, channels: list) -> None:
        self._channels = list(channels)
        self._display_image()

    def set_outline_mode(self, value: bool) -> None:
        if value == self._outline_mode:
            return
        self._outline_mode = value
        for btn in self._outline_buttons:
            if btn.isChecked() != value:
                btn.blockSignals(True)
                btn.setChecked(value)
                btn.blockSignals(False)
        self.outline_mode_changed.emit(value)
        self.update_overlay()

    def load_section(self, section: Section | None) -> None:
        self._section = section
        self._raw_image = None
        self.canvas.clear()

        if section is None:
            self._region_bar.setText("")
            self._region_bar.setStyleSheet(_REGION_BAR_IDLE_QSS)
            for lbl in self._status_labels:
                lbl.setText("No section loaded")
            self.section_loaded.emit(None)
            return

        for lbl in self._status_labels:
            lbl.setText(os.path.basename(section.original_path))

        from PyQt6.QtWidgets import QMessageBox

        from verso.engine.io.image_io import ensure_working_copy
        try:
            self._raw_image = ensure_working_copy(section)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            self.section_loaded.emit(section)
            return

        self._display_image()
        self.update_overlay()
        self.section_loaded.emit(section)

    def refresh_display(self) -> None:
        self._display_image()
        self.update_overlay()

    # ------------------------------------------------------------------
    # Background pipeline
    # ------------------------------------------------------------------

    def _update_channel_layers(self, img: np.ndarray, flip_h: bool, flip_v: bool) -> None:
        from verso.engine.preprocessing import compute_channel_layer
        if img.ndim == 2:
            img = img[..., np.newaxis]
        n = min(img.shape[2], len(self._channels))
        image_key = (id(self._raw_image), flip_h, flip_v, n)
        if image_key != self._layer_image_key:
            self._layer_image_key = image_key
            self._channel_layers = [
                compute_channel_layer(img, i, self._channels[i]) for i in range(n)
            ]
            self._cached_channel_specs = [
                (self._channels[i].scale, tuple(self._channels[i].color)) for i in range(n)
            ]
            return
        for i in range(n):
            spec = self._channels[i]
            key = (spec.scale, tuple(spec.color))
            if key != self._cached_channel_specs[i]:
                self._channel_layers[i] = compute_channel_layer(img, i, spec)
                self._cached_channel_specs[i] = key

    def _display_image(self) -> None:
        if self._raw_image is None:
            return
        from verso.engine.preprocessing import composite_from_layers
        img = self._raw_image
        flip_h = bool(self._section and self._section.preprocessing.flip_horizontal)
        flip_v = bool(self._section and self._section.preprocessing.flip_vertical)
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        self._update_channel_layers(img, flip_h, flip_v)
        rgb = composite_from_layers(self._channel_layers, self._channels)
        self.canvas.set_background(np.ascontiguousarray(rgb))

    # ------------------------------------------------------------------
    # Atlas overlay pipeline
    # ------------------------------------------------------------------

    def update_overlay(self) -> None:
        if self._atlas is None or self._section is None or self._raw_image is None:
            self.canvas.set_overlay(None)
            return

        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            h, w = self._raw_image.shape[:2]
            anchoring = self._atlas.default_anchoring(aspect_ratio=w / h)
            self._section.alignment.anchoring = anchoring

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
            self.canvas.set_overlay(None)
            self.overlay_updated.emit(list(anchoring), w_bg, h_bg)
            return

        if self.overlay_post_processor is not None:
            try:
                rgba = self.overlay_post_processor(rgba)
            except Exception:
                pass

        self.canvas.set_overlay(rgba, display_w=w_bg, display_h=h_bg)
        self.overlay_updated.emit(list(anchoring), w_bg, h_bg)

    # ------------------------------------------------------------------
    # Mouse → region label
    # ------------------------------------------------------------------

    def _on_canvas_mouse_moved(self, x: float, y: float) -> None:
        self.mouse_position_changed.emit(x, y)

        if self._atlas is None or self._section is None or self._raw_image is None:
            return
        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            return
        h_bg, w_bg = self._raw_image.shape[:2]
        if x < 0 or y < 0 or x >= w_bg or y >= h_bg:
            self._region_bar.setText("")
            self._region_bar.setStyleSheet(_REGION_BAR_IDLE_QSS)
            return
        s, t = x / w_bg, y / h_bg
        if self.cursor_to_atlas_mapper is not None:
            try:
                s, t = self.cursor_to_atlas_mapper(s, t)
            except Exception:
                pass
        name, (r, g, b) = self._atlas.get_region_info(anchoring, s, t)
        # Darken the region colour slightly so white text stays legible
        br, bg, bb = int(r * 0.55), int(g * 0.55), int(b * 0.55)
        self._region_bar.setText(name)
        self._region_bar.setStyleSheet(
            f"background: rgb({br},{bg},{bb}); color: #fff; font-size: 20px;"
            " font-weight: bold; border-top: 1px solid #333;"
        )

    # ------------------------------------------------------------------
    # Toolbar helpers (so AlignView and WarpView share styling without
    # duplicating button setup, while still owning their own toolbars)
    # ------------------------------------------------------------------

    def make_outline_button(self) -> QPushButton:
        """Create an outline toggle wired to this panel.  Synced across views."""
        btn = QPushButton("Outline")
        btn.setCheckable(True)
        btn.setFixedHeight(28)
        btn.setToolTip("Show white region outlines instead of coloured fill")
        btn.setStyleSheet(_OUTLINE_BTN_QSS)
        btn.setChecked(self._outline_mode)
        btn.toggled.connect(self.set_outline_mode)
        self._outline_buttons.append(btn)
        return btn

    def make_status_label(self) -> QLabel:
        """Create a status label that tracks the loaded section's filename."""
        if self._section is None:
            text = "No section loaded"
        else:
            text = os.path.basename(self._section.original_path)
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888; font-size: 11px; padding-left: 8px;")
        self._status_labels.append(lbl)
        return lbl


__all__ = ["SectionCanvasPanel"]
