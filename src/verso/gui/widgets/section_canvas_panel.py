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
    QVBoxLayout,
    QWidget,
)

from verso.engine.io.image_io import WORKING_SCALE
from verso.engine.model.project import Section
from verso.engine.preprocessing import channel_lut
from verso.gui.widgets.canvas import ImageCanvas

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume



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
    overlay_mode_changed = pyqtSignal(str)         # "annotation" | "outline" | "reference"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Shared state
        self._section: Section | None = None
        self._raw_image: np.ndarray | None = None
        self._atlas: AtlasVolume | None = None
        self._channels: list = []
        # Project-wide working scale, pushed via set_working_scale. Only consulted
        # if a thumbnail must be regenerated from the original (see
        # ensure_working_copy); kept in sync with the active project.
        self._working_scale: float = WORKING_SCALE
        # (id(raw_image), flip_h, flip_v, n) — invalidated only by section /
        # flip / channel-count changes; brightness/colour edits don't touch it.
        self._channel_planes_key: tuple | None = None
        self._overlay_mode: str = "annotation"  # "annotation" | "outline" | "reference"
        self._outline_color: tuple[int, int, int] = (255, 255, 255)

        # Hooks set by the active view
        self.overlay_post_processor: Callable[[np.ndarray], np.ndarray] | None = None
        self.cursor_to_atlas_mapper: Callable[[float, float], tuple[float, float]] | None = None

        # Labels created via make_status_label — kept in sync here.
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
    def overlay_mode(self) -> str:
        return self._overlay_mode

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

    def set_working_scale(self, working_scale: float) -> None:
        """Set the project's working scale used for any thumbnail regeneration."""
        self._working_scale = working_scale

    def set_overlay_mode(self, mode: str) -> None:
        if mode == self._overlay_mode:
            return
        self._overlay_mode = mode
        self.overlay_mode_changed.emit(mode)
        self.update_overlay()

    def set_outline_color(self, color: tuple[int, int, int]) -> None:
        """Set the outline overlay line color and refresh the canvas."""
        self._outline_color = color
        if self._overlay_mode == "outline":
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
            self._raw_image = ensure_working_copy(section, self._working_scale)
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

    def _display_image(self) -> None:
        if self._raw_image is None:
            self.canvas.clear()
            self._channel_planes_key = None
            return
        img = self._raw_image
        if img.ndim == 2:
            img = img[..., np.newaxis]
        flip_h = bool(self._section and self._section.preprocessing.flip_horizontal)
        flip_v = bool(self._section and self._section.preprocessing.flip_vertical)
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        n = min(img.shape[2], len(self._channels))

        # Push raw planes only when section / flip / channel-count actually
        # changed; this is the only path that touches the GPU texture.
        planes_key = (id(self._raw_image), flip_h, flip_v, n)
        if planes_key != self._channel_planes_key:
            planes = [np.ascontiguousarray(img[:, :, i]) for i in range(n)]
            self.canvas.set_channel_planes(planes)
            self._channel_planes_key = planes_key

        # Apply per-channel LUT / visibility — drives the brightness slider.
        for i in range(n):
            spec = self._channels[i]
            if not getattr(spec, "visible", True) or float(spec.scale) <= 0:
                self.canvas.set_channel_visible(i, False)
            else:
                self.canvas.set_channel_lut(i, channel_lut(spec))

    # ------------------------------------------------------------------
    # Atlas overlay pipeline
    # ------------------------------------------------------------------

    def update_overlay(self) -> None:
        if self._atlas is None or self._section is None or self._raw_image is None:
            self.canvas.set_overlay(None)
            return

        anchoring = self._section.alignment.anchoring
        if not anchoring or all(v == 0.0 for v in anchoring):
            # Render-only fallback for a section with no interpolated/edited
            # plane yet.  Do NOT write this centered plane back to the section —
            # leaving the anchoring unset lets project-wide interpolation fill in
            # the correct guess (otherwise the section gets stuck mid-brain and
            # the AP-plot dot reads a stale position_mm).
            h, w = self._raw_image.shape[:2]
            anchoring = self._atlas.default_anchoring(aspect_ratio=w / h)

        h_bg, w_bg = self._raw_image.shape[:2]
        # Sample atlas at a capped resolution for speed; canvas stretches it to
        # fill the background exactly via setRect (no visual quality loss).
        ATLAS_MAX_SIDE = 512
        scale = min(1.0, ATLAS_MAX_SIDE / max(w_bg, h_bg))
        out_w = max(1, round(w_bg * scale))
        out_h = max(1, round(h_bg * scale))

        try:
            if self._overlay_mode == "outline":
                rgba = self._atlas.slice_outline(anchoring, out_w, out_h, self._outline_color)
            elif self._overlay_mode == "reference":
                rgba = self._atlas.slice_reference_rgba(anchoring, out_w, out_h)
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

    def make_status_label(self) -> QLabel:
        """Create a status label that tracks the loaded section's filename.

        Style is applied by ``view_chrome.make_view_status_bar`` so the bar
        looks identical across Prep / Align / Warp.
        """
        if self._section is None:
            text = "No section loaded"
        else:
            text = os.path.basename(self._section.original_path)
        lbl = QLabel(text)
        self._status_labels.append(lbl)
        return lbl


__all__ = ["SectionCanvasPanel"]
