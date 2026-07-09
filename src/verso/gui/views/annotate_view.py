"""Annotate view — canvas for viewing/editing annotations on a section.

D2 scaffolding: this view mirrors :class:`~verso.gui.views.prep_view.PrepView`'s
background-image display path (its own :class:`ImageCanvas`, a filename strip, and
the shared channel-display pipeline) but carries no editing behaviour yet. The
annotation manager, overlay rendering, and point-editing tools are layered on in
later deliverables.

Unlike Prep/Align/Warp this is not a :class:`BaseCanvasView`: annotations are a
project-global resource with their own save model, so the per-section draft
plumbing in that base does not apply here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from verso.gui.utils import require
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.channel_display import push_channel_display
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.engine.model.project import ChannelSpec, Section
    from verso.gui.state import AppState


class AnnotateView(QWidget):
    """Canvas view for placing and viewing annotations on a section."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._section: Section | None = None
        self._raw_image: np.ndarray | None = None
        self._channels: list[ChannelSpec] = []
        # (planes_version, flip_h, flip_v, n) GPU-upload cache key — see
        # push_channel_display. planes_version is bumped on every raw (re)load.
        self._channel_planes_key: tuple | None = None
        self._planes_version: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._status_label = QLabel("No section loaded")
        layout.addWidget(make_view_status_bar(self._status_label))

        self._canvas = ImageCanvas()
        self._canvas.set_interaction_mode("view")
        layout.addWidget(self._canvas, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def load_section(self, section: Section | None) -> None:
        """Load a section's working-resolution image into the canvas."""
        self._section = section
        self._raw_image = None
        self._canvas.clear()
        if section is None:
            self._status_label.setText("No section loaded")
            return

        self._status_label.setText(os.path.basename(section.original_path))

        from PyQt6.QtWidgets import QMessageBox

        from verso.engine.io.image_io import ensure_working_copy

        try:
            self._raw_image = ensure_working_copy(
                section, require(self._state.project).working_scale
            )
            self._planes_version += 1
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        self._display_image()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._channels = list(channels)
        self._display_image()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing/channel changes."""
        self._display_image()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_image(self) -> None:
        self._channel_planes_key = push_channel_display(
            self._canvas,
            self._raw_image,
            self._section,
            self._channels,
            self._planes_version,
            self._channel_planes_key,
        )
