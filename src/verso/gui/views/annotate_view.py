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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from verso.gui.utils import require
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.channel_display import push_channel_display
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.engine.model.annotation import PointSeries
    from verso.engine.model.project import ChannelSpec, Section
    from verso.gui.state import AppState

# Diameter (screen px) of rendered annotation points.
_POINT_SIZE = 9
_LASSO_COLOR = (255, 90, 90)


class AnnotateView(QWidget):
    """Canvas view for placing and viewing annotations on a section."""

    # Editing intents handed to AnnotationController (original-resolution coords).
    point_added = pyqtSignal(float, float)
    points_lassoed = pyqtSignal(object)  # list[tuple[float, float]] polygon
    undo_requested = pyqtSignal()
    tool_changed = pyqtSignal(str)  # "add" | "remove" — echoed so the page syncs

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
        # Annotations to render (pushed by AnnotationController) and the active
        # one (drawn with a white ring); coordinates are original-resolution px.
        self._annotations: list[PointSeries] = []
        self._active_index: int = -1
        # Editing tool + in-progress lasso stroke (display coords).
        self._tool = "add"
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._status_label = QLabel("No section loaded")
        layout.addWidget(make_view_status_bar(self._status_label))

        self._canvas = ImageCanvas()
        self._canvas.set_interaction_mode("annotate")
        self._canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self._canvas.canvas_drag_started.connect(self._on_drag_started)
        self._canvas.canvas_dragged.connect(self._on_dragged)
        self._canvas.canvas_drag_ended.connect(self._on_drag_ended)
        layout.addWidget(self._canvas, stretch=1)

        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        shortcuts = [
            (Qt.Key.Key_A, lambda: self._select_tool("add")),
            (Qt.Key.Key_R, lambda: self._select_tool("remove")),
            (Qt.Key.Key_U, self.undo_requested.emit),
            (QKeySequence.StandardKey.Undo, self.undo_requested.emit),
        ]
        for key, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(self._gated(slot))

    def _gated(self, slot):
        """Wrap a shortcut slot so it only fires while this view is visible."""

        def wrapper() -> None:
            if self.isVisible():
                slot()

        return wrapper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def set_tool(self, tool: str) -> None:
        """Switch the editing tool ('add' or 'remove')."""
        self._tool = "remove" if tool == "remove" else "add"
        self._cancel_stroke()

    def _select_tool(self, tool: str) -> None:
        """Change tool from a shortcut and notify the page to sync its buttons."""
        tool = "remove" if tool == "remove" else "add"
        if tool == self._tool:
            return
        self.set_tool(tool)
        self.tool_changed.emit(tool)

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
        self._render_annotations()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._channels = list(channels)
        self._display_image()
        self._render_annotations()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing/channel changes."""
        self._display_image()
        self._render_annotations()

    def set_annotations(self, annotations: list[PointSeries], active_index: int = -1) -> None:
        """Set the annotations to render and which one is active (highlighted)."""
        self._annotations = list(annotations)
        self._active_index = active_index
        self._render_annotations()

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

    def _render_annotations(self) -> None:
        """Draw the current section's annotation points onto the canvas.

        Points are stored in original-resolution pixels keyed by image filename;
        here they are filtered to the current section, scaled to working
        resolution, and mirrored to match the displayed (possibly flipped) image.
        """
        section = self._section
        project = self._state.project
        if self._raw_image is None or section is None or project is None:
            self._canvas.clear_annotations()
            return

        filename = os.path.basename(section.original_path).lower()
        h, w = self._raw_image.shape[:2]
        scale = project.working_scale
        flip_h = section.preprocessing.flip_horizontal
        flip_v = section.preprocessing.flip_vertical

        layers = []
        for i, ann in enumerate(self._annotations):
            if not ann.visible:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for p in ann.points:
                if os.path.basename(p.image).lower() != filename:
                    continue
                x = p.x * scale
                y = p.y * scale
                if flip_h:
                    x = (w - 1) - x
                if flip_v:
                    y = (h - 1) - y
                xs.append(x)
                ys.append(y)
            layers.append(
                {
                    "xs": xs,
                    "ys": ys,
                    "color": ann.color,
                    "opacity": ann.opacity,
                    "size": _POINT_SIZE,
                    "active": i == self._active_index,
                }
            )
        self._canvas.set_annotations(layers)

    # ------------------------------------------------------------------
    # Editing interactions
    # ------------------------------------------------------------------

    def _editable(self) -> bool:
        """Editing is possible only with an image and an active annotation."""
        return (
            self._raw_image is not None
            and self._section is not None
            and self._state.project is not None
            and 0 <= self._active_index < len(self._annotations)
        )

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        if self._tool != "add" or not self._editable():
            return
        ox, oy = self._display_to_original(x, y)
        self.point_added.emit(ox, oy)

    def _on_drag_started(self, x: float, y: float) -> None:
        if self._tool != "remove" or not self._editable():
            return
        self._stroke_points = [(x, y)]
        self._stroke_active = True
        self._canvas.clear_stroke_preview()

    def _on_dragged(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        self._stroke_points.append((x, y))
        self._canvas.set_stroke_preview(self._stroke_points, color=_LASSO_COLOR)

    def _on_drag_ended(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        self._stroke_points.append((x, y))
        self._stroke_active = False
        self._canvas.clear_stroke_preview()
        if len(self._stroke_points) < 3:
            self._stroke_points = []
            return
        polygon = [self._display_to_original(px, py) for px, py in self._stroke_points]
        self._stroke_points = []
        self.points_lassoed.emit(polygon)

    def _cancel_stroke(self) -> None:
        self._stroke_points = []
        self._stroke_active = False
        self._canvas.clear_stroke_preview()

    def _display_to_original(self, x: float, y: float) -> tuple[float, float]:
        """Map a canvas (display) coordinate back to original-resolution pixels.

        Inverse of the render transform in :meth:`_render_annotations`: undo the
        section flips, then divide by ``working_scale``. The display coordinate is
        clamped to the working image so clicks just outside stay in-bounds.
        """
        h, w = self._raw_image.shape[:2]
        section = require(self._section)
        scale = require(self._state.project).working_scale
        dx = min(max(x, 0.0), float(w - 1))
        dy = min(max(y, 0.0), float(h - 1))
        if section.preprocessing.flip_horizontal:
            dx = (w - 1) - dx
        if section.preprocessing.flip_vertical:
            dy = (h - 1) - dy
        return dx / scale, dy / scale
