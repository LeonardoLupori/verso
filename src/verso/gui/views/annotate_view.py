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
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from verso.engine.model.annotation import AreaAnnotation, PointSeries
from verso.engine.preprocessing import apply_brush_stroke, apply_freehand_stroke, mask_to_rgba
from verso.gui.utils import require
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.channel_display import push_channel_display
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.engine.model.annotation import Annotation
    from verso.engine.model.project import ChannelSpec, Section
    from verso.gui.state import AppState

# Diameter (screen px) of rendered annotation points.
_POINT_SIZE = 9
_LASSO_COLOR = (255, 90, 90)
_DRAW_COLOR = (80, 160, 255)
_ERASE_COLOR = (255, 90, 90)


class AnnotateView(QWidget):
    """Canvas view for placing and viewing annotations on a section."""

    # Point-editing intents handed to AnnotationController (original-res coords).
    point_added = pyqtSignal(float, float)
    points_lassoed = pyqtSignal(object)  # list[tuple[float, float]] polygon
    undo_requested = pyqtSignal()
    tool_changed = pyqtSignal(str)  # "add" | "remove" — echoed so the page syncs
    # Area-editing intents: the view paints the active area's mask in place and
    # brackets each stroke with these so the controller snapshots undo + dirties.
    area_edit_started = pyqtSignal()
    area_edit_committed = pyqtSignal()
    area_tool_changed = pyqtSignal(str)  # "brush" | "freehand"
    brush_size_changed = pyqtSignal(int)

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
        self._annotations: list[Annotation] = []
        self._active_index: int = -1
        # Point tool + in-progress lasso stroke (display coords).
        self._tool = "add"
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        # Area tool (brush/freehand) + in-progress mask stroke state.
        self._area_tool = "brush"
        self._brush_radius = 20
        self._area_stroke_points: list[tuple[float, float]] = []
        self._area_stroke_active = False
        self._area_stroke_erase = False
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
        self._canvas.alt_wheel_scrolled.connect(self._on_alt_wheel)
        layout.addWidget(self._canvas, stretch=1)

        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        shortcuts = [
            (Qt.Key.Key_A, lambda: self._select_tool("add")),
            (Qt.Key.Key_R, lambda: self._select_tool("remove")),
            (Qt.Key.Key_B, lambda: self._select_area_tool("brush")),
            (Qt.Key.Key_F, lambda: self._select_area_tool("freehand")),
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

    def set_area_tool(self, tool: str) -> None:
        """Switch the area editing tool ('brush' or 'freehand')."""
        self._area_tool = "freehand" if tool == "freehand" else "brush"
        self._cancel_area_stroke()
        self._update_brush_cursor()

    def _select_area_tool(self, tool: str) -> None:
        """Change area tool from a shortcut and sync the page (only for areas)."""
        if not isinstance(self._active_annotation(), AreaAnnotation):
            return
        tool = "freehand" if tool == "freehand" else "brush"
        if tool == self._area_tool:
            return
        self.set_area_tool(tool)
        self.area_tool_changed.emit(tool)

    def set_brush_size(self, size: int) -> None:
        self._brush_radius = max(int(size), 1)
        self._update_brush_cursor()

    def _on_alt_wheel(self, delta: int) -> None:
        if not (
            isinstance(self._active_annotation(), AreaAnnotation) and self._area_tool == "brush"
        ):
            return
        step = (delta // 120) * 5
        new_size = max(5, min(200, self._brush_radius + step))
        if new_size != self._brush_radius:
            self.set_brush_size(new_size)
            self.brush_size_changed.emit(new_size)

    def _update_brush_cursor(self) -> None:
        """Show the circular brush cursor only when brushing an active area."""
        area_brush = (
            isinstance(self._active_annotation(), AreaAnnotation) and self._area_tool == "brush"
        )
        self._canvas.set_brush_cursor(area_brush, self._brush_radius)

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
        self._update_brush_cursor()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._channels = list(channels)
        self._display_image()
        self._render_annotations()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing/channel changes."""
        self._display_image()
        self._render_annotations()

    def set_annotations(self, annotations: list[Annotation], active_index: int = -1) -> None:
        """Set the annotations to render and which one is active (highlighted)."""
        self._annotations = list(annotations)
        self._active_index = active_index
        self._render_annotations()
        self._update_brush_cursor()

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
        """Draw the current section's annotations onto the canvas.

        Point series are filtered to the current section, scaled to working
        resolution, and mirrored to match the displayed (possibly flipped) image.
        Area masks (working resolution, mirrored the same way) render as coloured
        overlays below the points.
        """
        section = self._section
        project = self._state.project
        if self._raw_image is None or section is None or project is None:
            self._canvas.clear_annotations()
            self._canvas.clear_area_masks()
            return

        filename = os.path.basename(section.original_path).lower()
        h, w = self._raw_image.shape[:2]
        scale = project.working_scale
        flip_h = section.preprocessing.flip_horizontal
        flip_v = section.preprocessing.flip_vertical

        point_layers = []
        area_layers = []
        for i, ann in enumerate(self._annotations):
            if not ann.visible:
                continue
            if isinstance(ann, PointSeries):
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
                point_layers.append(
                    {
                        "xs": xs,
                        "ys": ys,
                        "color": ann.color,
                        "opacity": ann.opacity,
                        "size": _POINT_SIZE,
                        "active": i == self._active_index,
                    }
                )
            elif isinstance(ann, AreaAnnotation):
                mask = self._section_mask(ann, filename)
                if mask is None or not mask.any():
                    continue
                disp = mask
                if flip_h:
                    disp = np.fliplr(disp)
                if flip_v:
                    disp = np.flipud(disp)
                rgba = mask_to_rgba(
                    np.ascontiguousarray(disp), negative=False, opacity=1.0, color=ann.color
                )
                area_layers.append({"rgba": rgba, "w": w, "h": h, "opacity": ann.opacity})

        self._canvas.set_annotations(point_layers)
        self._canvas.set_area_masks(area_layers)

    @staticmethod
    def _section_mask(area: AreaAnnotation, filename_lower: str) -> np.ndarray | None:
        """The area's mask for a section (matched by filename, case-insensitive)."""
        for key, mask in area.masks.items():
            if os.path.basename(key).lower() == filename_lower:
                return mask
        return None

    # ------------------------------------------------------------------
    # Editing interactions
    # ------------------------------------------------------------------

    def _active_annotation(self) -> Annotation | None:
        if 0 <= self._active_index < len(self._annotations):
            return self._annotations[self._active_index]
        return None

    # -- dispatch by active annotation type ----------------------------

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        active = self._active_annotation()
        if isinstance(active, PointSeries):
            self._point_click(x, y)
        elif isinstance(active, AreaAnnotation):
            self._area_click(x, y)

    def _on_drag_started(self, x: float, y: float) -> None:
        active = self._active_annotation()
        if isinstance(active, PointSeries):
            self._point_drag_started(x, y)
        elif isinstance(active, AreaAnnotation):
            self._area_drag_started(x, y)

    def _on_dragged(self, x: float, y: float) -> None:
        if self._stroke_active:
            self._point_dragged(x, y)
        elif self._area_stroke_active:
            self._area_dragged(x, y)

    def _on_drag_ended(self, x: float, y: float) -> None:
        if self._stroke_active:
            self._point_drag_ended(x, y)
        elif self._area_stroke_active:
            self._area_drag_ended(x, y)

    # -- point series (click to add, lasso to remove) ------------------

    def _point_editable(self) -> bool:
        return (
            self._raw_image is not None
            and self._section is not None
            and self._state.project is not None
            and isinstance(self._active_annotation(), PointSeries)
        )

    def _point_click(self, x: float, y: float) -> None:
        if self._tool != "add" or not self._point_editable():
            return
        ox, oy = self._display_to_original(x, y)
        self.point_added.emit(ox, oy)

    def _point_drag_started(self, x: float, y: float) -> None:
        if self._tool != "remove" or not self._point_editable():
            return
        self._stroke_points = [(x, y)]
        self._stroke_active = True
        self._canvas.clear_stroke_preview()

    def _point_dragged(self, x: float, y: float) -> None:
        self._stroke_points.append((x, y))
        self._canvas.set_stroke_preview(self._stroke_points, color=_LASSO_COLOR)

    def _point_drag_ended(self, x: float, y: float) -> None:
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

    # -- area (brush / freehand mask painting) -------------------------

    def _area_editable(self) -> bool:
        return (
            self._raw_image is not None
            and self._section is not None
            and self._state.project is not None
            and isinstance(self._active_annotation(), AreaAnnotation)
        )

    def _area_click(self, x: float, y: float) -> None:
        # A plain click in brush mode stamps a single dab (freehand needs a drag).
        if self._area_tool != "brush" or not self._area_editable():
            return
        self._area_stroke_erase = self._shift_held()
        self.area_edit_started.emit()
        self._paint_area_brush([(x, y)])
        self.area_edit_committed.emit()

    def _area_drag_started(self, x: float, y: float) -> None:
        if not self._area_editable():
            return
        self._area_stroke_erase = self._shift_held()
        self._area_stroke_points = [(x, y)]
        self._area_stroke_active = True
        self._canvas.clear_stroke_preview()
        if self._area_tool == "brush":
            self.area_edit_started.emit()
            self._paint_area_brush([(x, y)])

    def _area_dragged(self, x: float, y: float) -> None:
        point = (x, y)
        if self._area_tool == "brush":
            prev = self._area_stroke_points[-1]
            self._area_stroke_points.append(point)
            self._paint_area_brush([prev, point])
        else:
            self._area_stroke_points.append(point)
            color = _ERASE_COLOR if self._area_stroke_erase else _DRAW_COLOR
            self._canvas.set_stroke_preview(self._area_stroke_points, color=color)

    def _area_drag_ended(self, x: float, y: float) -> None:
        point = (x, y)
        self._area_stroke_active = False
        self._canvas.clear_stroke_preview()
        if self._area_tool == "brush":
            prev = self._area_stroke_points[-1] if self._area_stroke_points else point
            self._paint_area_brush([prev, point])
            self._area_stroke_points = []
            self.area_edit_committed.emit()
            return
        # Freehand: fill the polygon on release.
        self._area_stroke_points.append(point)
        if len(self._area_stroke_points) < 3:
            self._area_stroke_points = []
            return
        self.area_edit_started.emit()
        self._fill_area_freehand(self._area_stroke_points)
        self._area_stroke_points = []
        self.area_edit_committed.emit()

    def _cancel_area_stroke(self) -> None:
        self._area_stroke_points = []
        self._area_stroke_active = False
        self._canvas.clear_stroke_preview()

    def _paint_area_brush(self, display_points: list[tuple[float, float]]) -> None:
        mask = self._active_area_buffer()
        if mask is None:
            return
        pts = self._stroke_to_mask_coords(display_points)
        new = apply_brush_stroke(
            mask, pts, radius=self._brush_radius, add=not self._area_stroke_erase
        )
        self._store_area_mask(new)
        self._render_annotations()

    def _fill_area_freehand(self, display_points: list[tuple[float, float]]) -> None:
        mask = self._active_area_buffer()
        if mask is None:
            return
        pts = self._stroke_to_mask_coords(display_points)
        new = apply_freehand_stroke(mask, pts, add=not self._area_stroke_erase)
        self._store_area_mask(new)
        self._render_annotations()

    def _active_area_buffer(self) -> np.ndarray | None:
        """The active area's mask for the current section, created empty if absent."""
        area = self._active_annotation()
        if not isinstance(area, AreaAnnotation) or self._raw_image is None or self._section is None:
            return None
        shape = self._raw_image.shape[:2]
        key = os.path.basename(self._section.original_path)
        mask = area.masks.get(key)
        if mask is None or mask.shape != shape:
            return np.zeros(shape, dtype=bool)
        return mask

    def _store_area_mask(self, mask: np.ndarray) -> None:
        area = self._active_annotation()
        if isinstance(area, AreaAnnotation) and self._section is not None:
            area.masks[os.path.basename(self._section.original_path)] = mask

    def _stroke_to_mask_coords(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Display coords → working-res mask coords (clamp, then undo flips)."""
        h, w = self._raw_image.shape[:2]
        pts = np.asarray(points, dtype=float)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        section = require(self._section)
        if section.preprocessing.flip_horizontal:
            pts[:, 0] = float(w - 1) - pts[:, 0]
        if section.preprocessing.flip_vertical:
            pts[:, 1] = float(h - 1) - pts[:, 1]
        return pts

    @staticmethod
    def _shift_held() -> bool:
        return bool(QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

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
