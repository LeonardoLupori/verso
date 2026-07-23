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
import weakref
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from verso.engine.annotations import point_coords_by_image
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
        # Per-series cache of points grouped by image basename (original-res
        # coords). A series can hold 100k+ points across sections; grouping once
        # keeps each render proportional to the *current* section's points rather
        # than rescanning the whole series. Keyed by id(series); each entry holds
        # (series weakref, points-list ref, count, buckets). While the same points
        # list object grows it can only have gained appends at the end, so those
        # are folded in incrementally; a new list object or a shrink rebuilds. The
        # weakref drops deleted series and guards against id reuse.
        self._point_bucket_cache: dict[
            int,
            tuple[weakref.ref, list, int, dict[str, tuple[np.ndarray, np.ndarray]]],
        ] = {}
        # Point tool + in-progress lasso stroke (display coords).
        self._tool = "add"
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        # Press position of an add-mode gesture Qt routed as a drag. In add mode
        # drags carry no gesture of their own, so any such "drag" is treated as a
        # click here and lands a point at this position on release.
        self._add_drag_start: tuple[float, float] | None = None
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
        """Draw the current section's point and area annotations onto the canvas."""
        ctx = self._render_context()
        if ctx is None:
            self._canvas.clear_annotations()
            self._canvas.clear_area_masks()
            return
        self._canvas.set_annotations(self._build_point_layers(ctx))
        self._canvas.set_area_masks(self._build_area_layers(ctx))

    def _render_area_masks(self) -> None:
        """Re-render only the area overlays (used mid-stroke while brushing).

        Point layers cannot change during an area stroke, so skipping them avoids
        re-pushing a potentially huge scatter to the GPU on every brush tick.
        """
        ctx = self._render_context()
        if ctx is None:
            self._canvas.clear_area_masks()
            return
        self._canvas.set_area_masks(self._build_area_layers(ctx))

    def _render_context(self) -> tuple | None:
        """Shared per-render geometry, or ``None`` when there is nothing to draw."""
        section = self._section
        project = self._state.project
        if self._raw_image is None or section is None or project is None:
            return None
        filename = section.image_key.lower()
        h, w = self._raw_image.shape[:2]
        return (
            filename,
            w,
            h,
            project.working_scale,
            section.preprocessing.flip_horizontal,
            section.preprocessing.flip_vertical,
        )

    def _build_point_layers(self, ctx: tuple) -> list:
        """Point layers for the current section, scaled + mirrored to display.

        Uses the cached per-image coordinate buckets so only the current
        section's points are transformed (a vectorised scale + flip), never the
        whole series.
        """
        filename, w, h, scale, flip_h, flip_v = ctx
        layers = []
        for ann in self._annotations:
            if not ann.visible or not isinstance(ann, PointSeries):
                continue
            coords = self._series_buckets(ann).get(filename)
            if coords is None:
                xs = ys = np.empty(0, dtype=np.float64)
            else:
                xo, yo = coords
                xs = (w - 1) - xo * scale if flip_h else xo * scale
                ys = (h - 1) - yo * scale if flip_v else yo * scale
            layers.append(
                {
                    "xs": xs,
                    "ys": ys,
                    "color": ann.color,
                    "size": ann.point_size,
                }
            )
        return layers

    def _build_area_layers(self, ctx: tuple) -> list:
        """Area-mask overlays for the current section, mirrored to display."""
        filename, w, h, _scale, flip_h, flip_v = ctx
        layers = []
        for ann in self._annotations:
            if not ann.visible or not isinstance(ann, AreaAnnotation):
                continue
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
            layers.append({"rgba": rgba, "w": w, "h": h, "opacity": ann.opacity})
        return layers

    def _series_buckets(self, series: PointSeries) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Cached ``{image: (xs, ys)}`` for a series, updated incrementally on add.

        Points are only ever appended in place (``add_point``); removal and undo
        reassign ``series.points`` to a new list. So while the cached entry holds
        the *same* list object, any growth is new points at the tail: those are
        folded into the buckets instead of rescanning the whole (100k+ point)
        series. A changed list object, or a shrink, triggers a full rebuild. The
        weakref guards against ``id`` reuse after a series is deleted.
        """
        key = id(series)
        points = series.points
        n = len(points)
        cached = self._point_bucket_cache.get(key)
        if cached is not None and cached[0]() is series and cached[1] is points and cached[2] <= n:
            buckets = cached[3]
            if cached[2] < n:
                self._append_to_buckets(buckets, points[cached[2] : n])
                self._point_bucket_cache[key] = (cached[0], points, n, buckets)
            return buckets
        buckets = point_coords_by_image(series)
        self._point_bucket_cache[key] = (
            weakref.ref(series, lambda _ref, k=key: self._point_bucket_cache.pop(k, None)),
            points,
            n,
            buckets,
        )
        return buckets

    @staticmethod
    def _append_to_buckets(
        buckets: dict[str, tuple[np.ndarray, np.ndarray]],
        new_points: list,
    ) -> None:
        """Fold appended points into existing image buckets (see ``_series_buckets``).

        Mirrors :func:`point_coords_by_image`'s basename-lower keying so the
        incremental result is identical to a full rebuild.
        """
        by_key: dict[str, tuple[list[float], list[float]]] = {}
        for p in new_points:
            k = os.path.basename(p.image).lower()
            xs, ys = by_key.setdefault(k, ([], []))
            xs.append(p.x)
            ys.append(p.y)
        for k, (xs, ys) in by_key.items():
            nx = np.asarray(xs, dtype=np.float64)
            ny = np.asarray(ys, dtype=np.float64)
            if k in buckets:
                px, py = buckets[k]
                buckets[k] = (np.concatenate((px, nx)), np.concatenate((py, ny)))
            else:
                buckets[k] = (nx, ny)

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
        elif self._add_drag_start is not None:
            self._point_add_drag_ended()

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
        if not self._point_editable():
            return
        if self._tool == "add":
            # Add mode has no drag gesture: Qt only routes a click here as a drag
            # when the press jittered a few pixels. Remember where it began so the
            # release still lands a point (see _point_add_drag_ended).
            self._add_drag_start = (x, y)
            return
        self._stroke_points = [(x, y)]
        self._stroke_active = True
        self._canvas.clear_stroke_preview()

    def _point_add_drag_ended(self) -> None:
        """Land a point for an add-mode gesture Qt classified as a drag.

        In add mode a drag carries no gesture of its own, so it is treated exactly
        like a click at the press position — no movement threshold to tune.
        """
        start = self._add_drag_start
        self._add_drag_start = None
        if start is not None and self._tool == "add":
            self._point_click(*start)

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
        self._add_drag_start = None
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
        self._render_area_masks()

    def _fill_area_freehand(self, display_points: list[tuple[float, float]]) -> None:
        mask = self._active_area_buffer()
        if mask is None:
            return
        pts = self._stroke_to_mask_coords(display_points)
        new = apply_freehand_stroke(mask, pts, add=not self._area_stroke_erase)
        self._store_area_mask(new)
        self._render_area_masks()

    def _active_area_buffer(self) -> np.ndarray | None:
        """The active area's mask for the current section, created empty if absent."""
        area = self._active_annotation()
        if not isinstance(area, AreaAnnotation) or self._raw_image is None or self._section is None:
            return None
        shape = self._raw_image.shape[:2]
        key = self._section.image_key
        mask = area.masks.get(key)
        if mask is None or mask.shape != shape:
            return np.zeros(shape, dtype=bool)
        return mask

    def _store_area_mask(self, mask: np.ndarray) -> None:
        area = self._active_annotation()
        if isinstance(area, AreaAnnotation) and self._section is not None:
            area.masks[self._section.image_key] = mask

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
