"""Prep view - canvas for section preprocessing (masks, flipping)."""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from verso.engine.drafts import commit_prep_draft
from verso.engine.model.project import ChannelSpec, Preprocessing, Section
from verso.engine.preprocessing import (
    apply_brush_stroke,
    apply_freehand_stroke,
    detect_foreground,
    load_mask,
    mask_to_rgba,
    morph_mask,
)
from verso.gui.utils import require
from verso.gui.views.base_canvas_view import BaseCanvasView
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.channel_display import push_channel_display
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.gui.state import AppState


class PrepView(BaseCanvasView):
    """Canvas view for the Prep (mask drawing / flip) step.

    The step's two sub-edits map onto the shared draft model like this: the
    in-progress slice **mask** is parked in the DraftStore's ``"prep"`` working
    payload (so it survives navigation and drives the Overview mask dot), while
    the **flips** live on ``section.preprocessing`` and are compared against the
    last-saved baseline.  ``_saved_mask`` holds the last-saved mask in memory so
    undo can tell when an edit has returned to the saved state.
    """

    STEP = "prep"

    mask_negative_changed = pyqtSignal(bool)
    mask_visibility_changed = pyqtSignal(bool)
    brush_size_changed = pyqtSignal(int)
    draw_mode_changed = pyqtSignal(str)  # "freehand" | "brush"
    # Emitted when a prep clear/reset removes a saved flip and thereby
    # invalidates the section's alignment + warp, so MainWindow can clear their
    # dirty flags + refresh.
    alignment_invalidated = pyqtSignal()

    _DRAW_COLOR = (80, 160, 255)
    _ERASE_COLOR = (255, 90, 90)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._section: Section | None = None
        self._raw_image: np.ndarray | None = None
        # ``_current_mask`` is the live edit / render buffer; ``_saved_mask`` is
        # the last-saved mask, kept in memory for undo's clean-state check.
        self._current_mask: np.ndarray | None = None
        self._saved_mask: np.ndarray | None = None
        self._mask_opacity = 0.4
        self._mask_color = (255, 255, 255)
        self._negative_mask = False
        self._mask_visible = True
        self._channels: list[ChannelSpec] = []
        # (planes_version, flip_h, flip_v, n) — tracks whether we still need to
        # re-push the per-channel uint8 planes to the canvas.  Brightness /
        # colour / visibility changes never invalidate this key.  planes_version
        # is bumped on every raw-image (re)load; it replaces id(raw_image),
        # which CPython reuses across freed objects and could collide with the
        # previous key, skipping the GPU texture update.
        self._channel_planes_key: tuple | None = None
        self._planes_version: int = 0
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        # Latched at stroke start from the Shift modifier — releasing Shift
        # mid-stroke does not flip add/erase, matching the cursor color the
        # user committed to when they began the drag.
        self._stroke_erase = False
        # Slice-mask draw mode: "freehand" (polygon fill on release) or
        # "brush" (live disk painting). Brush radius is in mask pixels.
        self._draw_mode = "freehand"
        self._brush_radius = 20
        # Set by _wipe() when a saved flip is removed, so _after_clear() knows to
        # emit alignment_invalidated after the clear has settled.
        self._clear_invalidated_alignment = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Central canvas with filename strip
        canvas_col = QVBoxLayout()
        canvas_col.setContentsMargins(0, 0, 0, 0)
        canvas_col.setSpacing(0)

        self._status_label = QLabel("No section loaded")
        canvas_col.addWidget(make_view_status_bar(self._status_label))

        self._canvas = ImageCanvas()
        self._canvas.set_interaction_mode("prep")
        self._canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self._canvas.canvas_drag_started.connect(self._on_canvas_drag_started)
        self._canvas.canvas_dragged.connect(self._on_canvas_dragged)
        self._canvas.canvas_drag_ended.connect(self._on_canvas_drag_ended)
        self._canvas.alt_wheel_scrolled.connect(self._on_alt_wheel)
        canvas_col.addWidget(self._canvas, stretch=1)
        layout.addLayout(canvas_col, stretch=1)

        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        # WindowShortcut so M/N/U work regardless of which widget in the
        # window holds focus (canvas, properties dock, etc).  The slots
        # no-op when prep view isn't visible.
        shortcuts = [
            (Qt.Key.Key_M, lambda: self.set_mask_visible(not self._mask_visible)),
            (Qt.Key.Key_N, lambda: self.set_mask_negative(not self._negative_mask)),
            (Qt.Key.Key_U, self.undo),
            (QKeySequence.StandardKey.Undo, self.undo),
            (Qt.Key.Key_B, lambda: self._select_draw_mode("brush")),
            (Qt.Key.Key_F, lambda: self._select_draw_mode("freehand")),
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

    def load_section(self, section: Section | None) -> None:
        self._section = section
        self._raw_image = None
        self._current_mask = None
        self._saved_mask = None
        self._undo_stack.clear()
        self._stroke_points.clear()
        self._stroke_active = False
        self._canvas.clear()
        if section is None:
            self._status_label.setText("No section loaded")
            return

        import os

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

        # The saved mask comes from disk; a resident working payload (an unsaved
        # edit carried across navigation) overrides it for display.
        self._saved_mask = self._load_saved_mask()
        working = self._state.get_working(section.id, "prep")
        self._current_mask = working if working is not None else self._saved_mask
        # Sync the last-saved Preprocessing baseline into AppState (a no-op while
        # dirty, so a stash carried across navigation survives — the section's
        # flips may already hold the unsaved edit).  The window refreshes the
        # save bar for the new section.
        self._state.sync_baseline(section.id, "prep", copy.deepcopy(section.preprocessing))
        self._display_image()
        self._update_mask_overlay()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing parameter changes."""
        self._display_image()
        self._update_mask_overlay()

    def set_mask_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._mask_visible == visible:
            return
        self._mask_visible = visible
        self._update_mask_overlay()
        self.mask_visibility_changed.emit(visible)

    def set_mask_negative(self, negative: bool) -> None:
        negative = bool(negative)
        if self._negative_mask == negative:
            return
        self._negative_mask = negative
        self._update_mask_overlay()
        self.mask_negative_changed.emit(negative)

    def set_mask_opacity(self, opacity: float) -> None:
        self._mask_opacity = min(max(opacity, 0.0), 1.0)
        self._canvas.set_overlay_opacity(self._mask_opacity)

    def set_mask_color(self, color: tuple[int, int, int]) -> None:
        self._mask_color = color
        self._update_mask_overlay()

    def set_draw_mode(self, mode: str) -> None:
        self._draw_mode = "brush" if mode == "brush" else "freehand"
        self._canvas.set_brush_cursor(self._draw_mode == "brush", self._brush_radius)

    def _select_draw_mode(self, mode: str) -> None:
        """Switch draw mode from a shortcut and notify the mask panel to sync."""
        mode = "brush" if mode == "brush" else "freehand"
        if mode == self._draw_mode:
            return
        self.set_draw_mode(mode)
        self.draw_mode_changed.emit(mode)

    def set_brush_size(self, size: int) -> None:
        self._brush_radius = max(int(size), 1)
        self._canvas.set_brush_cursor(self._draw_mode == "brush", self._brush_radius)

    def _on_alt_wheel(self, delta: int) -> None:
        if self._draw_mode != "brush":
            return
        step = (delta // 120) * 5
        new_size = max(5, min(200, self._brush_radius + step))
        self.set_brush_size(new_size)
        self.brush_size_changed.emit(new_size)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._channels = list(channels)
        self._display_image()

    def autodetect_mask(self) -> None:
        if self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        visible = [
            i
            for i, spec in enumerate(self._channels[: self._raw_image.shape[2]])
            if getattr(spec, "visible", True) and float(spec.scale) > 0
        ]
        img = self._raw_image[:, :, visible] if visible else self._raw_image
        self._current_mask = detect_foreground(img)
        self._mark_mask_edited()
        self._update_mask_overlay()

    def clear_mask(self) -> None:
        if self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = np.zeros(self._raw_image.shape[:2], dtype=bool)
        self._mark_mask_edited()
        self._update_mask_overlay()

    def apply_morph(self, pixels: int, operation: str) -> None:
        if self._current_mask is None or self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = morph_mask(self._current_mask, pixels, operation)
        self._mark_mask_edited()
        self._update_mask_overlay()

    def mark_flip_changed(self) -> None:
        """Called by MainWindow after toggling a flip flag on the section."""
        if self._section is None:
            return
        self._set_dirty(True)

    def has_persisted_state(self) -> bool:
        """Whether Clear has anything to wipe in the project for this slice."""
        if self._section is None:
            return False
        # The last-saved baseline reflects the persisted mask path + flips even
        # mid-edit; fall back to the section itself before a baseline is stashed.
        baseline = self._saved_state()
        pp = baseline if baseline is not None else self._section.preprocessing
        return bool(pp.slice_mask_path) or bool(pp.flip_horizontal) or bool(pp.flip_vertical)

    # ------------------------------------------------------------------
    # Draft-view hooks (see BaseCanvasView)
    # ------------------------------------------------------------------

    def _current_section(self) -> Section | None:
        return self._section

    def _capture_edit(self) -> np.ndarray:
        # Callers all run _ensure_mask() first, so the mask is never None here.
        return require(self._current_mask).copy()

    def _restore(self, snapshot: np.ndarray) -> None:
        self._current_mask = snapshot
        self._update_mask_overlay()

    def _matches_saved(self, snapshot: np.ndarray) -> bool:
        if self._flip_is_dirty():
            return False
        return self._saved_mask is not None and np.array_equal(snapshot, self._saved_mask)

    def _saved_copy(self) -> Preprocessing:
        return copy.deepcopy(require(self._section).preprocessing)

    def _apply_saved(self, baseline: Preprocessing) -> None:
        section = require(self._section)
        section.preprocessing = copy.deepcopy(baseline)
        self._state.pop_working(section.id, "prep")
        if self._raw_image is not None:
            self._saved_mask = self._load_saved_mask()
            self._current_mask = self._saved_mask
            self._display_image()
            self._update_mask_overlay()

    def _commit(self) -> bool:
        section = require(self._section)
        mask = self._state.get_working(section.id, "prep")
        changed = mask is not None or self._flip_is_dirty()
        commit_prep_draft(section, mask)
        self._state.pop_working(section.id, "prep")
        if mask is not None:
            self._saved_mask = mask
        return changed

    def _wipe(self) -> None:
        """Wipe this slice's prep state: mask + flips.

        Deletes the on-disk PNG and resets preprocessing to defaults.  If a
        previously-*saved* flip is thereby removed, the slice's alignment + warp
        are invalidated (emitted from _after_clear once the wipe has settled).
        """
        section = require(self._section)
        baseline = self._saved_state()
        self._clear_invalidated_alignment = bool(
            baseline is not None and (baseline.flip_horizontal or baseline.flip_vertical)
        )

        path_str = section.preprocessing.slice_mask_path
        if path_str:
            with contextlib.suppress(OSError):
                Path(path_str).unlink(missing_ok=True)

        section.preprocessing = Preprocessing()
        self._state.pop_working(section.id, "prep")
        if self._clear_invalidated_alignment:
            self._wipe_alignment_for_flip()

        if self._raw_image is not None:
            self._saved_mask = self._load_saved_mask()
            self._current_mask = self._saved_mask
            self._display_image()
            self._update_mask_overlay()

    def _end_edit_gesture(self) -> None:
        self._stroke_points.clear()
        self._stroke_active = False

    def _after_undo_restore(self) -> None:
        self._sync_working()

    def _after_clear(self) -> None:
        if self._clear_invalidated_alignment:
            self.alignment_invalidated.emit()

    # ------------------------------------------------------------------
    # Prep-specific draft helpers
    # ------------------------------------------------------------------

    def _mark_mask_edited(self) -> None:
        """Park the current mask as the unsaved working payload + mark dirty."""
        if self._section is None:
            return
        self._state.set_working(self._section.id, "prep", self._current_mask)
        self._set_dirty(True)

    def _sync_working(self) -> None:
        """Set or drop the working mask so it matches the current edit.

        Used after undo: if the mask is back at the saved state the payload is
        dropped (mask clean); otherwise the current mask is (re)parked.
        """
        if self._section is None:
            return
        if self._current_mask is not None and not self._mask_matches_saved():
            self._state.set_working(self._section.id, "prep", self._current_mask)
        else:
            self._state.pop_working(self._section.id, "prep")

    def _mask_matches_saved(self) -> bool:
        return self._saved_mask is not None and np.array_equal(self._current_mask, self._saved_mask)

    def _flip_is_dirty(self) -> bool:
        if self._section is None:
            return False
        baseline = self._saved_state()
        if baseline is None:
            return False
        pp = self._section.preprocessing
        return (
            baseline.flip_horizontal != pp.flip_horizontal
            or baseline.flip_vertical != pp.flip_vertical
        )

    def _load_saved_mask(self) -> np.ndarray | None:
        """Load the last-saved slice mask from disk (or zeros if none)."""
        if self._section is None or self._raw_image is None:
            return None
        shape = self._raw_image.shape[:2]
        path = self._section.preprocessing.slice_mask_path
        if path and Path(path).exists():
            try:
                return load_mask(path, shape)
            except Exception:
                pass
        return np.zeros(shape, dtype=bool)

    def _wipe_alignment_for_flip(self) -> None:
        if self._section is None:
            return
        from verso.engine.drafts import reset_alignment

        reset_alignment(self._section)

    # ------------------------------------------------------------------
    # Display / mask state
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

    def _ensure_mask(self) -> None:
        if self._current_mask is None and self._raw_image is not None:
            self._current_mask = np.zeros(self._raw_image.shape[:2], dtype=bool)

    def _update_mask_overlay(self) -> None:
        if self._current_mask is None or not self._mask_visible:
            self._canvas.set_overlay(None)
            return

        display_mask = self._mask_for_display()
        rgba = mask_to_rgba(
            display_mask,
            negative=self._negative_mask,
            opacity=1.0,
            color=self._mask_color,
        )
        h, w = display_mask.shape
        self._canvas.set_overlay(rgba, display_w=w, display_h=h)
        self._canvas.set_overlay_opacity(self._mask_opacity)

    def _mask_for_display(self) -> np.ndarray:
        if self._current_mask is None:
            raise RuntimeError("No mask loaded")

        mask = self._current_mask
        if self._section and self._section.preprocessing.flip_horizontal:
            mask = np.fliplr(mask)
        if self._section and self._section.preprocessing.flip_vertical:
            mask = np.flipud(mask)
        return np.ascontiguousarray(mask)

    # ------------------------------------------------------------------
    # Tool / stroke handling
    # ------------------------------------------------------------------

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        if self._raw_image is None or self._section is None:
            return
        if self._draw_mode != "brush":
            return
        mods = QGuiApplication.keyboardModifiers()
        self._stroke_erase = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        point = self._clamped_display_point(x, y)
        self._ensure_mask()
        self._push_undo()
        self._paint_brush_segment([point])

    def _on_canvas_drag_started(self, x: float, y: float) -> None:
        if self._raw_image is None or self._section is None:
            return
        mods = QGuiApplication.keyboardModifiers()
        self._stroke_erase = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        point = self._clamped_display_point(x, y)
        self._stroke_points = [point]
        self._stroke_active = True
        self._canvas.clear_stroke_preview()
        if self._draw_mode == "brush":
            # Brush paints live; snapshot once for undo and stamp the first dab.
            self._ensure_mask()
            self._push_undo()
            self._paint_brush_segment([point])

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        point = self._clamped_display_point(x, y)
        if self._draw_mode == "brush":
            # Paint the new segment (previous point → this one) into the mask.
            prev = self._stroke_points[-1]
            self._stroke_points.append(point)
            self._paint_brush_segment([prev, point])
            return
        self._stroke_points.append(point)
        color = self._ERASE_COLOR if self._stroke_erase else self._DRAW_COLOR
        self._canvas.set_stroke_preview(self._stroke_points, color=color)

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        point = self._clamped_display_point(x, y)
        self._stroke_active = False
        self._canvas.clear_stroke_preview()

        if self._draw_mode == "brush":
            # Mask already updated incrementally; just paint the final segment.
            prev = self._stroke_points[-1] if self._stroke_points else point
            self._paint_brush_segment([prev, point])
            self._stroke_points.clear()
            return

        self._stroke_points.append(point)
        if len(self._stroke_points) < 3:
            self._stroke_points.clear()
            return

        self._ensure_mask()
        self._push_undo()
        pts = self._stroke_points_to_mask_coords(self._stroke_points)
        self._current_mask = apply_freehand_stroke(
            require(self._current_mask),
            pts,
            add=not self._stroke_erase,
        )
        self._stroke_points.clear()
        self._mark_mask_edited()
        self._update_mask_overlay()

    def _paint_brush_segment(self, display_points: list[tuple[float, float]]) -> None:
        """Stamp the brush along ``display_points`` into the live mask."""
        pts = self._stroke_points_to_mask_coords(display_points)
        self._current_mask = apply_brush_stroke(
            require(self._current_mask),
            pts,
            radius=self._brush_radius,
            add=not self._stroke_erase,
        )
        self._mark_mask_edited()
        self._update_mask_overlay()

    def _clamped_display_point(self, x: float, y: float) -> tuple[float, float]:
        if self._raw_image is None:
            return x, y
        h, w = self._raw_image.shape[:2]
        return (
            min(max(x, 0.0), float(w - 1)),
            min(max(y, 0.0), float(h - 1)),
        )

    def _stroke_points_to_mask_coords(
        self,
        points: list[tuple[float, float]],
    ) -> np.ndarray:
        if self._raw_image is None:
            return np.asarray(points, dtype=float)

        h, w = self._raw_image.shape[:2]
        pts = np.asarray(points, dtype=float)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        if self._section and self._section.preprocessing.flip_horizontal:
            pts[:, 0] = float(w - 1) - pts[:, 0]
        if self._section and self._section.preprocessing.flip_vertical:
            pts[:, 1] = float(h - 1) - pts[:, 1]
        return pts
