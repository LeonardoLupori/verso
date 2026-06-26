"""Prep view - canvas for section preprocessing (masks, flipping)."""

from __future__ import annotations

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

from verso.engine.drafts import PrepDraft, persist_prep_draft
from verso.engine.model.project import ChannelSpec, Preprocessing, Section
from verso.engine.preprocessing import (
    apply_brush_stroke,
    apply_freehand_stroke,
    channel_lut,
    detect_foreground,
    load_mask,
    mask_to_rgba,
    morph_mask,
)
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.view_chrome import make_view_status_bar

if TYPE_CHECKING:
    from verso.gui.state import AppState


class PrepView(QWidget):
    """Canvas view for the Prep (mask drawing / flip) step."""

    dirty_changed = pyqtSignal(bool)
    mask_negative_changed = pyqtSignal(bool)
    mask_visibility_changed = pyqtSignal(bool)
    brush_size_changed = pyqtSignal(int)
    draw_mode_changed = pyqtSignal(str)  # "freehand" | "brush"
    # Emitted when a prep save/clear flips the section and thereby invalidates
    # its alignment + warp, so MainWindow can clear their dirty flags + refresh.
    alignment_invalidated = pyqtSignal()

    _UNDO_LIMIT = 20
    _DRAW_COLOR = (80, 160, 255)
    _ERASE_COLOR = (255, 90, 90)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._section: Section | None = None
        # Last-saved flip flags for the loaded section, carried across navigation
        # via the prep draft store so save() can detect a flip change even after
        # the baseline was re-snapshotted on reload.
        self._prep_base_flip: tuple[bool, bool] = (False, False)
        self._raw_image: np.ndarray | None = None
        self._current_mask: np.ndarray | None = None
        self._mask_dirty = False
        self._mask_opacity = 0.4
        self._mask_color = (255, 255, 255)
        self._negative_mask = False
        self._mask_visible = True
        self._channels: list[ChannelSpec] = []
        # (id(raw_image), flip_h, flip_v, n) — tracks whether we still need to
        # re-push the per-channel uint8 planes to the canvas.  Brightness /
        # colour / visibility changes never invalidate this key.
        self._channel_planes_key: tuple | None = None
        self._undo_stack: list[np.ndarray] = []
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
        # Set by flush_draft() when the mask arrays are released on navigation
        # away; tells refresh_display() to restore them on re-entry so the
        # overlay doesn't render empty when returning to the same section.
        self._arrays_released = False
        # Snapshot of section.preprocessing taken at load_section time; restored
        # by discard() so unsaved edits roll back when the user switches view
        # or slice.
        self._baseline_preprocessing: Preprocessing | None = None
        self._dirty = False
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
            (Qt.Key.Key_U, self.undo_mask_edit),
            (QKeySequence.StandardKey.Undo, self.undo_mask_edit),
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
        self._mask_dirty = False
        self._arrays_released = False
        self._undo_stack.clear()
        self._stroke_points.clear()
        self._stroke_active = False
        self._canvas.clear()
        if section is None:
            self._baseline_preprocessing = None
            self._prep_base_flip = (False, False)
            self._set_dirty(False)
            self._status_label.setText("No section loaded")
            return

        self._baseline_preprocessing = copy.deepcopy(section.preprocessing)
        # Default last-saved flips = current (clean) flips; a resident draft
        # overrides this below to carry the saved flips across navigation.
        self._prep_base_flip = (
            section.preprocessing.flip_horizontal,
            section.preprocessing.flip_vertical,
        )

        import os

        self._status_label.setText(os.path.basename(section.original_path))

        from PyQt6.QtWidgets import QMessageBox

        from verso.engine.io.image_io import ensure_working_copy

        try:
            self._raw_image = ensure_working_copy(section, self._state.project.working_scale)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        # Check out any resident draft now that the image is available.  Its
        # base_flip carries the last-saved flip state across navigation.
        draft = self._state.pop_prep_draft(section.id)
        if draft is not None:
            self._prep_base_flip = (draft.base_flip_h, draft.base_flip_v)

        self._load_or_init_mask()
        # Overlay any resident draft edits on top of the disk-loaded mask.
        if draft is not None:
            if draft.mask_dirty:
                self._current_mask = draft.slice_mask
                self._mask_dirty = True
        # Reflect the section's persistent dirty state (mask/flip edits).
        # When still dirty, recover the genuine last-saved baseline from the
        # stash (the section's flips may already carry the unsaved edit).
        if self._state.is_dirty(section.id, "prep"):
            stashed = self._state.get_baseline(section.id, "prep")
            if stashed is not None:
                self._baseline_preprocessing = stashed
            self._set_dirty(True)
        else:
            self._state.pop_baseline(section.id, "prep")
            self._set_dirty(False)
        self._display_image()
        self._update_mask_overlay()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing parameter changes."""
        # If the mask was released by flush_draft() on a prior navigation
        # away, re-checkout the resident draft / reload from disk before
        # rendering — otherwise re-entry on the same section shows no overlay.
        if self._arrays_released:
            self._restore_released_masks()
        self._display_image()
        self._update_mask_overlay()

    def _restore_released_masks(self) -> None:
        """Reload the slice mask released by :meth:`flush_draft`.

        Mirrors the mask-checkout portion of :meth:`load_section` (disk load
        plus any resident draft overlay) without disturbing the still-valid
        dirty state, baseline, or flip bookkeeping held in memory.
        """
        self._arrays_released = False
        if self._section is None or self._raw_image is None:
            return
        draft = self._state.pop_prep_draft(self._section.id)
        self._load_or_init_mask()
        if draft is not None:
            if draft.mask_dirty:
                self._current_mask = draft.slice_mask
                self._mask_dirty = True

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
        self._mask_dirty = True
        self._set_dirty(True)
        self._update_mask_overlay()

    def clear_mask(self) -> None:
        if self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = np.zeros(self._raw_image.shape[:2], dtype=bool)
        self._mask_dirty = True
        self._set_dirty(True)
        self._update_mask_overlay()

    def apply_morph(self, pixels: int, operation: str) -> None:
        if self._current_mask is None or self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = morph_mask(self._current_mask, pixels, operation)
        self._mask_dirty = True
        self._set_dirty(True)
        self._update_mask_overlay()

    def undo_mask_edit(self) -> None:
        if not self._undo_stack:
            return
        self._current_mask = self._undo_stack.pop()
        self._mask_dirty = True
        self._set_dirty(True)
        self._update_mask_overlay()

    # ------------------------------------------------------------------
    # Draft / save / clear / flush
    # ------------------------------------------------------------------

    def flush_draft(self) -> None:
        """Stash unsaved edits into the resident draft store; release arrays.

        Called when navigating away from a prep section so its in-RAM mask
        survives (keyed by section id) without this view holding the array.
        The section stays dirty in the registry — this only moves the payload.
        """
        section = self._section
        if section is None:
            return
        if self._dirty:
            draft = PrepDraft(
                slice_mask=self._current_mask if self._mask_dirty else None,
                mask_dirty=self._mask_dirty,
                base_flip_h=self._prep_base_flip[0],
                base_flip_v=self._prep_base_flip[1],
            )
            self._state.set_prep_draft(section.id, draft)
        self._current_mask = None
        self._arrays_released = True
        self._undo_stack.clear()
        self._stroke_points.clear()
        self._stroke_active = False

    def mark_flip_changed(self) -> None:
        """Called by MainWindow after toggling a flip flag on the section."""
        if self._section is None:
            return
        self._set_dirty(True)

    def is_dirty(self) -> bool:
        return self._dirty

    def has_persisted_state(self) -> bool:
        """Whether Clear has anything to wipe in the project for this slice."""
        baseline = self._baseline_preprocessing
        if baseline is None:
            return False
        return bool(baseline.slice_mask_path or baseline.flip_horizontal or baseline.flip_vertical)

    def save(self) -> bool:
        """Persist the current draft to disk + section.

        Writes the slice mask PNG and updates the section's preprocessing
        paths.  A flip already invalidated the alignment + warp at the moment
        it was toggled (see ``_invalidate_alignment_for_flip``), so saving the
        prep draft never touches the alignment here.

        Returns True iff anything actually changed.
        """
        if self._section is None:
            return False

        draft = PrepDraft(
            slice_mask=self._current_mask if self._mask_dirty else None,
            mask_dirty=self._mask_dirty,
            base_flip_h=self._prep_base_flip[0],
            base_flip_v=self._prep_base_flip[1],
        )
        flip_changed = self._prep_base_flip != (
            self._section.preprocessing.flip_horizontal,
            self._section.preprocessing.flip_vertical,
        )
        changed = self._mask_dirty or flip_changed
        persist_prep_draft(self._section, draft)

        self._mask_dirty = False
        self._state.pop_prep_draft(self._section.id)
        self._baseline_preprocessing = copy.deepcopy(self._section.preprocessing)
        self._state.pop_baseline(self._section.id, "prep")
        self._prep_base_flip = (
            self._section.preprocessing.flip_horizontal,
            self._section.preprocessing.flip_vertical,
        )
        self._set_dirty(False)
        return changed

    def revert(self) -> bool:
        """Discard unsaved prep edits, restoring the last-saved preprocessing.

        Drops the resident mask draft and any flip toggled since the last
        save, then reloads mask + flips from the baseline so the canvas matches.
        Does not touch on-disk state.
        """
        if self._section is None or self._baseline_preprocessing is None:
            return False
        self._section.preprocessing = copy.deepcopy(self._baseline_preprocessing)
        self._state.pop_prep_draft(self._section.id)
        self._state.pop_baseline(self._section.id, "prep")
        self._current_mask = None
        self._mask_dirty = False
        self._undo_stack.clear()
        self._stroke_points.clear()
        self._stroke_active = False
        self._prep_base_flip = (
            self._section.preprocessing.flip_horizontal,
            self._section.preprocessing.flip_vertical,
        )
        self._set_dirty(False)
        if self._raw_image is not None:
            self._load_or_init_mask()
            self._display_image()
            self._update_mask_overlay()
        return True

    def clear(self) -> bool:
        """Wipe this slice's prep state: mask, flips.

        Deletes the on-disk PNG, drops any resident draft, and resets the
        section's preprocessing to defaults.  If a saved flip is thereby
        removed, the slice's alignment + warp are invalidated too.
        """
        if self._section is None:
            return False

        # A previously-saved flip is being undone → alignment no longer valid.
        flip_changed = self._prep_base_flip[0] or self._prep_base_flip[1]

        path_str = self._section.preprocessing.slice_mask_path
        if path_str:
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError:
                pass

        self._section.preprocessing = Preprocessing()
        self._current_mask = None
        self._mask_dirty = False
        self._undo_stack.clear()
        self._state.pop_prep_draft(self._section.id)

        if flip_changed:
            self._wipe_alignment_for_flip()

        self._baseline_preprocessing = copy.deepcopy(self._section.preprocessing)
        self._state.pop_baseline(self._section.id, "prep")
        self._prep_base_flip = (False, False)
        self._set_dirty(False)

        # Reload from the now-empty state so the canvas matches.
        if self._raw_image is not None:
            self._load_or_init_mask()
            self._display_image()
            self._update_mask_overlay()
        if flip_changed:
            self.alignment_invalidated.emit()
        return True

    def _wipe_alignment_for_flip(self) -> None:
        if self._section is None:
            return
        from verso.engine.drafts import wipe_alignment_for_flip

        wipe_alignment_for_flip(self._section)

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        if dirty and self._section is not None and self._baseline_preprocessing is not None:
            self._state.set_baseline(
                self._section.id, "prep", copy.deepcopy(self._baseline_preprocessing)
            )
        self._dirty = dirty
        self.dirty_changed.emit(dirty)

    # ------------------------------------------------------------------
    # Display / mask state
    # ------------------------------------------------------------------

    def _push_channel_planes(self, img: np.ndarray, flip_h: bool, flip_v: bool, n: int) -> None:
        """Push raw uint8 planes to the canvas (only when section / flip changes)."""
        planes: list[np.ndarray | None] = [np.ascontiguousarray(img[:, :, i]) for i in range(n)]
        self._canvas.set_channel_planes(planes)

    def _display_image(self) -> None:
        if self._raw_image is None:
            self._canvas.clear()
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

        # Re-push raw planes only when section / flip / channel count changes;
        # this is the only path that touches the GPU texture.
        planes_key = (id(self._raw_image), flip_h, flip_v, n)
        if planes_key != self._channel_planes_key:
            self._push_channel_planes(img, flip_h, flip_v, n)
            self._channel_planes_key = planes_key

        # Apply per-channel LUT + visibility — this is what the brightness
        # slider drives, and it's the cheap path (a 1 KB table swap per tick).
        for i in range(n):
            spec = self._channels[i]
            if not getattr(spec, "visible", True) or float(spec.scale) <= 0:
                self._canvas.set_channel_visible(i, False)
            else:
                self._canvas.set_channel_lut(i, channel_lut(spec))

    def _load_or_init_mask(self) -> None:
        if self._section is None or self._raw_image is None:
            self._current_mask = None
            return

        shape = self._raw_image.shape[:2]
        path = self._section.preprocessing.slice_mask_path
        if path and Path(path).exists():
            try:
                self._current_mask = load_mask(path, shape)
                return
            except Exception:
                pass
        self._current_mask = np.zeros(shape, dtype=bool)

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

    def _push_undo(self) -> None:
        if self._current_mask is None:
            return
        self._undo_stack.append(self._current_mask.copy())
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)

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
            self._current_mask,
            pts,
            add=not self._stroke_erase,
        )
        self._stroke_points.clear()
        self._mask_dirty = True
        self._set_dirty(True)
        self._update_mask_overlay()

    def _paint_brush_segment(self, display_points: list[tuple[float, float]]) -> None:
        """Stamp the brush along ``display_points`` into the live mask."""
        pts = self._stroke_points_to_mask_coords(display_points)
        self._current_mask = apply_brush_stroke(
            self._current_mask,
            pts,
            radius=self._brush_radius,
            add=not self._stroke_erase,
        )
        self._mask_dirty = True
        self._set_dirty(True)
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
