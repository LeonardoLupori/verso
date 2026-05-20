"""Prep view - canvas for section preprocessing (masks, flipping)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec, Section
from verso.engine.preprocessing import (
    apply_freehand_stroke,
    composite_from_layers,
    compute_channel_layer,
    detect_foreground,
    flip_lr_mask,
    load_lr_mask,
    load_mask,
    lr_mask_to_rgba,
    mask_to_rgba,
    rasterize_lr_line,
    save_lr_mask,
    save_mask,
)
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.lr_line_editor import LRLineEditor


class PrepView(QWidget):
    """Canvas view for the Prep (mask drawing / flip) step."""

    section_modified = pyqtSignal()
    mask_negative_changed = pyqtSignal(bool)

    _UNDO_LIMIT = 20
    _DRAW_COLOR = (80, 160, 255)
    _ERASE_COLOR = (255, 90, 90)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section: Section | None = None
        self._raw_image: np.ndarray | None = None
        self._current_mask: np.ndarray | None = None
        self._mask_dirty = False
        self._mask_opacity = 0.4
        self._mask_color = (255, 255, 255)
        self._negative_mask = False
        self._mask_visible = True
        self._channels: list[ChannelSpec] = []
        self._channel_layers: list[np.ndarray | None] = []
        self._cached_channel_specs: list[tuple] = []
        self._layer_image_key: tuple = ()
        self._undo_stack: list[np.ndarray] = []
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        self._active_tool = "draw"
        # L/R hemisphere mask state.  Stored unflipped (matches slice mask).
        self._lr_mask: np.ndarray | None = None
        self._lr_dirty = False
        self._lr_visible = True
        self._lr_overlay_needs_update = False
        # Draw-mode state — created/destroyed by enter/exit_lr_draw_mode().
        self._lr_editor: LRLineEditor | None = None
        self._lr_draw_mode = False
        self._saved_lr_line: list[list[float]] | None = None
        self._saved_lr_mask_path: str | None = None
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
        self._status_label.setFixedHeight(28)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._status_label.setStyleSheet(
            "background: #252525; color: #aaa; font-size: 11px; padding-left: 8px;"
            " border-bottom: 1px solid #333;"
        )
        canvas_col.addWidget(self._status_label)

        self._canvas = ImageCanvas()
        self._canvas.set_interaction_mode("prep")
        self._canvas.canvas_drag_started.connect(self._on_canvas_drag_started)
        self._canvas.canvas_dragged.connect(self._on_canvas_dragged)
        self._canvas.canvas_drag_ended.connect(self._on_canvas_drag_ended)
        canvas_col.addWidget(self._canvas, stretch=1)
        layout.addLayout(canvas_col, stretch=1)

        # Right tool palette
        self._toolbar = self._make_toolbar()
        layout.addWidget(self._toolbar)

        self._install_shortcuts()

    def _make_toolbar(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(48)
        container.setStyleSheet("background: #2a2a2a;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 8, 4, 8)
        v.setSpacing(4)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tool_group = QButtonGroup()
        tools = [
            ("D", "Draw mask (D)", "draw"),
            ("E", "Erase mask (E)", "erase"),
        ]
        btn_ss = (
            "QToolButton { color: #ccc; border-radius: 4px; }"
            "QToolButton:checked { background: #1e5a8a; }"
            "QToolButton:hover { background: #444; }"
        )
        icon_font = QFont()
        icon_font.setPointSize(11)
        icon_font.setBold(True)

        for icon, tip, name in tools:
            btn = QToolButton()
            btn.setText(icon)
            btn.setFont(icon_font)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(btn_ss)
            btn.setProperty("tool_name", name)
            btn.clicked.connect(lambda _checked, n=name: self._set_tool(n))
            self._tool_group.addButton(btn)
            v.addWidget(btn)

        if self._tool_group.buttons():
            self._tool_group.buttons()[0].setChecked(True)

        v.addStretch()

        undo_font = QFont()
        undo_font.setPointSize(14)
        undo_btn = QToolButton()
        undo_btn.setText("U")
        undo_btn.setFont(undo_font)
        undo_btn.setToolTip("Undo (U or Ctrl+Z)")
        undo_btn.setFixedSize(36, 36)
        undo_btn.setStyleSheet(
            "QToolButton { color: #ccc; border-radius: 4px; }"
            "QToolButton:hover { background: #444; }"
        )
        undo_btn.clicked.connect(self.undo_mask_edit)
        v.addWidget(undo_btn)
        self._undo_btn = undo_btn

        return container

    def _install_shortcuts(self) -> None:
        shortcuts = [
            (Qt.Key.Key_D, lambda: self._set_tool("draw")),
            (Qt.Key.Key_E, lambda: self._set_tool("erase")),
            (Qt.Key.Key_M, lambda: self.set_mask_visible(not self._mask_visible)),
            (Qt.Key.Key_N, lambda: self.set_mask_negative(not self._negative_mask)),
            (Qt.Key.Key_U, self.undo_mask_edit),
            (QKeySequence.StandardKey.Undo, self.undo_mask_edit),
            (Qt.Key.Key_Return, self.save_current_mask),
            (Qt.Key.Key_Enter, self.save_current_mask),
        ]
        for key, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def load_section(self, section: Section | None) -> None:
        # If the user was mid-edit on an L/R line, drop the editor before
        # the section changes.  Cancelling discards the in-flight line.
        self.cancel_lr_draw_if_active()
        self.save_current_mask_if_dirty()
        self._section = section
        self._raw_image = None
        self._current_mask = None
        self._mask_dirty = False
        self._lr_mask = None
        self._lr_dirty = False
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
            self._raw_image = ensure_working_copy(section)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        self._load_or_init_mask()
        self._load_or_init_lr_mask()
        self._display_image()
        self._update_mask_overlay()
        self._update_lr_overlay()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing parameter changes."""
        self._display_image()
        self._update_mask_overlay()
        self._update_lr_overlay()

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_visible = visible
        self._update_mask_overlay()

    def set_lr_visible(self, visible: bool) -> None:
        self._lr_visible = bool(visible)
        if not visible:
            self._canvas.set_lr_overlay_visible(False)
        elif self._lr_mask is not None:
            if self._lr_overlay_needs_update:
                self._update_lr_overlay()
            else:
                self._canvas.set_lr_overlay_visible(True)

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
        self._canvas.set_lr_overlay_opacity(self._mask_opacity)

    def set_mask_color(self, color: tuple[int, int, int]) -> None:
        self._mask_color = color
        self._update_mask_overlay()

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        self._channels = list(channels)
        self._display_image()

    def autodetect_mask(self) -> None:
        if self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = detect_foreground(self._raw_image)
        self._mask_dirty = True
        self._update_mask_overlay()

    def clear_mask(self) -> None:
        if self._raw_image is None:
            return
        self._ensure_mask()
        self._push_undo()
        self._current_mask = np.zeros(self._raw_image.shape[:2], dtype=bool)
        self._mask_dirty = True
        self._update_mask_overlay()

    def undo_mask_edit(self) -> None:
        if not self._undo_stack:
            return
        self._current_mask = self._undo_stack.pop()
        self._mask_dirty = True
        self._update_mask_overlay()

    def save_current_mask(self) -> bool:
        return self._save_current_mask(force=True)

    def save_current_mask_if_dirty(self) -> bool:
        return self._save_current_mask(force=False)

    # ------------------------------------------------------------------
    # L/R hemisphere — public actions
    # ------------------------------------------------------------------

    def set_lr_all(self, side: int) -> None:
        """Label the entire section as left (side=1) or right (side=2).

        Clears any previously-drawn line and writes the mask to disk
        immediately so the change is persisted in the next project save.
        """
        if side not in (1, 2):
            raise ValueError(f"side must be 1 (left) or 2 (right), got {side}")
        if self._section is None or self._raw_image is None:
            return
        self._lr_mask = np.full(
            self._raw_image.shape[:2], np.uint8(side), dtype=np.uint8
        )
        self._section.preprocessing.lr_line = None
        path = self._lr_mask_path_for_section(self._section)
        save_lr_mask(self._lr_mask, path)
        self._section.preprocessing.lr_mask_path = str(path)
        self._lr_dirty = False
        self._update_lr_overlay()
        self.section_modified.emit()

    def clear_lr_mask(self) -> None:
        """Remove the L/R label for the current section (no longer 'edited')."""
        if self._section is None:
            return
        old_path = self._section.preprocessing.lr_mask_path
        if old_path:
            try:
                Path(old_path).unlink(missing_ok=True)
            except OSError:
                pass
        self._lr_mask = None
        self._section.preprocessing.lr_mask_path = None
        self._section.preprocessing.lr_line = None
        self._lr_dirty = False
        self._update_lr_overlay()
        self.section_modified.emit()

    def is_lr_draw_active(self) -> bool:
        return self._lr_draw_mode

    def cancel_lr_draw_if_active(self) -> bool:
        """If the L/R line editor is active, tear it down (no save).

        Returns True if draw mode was active and got cancelled — callers
        (e.g. flip handlers) can use this to also un-toggle the draw button
        in the properties panel.
        """
        if not self._lr_draw_mode:
            return False
        self.exit_lr_draw_mode(apply=False)
        return True

    def enter_lr_draw_mode(self) -> None:
        """Begin interactive editing of the L/R separating line.

        Seeds the editor from ``section.preprocessing.lr_line`` if present,
        otherwise from a default vertical line through the image centre.
        Snapshots the previous L/R state for :meth:`exit_lr_draw_mode`'s
        Cancel path.
        """
        if self._section is None or self._raw_image is None:
            return
        if self._lr_draw_mode:
            return
        self._saved_lr_line = (
            [list(p) for p in self._section.preprocessing.lr_line]
            if self._section.preprocessing.lr_line is not None
            else None
        )
        self._saved_lr_mask_path = self._section.preprocessing.lr_mask_path

        h, w = self._raw_image.shape[:2]
        if self._section.preprocessing.lr_line is not None:
            stored = self._section.preprocessing.lr_line
            p0 = self._line_endpoint_to_display((stored[0][0], stored[0][1]))
            p1 = self._line_endpoint_to_display((stored[1][0], stored[1][1]))
        else:
            # Default: vertical line down the centre, 80 % of image height.
            p0 = (w / 2.0, 0.1 * h)
            p1 = (w / 2.0, 0.9 * h)

        self._lr_editor = LRLineEditor(self._canvas)
        self._lr_editor.begin(p0, p1, w, h)
        self._lr_draw_mode = True

    def exit_lr_draw_mode(self, *, apply: bool) -> None:
        """Tear down the line editor.

        If *apply* is True, rasterize the current line into a fresh L/R
        mask, save it to disk, and persist the endpoints to the section.
        If False (cancel), restore the previous mask path / line endpoints
        and reload the on-disk mask without writing.
        """
        if not self._lr_draw_mode:
            return

        if apply and self._lr_editor is not None and self._section is not None \
                and self._raw_image is not None:
            eps = self._lr_editor.endpoints()
            if eps is not None:
                p0_disp, p1_disp = eps
                p0 = self._line_endpoint_to_display(p0_disp)  # involutive
                p1 = self._line_endpoint_to_display(p1_disp)
                h, w = self._raw_image.shape[:2]
                self._lr_mask = rasterize_lr_line(p0, p1, shape=(h, w))
                path = self._lr_mask_path_for_section(self._section)
                save_lr_mask(self._lr_mask, path)
                self._section.preprocessing.lr_mask_path = str(path)
                self._section.preprocessing.lr_line = [
                    [float(p0[0]), float(p0[1])],
                    [float(p1[0]), float(p1[1])],
                ]
                self._lr_dirty = False
                self.section_modified.emit()
        elif not apply and self._section is not None:
            # Cancel: restore the snapshot (path may or may not exist on disk).
            self._section.preprocessing.lr_line = (
                [list(p) for p in self._saved_lr_line]
                if self._saved_lr_line is not None else None
            )
            self._section.preprocessing.lr_mask_path = self._saved_lr_mask_path
            self._load_or_init_lr_mask()

        if self._lr_editor is not None:
            self._lr_editor.end()
            self._lr_editor = None
        self._lr_draw_mode = False
        self._saved_lr_line = None
        self._saved_lr_mask_path = None
        self._update_lr_overlay()

    def _line_endpoint_to_display(
        self,
        p: tuple[float, float],
    ) -> tuple[float, float]:
        """Convert an L/R line endpoint between display and storage frames.

        Storage frame is the unflipped working-resolution image; display
        frame is whatever the user currently sees (flipped per the section's
        preprocessing flags).  The mapping is the same in both directions
        because horizontal/vertical flips are involutive.
        """
        x, y = p
        if self._raw_image is None or self._section is None:
            return x, y
        h, w = self._raw_image.shape[:2]
        if self._section.preprocessing.flip_horizontal:
            x = float(w - 1) - x
        if self._section.preprocessing.flip_vertical:
            y = float(h - 1) - y
        return x, y

    def lr_status_text(self) -> str:
        """Return a short label for the current L/R state of this section.

        Used by the properties panel's "Status:" line.  Values:
        'Not set', 'All left', 'All right', or 'Line drawn'.
        """
        section = self._section
        if section is None or self._lr_mask is None:
            return "Not set"
        if section.preprocessing.lr_line is not None:
            return "Line drawn"
        unique = np.unique(self._lr_mask)
        if len(unique) == 1:
            if unique[0] == 1:
                return "All left"
            if unique[0] == 2:
                return "All right"
        return "Line drawn"

    # ------------------------------------------------------------------
    # Display / mask state
    # ------------------------------------------------------------------

    def _update_channel_layers(self, img: np.ndarray, flip_h: bool, flip_v: bool) -> None:
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
        img = self._raw_image
        flip_h = bool(self._section and self._section.preprocessing.flip_horizontal)
        flip_v = bool(self._section and self._section.preprocessing.flip_vertical)
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        self._update_channel_layers(img, flip_h, flip_v)
        rgb = composite_from_layers(self._channel_layers, self._channels)
        self._canvas.set_background(np.ascontiguousarray(rgb))

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

    def _save_current_mask(self, *, force: bool) -> bool:
        if self._section is None or self._current_mask is None:
            return False
        if not force and not self._mask_dirty:
            return False

        mask_path = self._mask_path_for_section(self._section)
        save_mask(self._current_mask, mask_path)
        self._section.preprocessing.slice_mask_path = str(mask_path)
        self._mask_dirty = False
        self.section_modified.emit()
        return True

    def _mask_path_for_section(self, section: Section) -> Path:
        masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
        return masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"

    # ------------------------------------------------------------------
    # L/R hemisphere mask — load / display / path
    # ------------------------------------------------------------------

    def _load_or_init_lr_mask(self) -> None:
        if self._section is None or self._raw_image is None:
            self._lr_mask = None
            return

        shape = self._raw_image.shape[:2]
        path = self._section.preprocessing.lr_mask_path
        if path and Path(path).exists():
            try:
                self._lr_mask = load_lr_mask(path, shape)
                return
            except Exception:
                pass
        self._lr_mask = None  # "not edited" — no overlay, no file on disk yet

    def _update_lr_overlay(self) -> None:
        if self._lr_mask is None:
            self._canvas.set_lr_overlay(None)
            self._lr_overlay_needs_update = False
            return
        if not self._lr_visible:
            self._canvas.set_lr_overlay_visible(False)
            self._lr_overlay_needs_update = True
            return
        display_mask = self._lr_mask_for_display()
        rgba = lr_mask_to_rgba(display_mask, opacity=1.0)
        h, w = display_mask.shape
        self._canvas.set_lr_overlay(rgba, display_w=w, display_h=h)
        self._canvas.set_lr_overlay_opacity(self._mask_opacity)
        self._canvas.set_lr_overlay_visible(True)
        self._lr_overlay_needs_update = False

    def _lr_mask_for_display(self) -> np.ndarray:
        if self._lr_mask is None:
            raise RuntimeError("No L/R mask loaded")
        flip_h = bool(self._section and self._section.preprocessing.flip_horizontal)
        flip_v = bool(self._section and self._section.preprocessing.flip_vertical)
        return flip_lr_mask(self._lr_mask, horizontal=flip_h, vertical=flip_v)

    def _lr_mask_path_for_section(self, section: Section) -> Path:
        lr_dir = Path(section.thumbnail_path).parent.parent / "lr_masks"
        return lr_dir / f"{Path(section.original_path).stem}_lr.png"

    # ------------------------------------------------------------------
    # Tool / stroke handling
    # ------------------------------------------------------------------

    def _set_tool(self, tool: str) -> None:
        if tool not in {"draw", "erase"}:
            return
        self._active_tool = tool
        for btn in self._tool_group.buttons():
            btn.setChecked(btn.property("tool_name") == tool)

    def _on_canvas_drag_started(self, x: float, y: float) -> None:
        if self._raw_image is None or self._section is None:
            return
        if self._active_tool not in {"draw", "erase"}:
            return
        point = self._clamped_display_point(x, y)
        self._stroke_points = [point]
        self._stroke_active = True
        self._canvas.clear_stroke_preview()

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        self._stroke_points.append(self._clamped_display_point(x, y))
        color = self._DRAW_COLOR if self._active_tool == "draw" else self._ERASE_COLOR
        self._canvas.set_stroke_preview(self._stroke_points, color=color)

    def _on_canvas_drag_ended(self, x: float, y: float) -> None:
        if not self._stroke_active:
            return
        self._stroke_points.append(self._clamped_display_point(x, y))
        self._stroke_active = False
        self._canvas.clear_stroke_preview()
        if len(self._stroke_points) < 3:
            self._stroke_points.clear()
            return

        self._ensure_mask()
        self._push_undo()
        pts = self._stroke_points_to_mask_coords(self._stroke_points)
        self._current_mask = apply_freehand_stroke(
            self._current_mask,
            pts,
            add=self._active_tool == "draw",
        )
        self._stroke_points.clear()
        self._mask_dirty = True
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
