"""Prep view - canvas for section preprocessing (masks, flipping)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import Section
from verso.engine.preprocessing import (
    apply_channel_luminance,
    apply_freehand_stroke,
    detect_foreground,
    load_mask,
    mask_to_rgba,
    save_mask,
)
from verso.gui.widgets.canvas import ImageCanvas


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
        self._red_luminance = 1.0
        self._green_luminance = 1.0
        self._red_previous_luminance = 1.0
        self._green_previous_luminance = 1.0
        self._undo_stack: list[np.ndarray] = []
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_active = False
        self._active_tool = "draw"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left tool palette
        self._toolbar = self._make_toolbar()
        layout.addWidget(self._toolbar)

        # Central canvas
        self._canvas = ImageCanvas()
        self._canvas.set_interaction_mode("prep")
        self._canvas.canvas_drag_started.connect(self._on_canvas_drag_started)
        self._canvas.canvas_dragged.connect(self._on_canvas_dragged)
        self._canvas.canvas_drag_ended.connect(self._on_canvas_drag_ended)
        layout.addWidget(self._canvas, stretch=1)

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
            ("D", "Draw mask (A)", "draw"),
            ("E", "Erase mask (D)", "erase"),
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
            (Qt.Key.Key_A, lambda: self._set_tool("draw")),
            (Qt.Key.Key_D, lambda: self._set_tool("erase")),
            (Qt.Key.Key_M, lambda: self.set_mask_visible(not self._mask_visible)),
            (Qt.Key.Key_N, lambda: self.set_mask_negative(not self._negative_mask)),
            (Qt.Key.Key_R, self.toggle_red_channel),
            (Qt.Key.Key_G, self.toggle_green_channel),
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
        self.save_current_mask_if_dirty()
        self._section = section
        self._raw_image = None
        self._current_mask = None
        self._mask_dirty = False
        self._undo_stack.clear()
        self._stroke_points.clear()
        self._stroke_active = False
        self._canvas.clear()
        if section is None:
            return

        from PyQt6.QtWidgets import QMessageBox

        from verso.engine.io.image_io import ensure_working_copy

        try:
            self._raw_image = ensure_working_copy(section)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        self._load_or_init_mask()
        self._display_image()
        self._update_mask_overlay()

    def refresh_display(self) -> None:
        """Re-render from cache after preprocessing parameter changes."""
        self._display_image()
        self._update_mask_overlay()

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_visible = visible
        self._update_mask_overlay()

    def set_mask_negative(self, negative: bool) -> None:
        negative = bool(negative)
        if self._negative_mask == negative:
            return
        self._negative_mask = negative
        self._update_mask_overlay()
        self.mask_negative_changed.emit(negative)

    def set_mask_opacity(self, opacity: float) -> None:
        self._mask_opacity = min(max(opacity, 0.0), 1.0)
        self._update_mask_overlay()

    def set_mask_color(self, color: tuple[int, int, int]) -> None:
        self._mask_color = color
        self._update_mask_overlay()

    def set_channel_luminance(self, red: float, green: float) -> None:
        self._red_luminance = min(max(red, 0.0), 1.0)
        self._green_luminance = min(max(green, 0.0), 1.0)
        if self._red_luminance > 0:
            self._red_previous_luminance = self._red_luminance
        if self._green_luminance > 0:
            self._green_previous_luminance = self._green_luminance
        self._display_image()

    def toggle_red_channel(self) -> None:
        if self._red_luminance > 0:
            self._red_previous_luminance = self._red_luminance
            self._red_luminance = 0.0
        else:
            self._red_luminance = self._red_previous_luminance
        self._display_image()

    def toggle_green_channel(self) -> None:
        if self._green_luminance > 0:
            self._green_previous_luminance = self._green_luminance
            self._green_luminance = 0.0
        else:
            self._green_luminance = self._green_previous_luminance
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
    # Display / mask state
    # ------------------------------------------------------------------

    def _display_image(self) -> None:
        if self._raw_image is None:
            return
        img = self._raw_image
        if self._section and self._section.preprocessing.flip_horizontal:
            img = np.fliplr(img)
        if self._section and self._section.preprocessing.flip_vertical:
            img = np.flipud(img)
        img = apply_channel_luminance(
            img,
            red=self._red_luminance,
            green=self._green_luminance,
        )
        self._canvas.set_background(np.ascontiguousarray(img))

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
            opacity=self._mask_opacity,
            color=self._mask_color,
        )
        h, w = display_mask.shape
        self._canvas.set_overlay(rgba, display_w=w, display_h=h)

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
