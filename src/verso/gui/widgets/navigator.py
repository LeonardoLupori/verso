"""Three-panel orthogonal atlas navigator for the Align/Warp view.

Displays coronal, sagittal, and horizontal reference slices with the
current cut-plane quadrilateral drawn as a coloured outline.  Interaction:

- Drag near the crosshair center  → translate cut plane
- Drag away from the crosshair    → rotate cut plane around that view's axis
- Side ↑/↓ buttons                → step the plane along the view's row axis
- Bottom ← ⟲ ⟳ → buttons          → step / rotate around the view's perp axis

Each view has its own height derived from atlas dimensions so that
proportions are preserved.  The horizontal view (LR × AP) is notably
taller for the mouse brain (~208 px vs ~120 px for coronal).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume

# Fixed display width for every mini-view (height is computed per-axis from dims)
_VIEW_W = 150
# Side ↑/↓ buttons: narrower than before, same height
_SIDE_BTN_W = 14
_SIDE_BTN_H = 24
# Bottom ← ⟲ ⟳ → buttons: same width as before, shorter height
_BOTTOM_BTN_W = 24
_BOTTOM_BTN_H = 14
# Title-bar height
_TITLE_H = 16
# Per-click translation step (atlas voxels)
_MOVE_STEP = 1
# Per-click rotation step (degrees)
_ROTATE_STEP_DEG = 1.0

_OUTLINE_COLOR = QColor(255, 80, 80, 220)
_CENTER_COLOR = QColor(255, 255, 0, 220)

# Radius (px) within which a press counts as "near center" → translate mode
_TRANSLATE_RADIUS = 14

_NAV_BTN_QSS = (
    "QPushButton { border-radius: 3px; padding: 0px; color: #ccc;"
    " background: #383838; border: 1px solid #555; font-size: 12px; }"
    "QPushButton:hover { background: #484848; }"
    "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
)


def _ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    data = np.ascontiguousarray(rgb)
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)


def _rot_around(vec: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of *vec* around *axis* by *deg* degrees."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    k = axis / np.linalg.norm(axis)
    return c * vec + s * np.cross(k, vec) + (1.0 - c) * np.dot(k, vec) * k


def _view_height(axis: int, dims: tuple[int, int, int]) -> int:
    """Return the display height (px) that preserves atlas proportions."""
    ap_dim, dv_dim, lr_dim = dims
    if axis == 0:    # sagittal:   cols = AP, rows = DV
        h = _VIEW_W * dv_dim / ap_dim
    elif axis == 1:  # coronal:    cols = LR, rows = DV
        h = _VIEW_W * dv_dim / lr_dim
    else:            # horizontal: cols = LR, rows = AP
        h = _VIEW_W * ap_dim / lr_dim
    return max(40, round(h))


class _SliceCanvas(QLabel):
    """QLabel that forwards mouse events as signals so :class:`_SliceView`
    can own drag logic without filtering by widget region.
    """

    pressed = pyqtSignal(float, float)
    moved = pyqtSignal(float, float)
    released = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        self.pressed.emit(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:
        self.moved.emit(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event) -> None:
        self.released.emit()


class _SliceView(QWidget):
    """One axis-aligned atlas view with cut-plane outline overlay.

    axis=0  sagittal   – fix LR, show AP (cols) × DV (rows)
    axis=1  coronal    – fix AP, show LR (cols) × DV (rows)
    axis=2  horizontal – fix DV, show LR (cols) × AP (rows)

    Rotation axes in atlas (LR, AP, DV) space:
        sagittal  (axis=0) → LR = [1, 0, 0]
        coronal   (axis=1) → AP = [0, 1, 0]
        horizontal(axis=2) → DV = [0, 0, 1]
    """

    anchoring_changed = pyqtSignal(list)

    _ROTATION_AXES = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([0.0, 0.0, 1.0]),
    }
    # Drag-clockwise → plane-rotates-clockwise in each view.
    # Derived from Rodrigues applied to the cut-plane center: positive θ moves
    # the leading edge in the clockwise direction in each view's screen space.
    _ANGLE_SIGNS = {0: +1, 1: -1, 2: +1}
    # Atlas axes addressed by each view's left/right and up/down buttons:
    #   view axis → (col_atlas_axis, row_atlas_axis)
    _TRANSLATE_AXES = {
        0: (1, 2),  # sagittal:   cols=AP, rows=DV
        1: (0, 2),  # coronal:    cols=LR, rows=DV
        2: (0, 1),  # horizontal: cols=LR, rows=AP
    }

    def __init__(
        self,
        axis: int,
        title: str,
        dims: tuple[int, int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._axis = axis
        self._dims = dims
        self._anchoring: list[float] | None = None
        self._view_h = _view_height(axis, dims)
        self._reverse_ap: bool = False

        # drag state
        self._drag_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_start_anchoring: list[float] | None = None

        self.setStyleSheet("background: #1a1a1a;")

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Row 0: title spans both columns.
        lbl_title = QLabel(title)
        lbl_title.setFixedHeight(_TITLE_H)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #aaa; font-size: 10px; background: #222;")
        layout.addWidget(lbl_title, 0, 0, 1, 2)

        # Row 1, col 0: canvas (the actual atlas slice).
        self._canvas = _SliceCanvas()
        self._canvas.setFixedSize(_VIEW_W, self._view_h)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        self._canvas.pressed.connect(self._on_canvas_pressed)
        self._canvas.moved.connect(self._on_canvas_moved)
        self._canvas.released.connect(self._on_canvas_released)
        layout.addWidget(self._canvas, 1, 0)

        # Row 1, col 1: ↑ / ↓ side column.
        self._btn_up = self._make_btn("↑", "Move up (1 voxel)", _SIDE_BTN_W, _SIDE_BTN_H)
        self._btn_down = self._make_btn("↓", "Move down (1 voxel)", _SIDE_BTN_W, _SIDE_BTN_H)
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(2)
        side.addStretch()
        side.addWidget(self._btn_up)
        side.addSpacing(3)
        side.addWidget(self._btn_down)
        side.addStretch()
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(_SIDE_BTN_W)
        layout.addWidget(side_widget, 1, 1)

        # Row 2, col 0: ← ⟲ ⟳ → bottom row, tightly grouped and centered under the canvas.
        self._btn_left = self._make_btn("←", "Move left (1 voxel)", _BOTTOM_BTN_W, _BOTTOM_BTN_H)
        self._btn_ccw = self._make_btn(
            "⟲", "Rotate counter-clockwise (1°)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_cw = self._make_btn(
            "⟳", "Rotate clockwise (1°)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_right = self._make_btn("→", "Move right (1 voxel)", _BOTTOM_BTN_W, _BOTTOM_BTN_H)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(2)
        bottom.addStretch()
        for b in (self._btn_left, self._btn_ccw, self._btn_cw, self._btn_right):
            bottom.addWidget(b)
        bottom.addStretch()
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom)
        bottom_widget.setFixedHeight(_BOTTOM_BTN_H)
        bottom_widget.setFixedWidth(_VIEW_W)
        layout.addWidget(bottom_widget, 2, 0)

        # Wire buttons → helpers.
        col_axis, row_axis = self._TRANSLATE_AXES[axis]
        self._btn_left.clicked.connect(lambda: self._translate_step(col_axis, -_MOVE_STEP))
        self._btn_right.clicked.connect(lambda: self._translate_step(col_axis, +_MOVE_STEP))
        self._btn_up.clicked.connect(lambda: self._translate_step(row_axis, -_MOVE_STEP))
        self._btn_down.clicked.connect(lambda: self._translate_step(row_axis, +_MOVE_STEP))
        self._btn_ccw.clicked.connect(lambda: self._rotate_step(-_ROTATE_STEP_DEG))
        self._btn_cw.clicked.connect(lambda: self._rotate_step(+_ROTATE_STEP_DEG))

        self.set_buttons_enabled(False)

        total_w = _VIEW_W + _SIDE_BTN_W + 4
        total_h = _TITLE_H + self._view_h + _BOTTOM_BTN_H + 6
        self.setFixedSize(total_w, total_h)

        self._base_pixmap: QPixmap | None = None
        self._corners: list[tuple[float, float]] = []
        self._center_display: tuple[float, float] | None = None

    def _make_btn(self, text: str, tooltip: str, w: int, h: int) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(w, h)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(_NAV_BTN_QSS)
        return btn

    # ------------------------------------------------------------------
    # Public update methods

    def set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            self._btn_up,
            self._btn_down,
            self._btn_left,
            self._btn_right,
            self._btn_ccw,
            self._btn_cw,
        ):
            btn.setEnabled(enabled)

    def set_reverse_ap(self, reverse: bool) -> None:
        self._reverse_ap = reverse

    def update_dims(self, dims: tuple[int, int, int]) -> None:
        """Update atlas dimensions and resize the widget to preserve proportions."""
        self._dims = dims
        new_h = _view_height(self._axis, dims)
        if new_h != self._view_h:
            self._view_h = new_h
            self._canvas.setFixedSize(_VIEW_W, self._view_h)
            total_w = _VIEW_W + _SIDE_BTN_W + 4
            total_h = _TITLE_H + self._view_h + _BOTTOM_BTN_H + 6
            self.setFixedSize(total_w, total_h)

    def set_image(self, rgb: np.ndarray) -> None:
        qimg = _ndarray_to_qimage(rgb)
        self._base_pixmap = QPixmap.fromImage(qimg).scaled(
            _VIEW_W, self._view_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._redraw()

    def set_cut(self, anchoring: list[float]) -> None:
        self._anchoring = anchoring
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        self._corners = [self._proj(c) for c in [o, o + u, o + u + v, o + v]]
        self._center_display = self._proj(o + u / 2.0 + v / 2.0)
        self._redraw()

    # ------------------------------------------------------------------
    # Button-driven translate / rotate helpers

    def _translate_step(self, atlas_axis: int, delta: float) -> None:
        """Translate cut-plane origin along an atlas axis by *delta* voxels."""
        if self._anchoring is None:
            return
        if atlas_axis == 1 and self._reverse_ap:
            delta = -delta
        new_anchoring = list(self._anchoring)
        new_anchoring[atlas_axis] += delta
        self.anchoring_changed.emit(new_anchoring)

    def _rotate_step(self, deg_signed: float) -> None:
        """Rotate cut plane around this view's perpendicular atlas axis."""
        if self._anchoring is None:
            return
        deg = deg_signed * self._ANGLE_SIGNS[self._axis]
        if self._reverse_ap and self._axis != 1:
            deg = -deg
        o = np.array(self._anchoring[:3])
        u = np.array(self._anchoring[3:6])
        v = np.array(self._anchoring[6:9])
        center = o + u / 2.0 + v / 2.0
        rot_axis = self._ROTATION_AXES[self._axis]
        u_new = _rot_around(u, rot_axis, deg)
        v_new = _rot_around(v, rot_axis, deg)
        new_o = center - u_new / 2.0 - v_new / 2.0
        self.anchoring_changed.emit(new_o.tolist() + u_new.tolist() + v_new.tolist())

    # ------------------------------------------------------------------
    # Coordinate helpers

    def _proj(self, pt: np.ndarray) -> tuple[float, float]:
        """Project atlas (LR, AP, DV) to display pixel coords."""
        ap_dim, dv_dim, lr_dim = self._dims
        lr, ap, dv = pt[0], pt[1], pt[2]
        if self._axis == 0:
            return ap / ap_dim * _VIEW_W, dv / dv_dim * self._view_h
        elif self._axis == 1:
            return lr / lr_dim * _VIEW_W, dv / dv_dim * self._view_h
        else:
            return lr / lr_dim * _VIEW_W, ap / ap_dim * self._view_h

    def _unproj_normalized(self, s: float, t: float) -> np.ndarray:
        """Normalised view coords (s=col/W, t=row/H) → atlas position vector."""
        ap_dim, dv_dim, lr_dim = self._dims
        if self._axis == 0:
            return np.array([0.0, s * ap_dim, t * dv_dim])
        elif self._axis == 1:
            return np.array([s * lr_dim, 0.0, t * dv_dim])
        else:
            return np.array([s * lr_dim, t * ap_dim, 0.0])

    # ------------------------------------------------------------------
    # Drawing

    def _redraw(self) -> None:
        if self._base_pixmap is None:
            return
        pm = self._base_pixmap.copy()
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._corners:
            painter.setPen(QPen(_OUTLINE_COLOR, 1.5))
            poly = QPolygonF([QPointF(x, y) for x, y in self._corners])
            painter.drawPolygon(poly)

        if self._center_display:
            cx, cy = self._center_display
            painter.setPen(QPen(_CENTER_COLOR, 2))
            r = 4
            painter.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
            painter.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))
            painter.setPen(QPen(QColor(200, 200, 100, 120), 1))
            painter.drawEllipse(QPointF(cx, cy), _TRANSLATE_RADIUS, _TRANSLATE_RADIUS)

        painter.end()
        self._canvas.setPixmap(pm)

    # ------------------------------------------------------------------
    # Mouse interaction (anchored to the canvas widget itself)

    def _center_in_view(self) -> bool:
        """Return True if the cut-plane center crosshair is within the view bounds."""
        if self._center_display is None:
            return False
        cx, cy = self._center_display
        return 0 <= cx <= _VIEW_W and 0 <= cy <= self._view_h

    def _on_canvas_pressed(self, cx: float, cy: float) -> None:
        if self._anchoring is None:
            return
        self._drag_start = (cx, cy)
        self._drag_start_anchoring = list(self._anchoring)
        # Force translate when center is off-screen; otherwise decide by proximity
        if not self._center_in_view():
            self._drag_mode = "translate"
        elif self._center_display is not None:
            dx = cx - self._center_display[0]
            dy = cy - self._center_display[1]
            dist = math.sqrt(dx * dx + dy * dy)
            self._drag_mode = "translate" if dist <= _TRANSLATE_RADIUS else "rotate"
        else:
            self._drag_mode = "translate"
        # For translate: apply immediately so a single click moves the plane
        if self._drag_mode == "translate":
            self._handle_translate(cx, cy)

    def _on_canvas_moved(self, cx: float, cy: float) -> None:
        if self._drag_mode is None or self._drag_start is None:
            return
        if self._drag_mode == "translate":
            self._handle_translate(cx, cy)
        else:
            self._handle_rotate(cx, cy)

    def _on_canvas_released(self) -> None:
        self._drag_mode = None
        self._drag_start = None
        self._drag_start_anchoring = None

    def _handle_translate(self, cx: float, cy: float) -> None:
        """Absolute placement: move cut-plane center to the cursor's atlas position."""
        if self._drag_start_anchoring is None:
            return
        anchoring = self._drag_start_anchoring
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        old_center = o + u / 2.0 + v / 2.0

        cur_atlas = self._unproj_normalized(cx / _VIEW_W, cy / self._view_h)

        new_center = old_center.copy()
        if self._axis == 0:       # sagittal: move AP and DV
            new_center[1] = cur_atlas[1]
            new_center[2] = cur_atlas[2]
        elif self._axis == 1:     # coronal: move LR and DV
            new_center[0] = cur_atlas[0]
            new_center[2] = cur_atlas[2]
        else:                     # horizontal: move LR and AP
            new_center[0] = cur_atlas[0]
            new_center[1] = cur_atlas[1]

        new_o = new_center - u / 2.0 - v / 2.0
        self.anchoring_changed.emit(new_o.tolist() + anchoring[3:])

    def _handle_rotate(self, cx: float, cy: float) -> None:
        if self._drag_start_anchoring is None or self._center_display is None:
            return
        sx, sy = self._drag_start
        ccx, ccy = self._center_display
        start_angle = math.atan2(sy - ccy, sx - ccx)
        cur_angle = math.atan2(cy - ccy, cx - ccx)
        deg = math.degrees(cur_angle - start_angle) * self._ANGLE_SIGNS[self._axis]
        # Sagittal (axis=0) and horizontal (axis=2) rotate around axes that tilt
        # the plane in the AP direction — invert when the series is AP-reversed.
        if self._reverse_ap and self._axis != 1:
            deg = -deg

        o = np.array(self._drag_start_anchoring[:3])
        u = np.array(self._drag_start_anchoring[3:6])
        v = np.array(self._drag_start_anchoring[6:9])
        center = o + u / 2.0 + v / 2.0

        rot_axis = self._ROTATION_AXES[self._axis]
        u_new = _rot_around(u, rot_axis, deg)
        v_new = _rot_around(v, rot_axis, deg)
        new_o = center - u_new / 2.0 - v_new / 2.0
        self.anchoring_changed.emit(new_o.tolist() + u_new.tolist() + v_new.tolist())


class NavigatorPanel(QWidget):
    """Scrollable vertical stack of three orthogonal atlas slice views."""

    anchoring_changed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atlas: AtlasVolume | None = None
        self._anchoring: list[float] | None = None
        # Reserve space for the vertical scrollbar so it never overlaps the
        # rightmost ↑/↓ buttons when the panel becomes too short to fit all
        # three slice views.
        sb_extent = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        self.setFixedWidth(_VIEW_W + _SIDE_BTN_W + 8 + sb_extent)
        self.setStyleSheet("background: #1a1a1a;")

        # Outer layout holds a scroll area so the tall horizontal view is reachable
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet("background: #1a1a1a;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)

        dims = (528, 320, 456)
        self._sag = _SliceView(0, "Sagittal (LR)", dims)
        self._cor = _SliceView(1, "Coronal (AP)", dims)
        self._hor = _SliceView(2, "Horizontal (DV)", dims)

        for view in (self._sag, self._cor, self._hor):
            view.anchoring_changed.connect(self._on_anchoring_changed)
            layout.addWidget(view)

        layout.addStretch()

    def set_reverse_ap(self, reverse: bool) -> None:
        for view in (self._sag, self._cor, self._hor):
            view.set_reverse_ap(reverse)

    def set_atlas(self, atlas: AtlasVolume | None) -> None:
        self._atlas = atlas
        if atlas is not None:
            dims = atlas.shape  # (AP, DV, LR)
            for view in (self._sag, self._cor, self._hor):
                view.update_dims(dims)
        self._refresh_images()

    def set_anchoring(self, anchoring: list[float] | None) -> None:
        self._anchoring = anchoring
        self._refresh_images()

    def _refresh_images(self) -> None:
        if self._atlas is None or self._anchoring is None:
            for view in (self._sag, self._cor, self._hor):
                view.set_buttons_enabled(False)
            return
        center = self._atlas.cut_center(self._anchoring)
        lr_c = int(round(center[0]))
        ap_c = int(round(center[1]))
        dv_c = int(round(center[2]))

        self._sag.set_image(self._atlas.get_orthogonal_slice(0, lr_c))
        self._cor.set_image(self._atlas.get_orthogonal_slice(1, ap_c))
        self._hor.set_image(self._atlas.get_orthogonal_slice(2, dv_c))

        for view in (self._sag, self._cor, self._hor):
            view.set_buttons_enabled(True)
            view.set_cut(self._anchoring)

    def _on_anchoring_changed(self, new_anchoring: list[float]) -> None:
        self._anchoring = new_anchoring
        self._refresh_images()
        self.anchoring_changed.emit(new_anchoring)
