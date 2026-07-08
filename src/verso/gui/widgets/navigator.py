"""Three-panel orthogonal atlas navigator for the Align/Warp view.

Displays coronal, sagittal, and horizontal reference slices with the
current cut plane drawn as a single coloured line through the frame
center, lettered at each end (e.g. D/V, L/R) to mark the edges.
Interaction:

- Drag near the crosshair center  → translate cut plane
- Drag away from the crosshair    → rotate cut plane around that view's axis
- Side up/down buttons             → step the plane along the view's row axis
- Bottom left/rotate/right buttons → step / rotate around the view's perp axis

Each view has its own height derived from atlas dimensions so that
proportions are preserved.  The horizontal view (LR × AP) is notably
taller for the mouse brain (~208 px vs ~120 px for coronal).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from verso.engine.anchoring import (
    clamp_rotation_to_max_tilt,
    tilt_plane_about_atlas_axis,
)
from verso.gui.utils import require
from verso.gui.widgets.properties._common import colored_icon

# Scale step per stretch button click
_SCALE_STEP = 1.02
_SCALE_STEP_FAST = 1.10


if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume

# Fixed display width for every mini-view (height is computed per-axis from dims)
_VIEW_W = 150
# Side ↑/↓ buttons: narrower than before, same height
_SIDE_BTN_W = 18
_SIDE_BTN_H = 22
# Bottom ← ⟲ ⟳ → buttons: same width as before, shorter height
_BOTTOM_BTN_W = 22
_BOTTOM_BTN_H = 18
# Per-click translation step (atlas voxels)
_MOVE_STEP = 1
_MOVE_STEP_FAST = 10
# Per-click rotation step (degrees)
_ROTATE_STEP_DEG = 1.0
# Maximum plane tilt (degrees) from the slicing axis allowed via navigator views
_MAX_TILT_DEG = 44.0

_OUTLINE_COLOR = QColor(255, 80, 80, 220)
_CENTER_COLOR = QColor(255, 255, 0, 220)
_LABEL_COLOR = QColor(255, 255, 0, 235)
# Anatomical letters at the two ends of the cut-plane line, keyed by the atlas
# axis the line runs along: (letter at low coordinate, letter at high coordinate).
#   axis 0 = LR → Left / Right;  axis 1 = AP → Anterior / Posterior;
#   axis 2 = DV → Dorsal / Ventral.  Matches the canvas orientation conventions.
_AXIS_LETTERS = {0: ("L", "R"), 1: ("A", "P"), 2: ("D", "V")}

# Radius (px) within which a press counts as "near center" → translate mode
_TRANSLATE_RADIUS = 14

# A widget's own stylesheet governs its tooltip rendering, so any styled button
# must carry this rule or its tooltip falls back to the stylesheet engine's
# darker default (mismatching the palette-based tooltips elsewhere).
_TOOLTIP_QSS = "QToolTip { background-color: #323232; color: #dcdcdc; border: 1px solid #555; }"

_NAV_BTN_QSS = (
    "QPushButton { border-radius: 3px; padding: 0px;"
    " background: #383838; border: 1px solid #555; }"
    "QPushButton:hover { background: #484848; }"
    "QPushButton:disabled { background: #2a2a2a; border-color: #333; }" + _TOOLTIP_QSS
)

# View group box: blends into the dark atlas canvas (#1a1a1a) while keeping a
# subtle frame + title so it still reads as a labelled container.  The
# background override disables Fusion's default frame, so border/title are
# redrawn here.
_VIEW_GROUP_QSS = (
    "QGroupBox { background-color: #1a1a1a; border: 1px solid #333;"
    " border-radius: 4px; margin-top: 1.1em; }"
    "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;"
    " left: 8px; padding: 0 4px; color: #bbb; }"
)


def _ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    data = np.ascontiguousarray(rgb)
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)


def _view_height(axis: int, dims: tuple[int, int, int]) -> int:
    """Return the display height (px) that preserves atlas proportions."""
    ap_dim, dv_dim, lr_dim = dims
    if axis == 0:  # sagittal:   cols = AP, rows = DV
        h = _VIEW_W * dv_dim / ap_dim
    elif axis == 1:  # coronal:    cols = LR, rows = DV
        h = _VIEW_W * dv_dim / lr_dim
    else:  # horizontal: cols = LR, rows = AP
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

    # Drag-clockwise → plane-rotates-clockwise in each view. The plane is rotated
    # about the view's own atlas axis (index == self._axis); positive θ moves the
    # leading edge clockwise in each view's screen space.
    _ANGLE_SIGNS: ClassVar[dict[int, int]] = {0: +1, 1: -1, 2: +1}
    # Atlas axes addressed by each view's left/right and up/down buttons:
    #   view axis → (col_atlas_axis, row_atlas_axis)
    _TRANSLATE_AXES: ClassVar[dict[int, tuple[int, int]]] = {
        0: (1, 2),  # sagittal:   cols=AP, rows=DV
        1: (0, 2),  # coronal:    cols=LR, rows=DV
        2: (0, 1),  # horizontal: cols=LR, rows=AP
    }

    def __init__(
        self,
        axis: int,
        dims: tuple[int, int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._axis = axis
        self._dims = dims
        self._anchoring: list[float] | None = None
        self._view_h = _view_height(axis, dims)
        self._reverse_axis: bool = False
        # The project's slicing axis (QuickNII voxel index 0/1/2).  Controls
        # which translate/rotate steps get sign-flipped when the series is
        # reversed.
        self._interpolation_axis: int = 1

        # drag state
        self._drag_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_start_anchoring: list[float] | None = None
        # Last cursor polar angle (rad) about the center, for incremental rotation.
        self._drag_last_angle: float | None = None

        self.setStyleSheet("background: #1a1a1a;")

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Row 0, col 0: canvas (the actual atlas slice).
        self._canvas = _SliceCanvas()
        self._canvas.setFixedSize(_VIEW_W, self._view_h)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        self._canvas.pressed.connect(self._on_canvas_pressed)
        self._canvas.moved.connect(self._on_canvas_moved)
        self._canvas.released.connect(self._on_canvas_released)
        layout.addWidget(self._canvas, 0, 0)

        # Row 0, col 1: ↑↑ ↑ / ↓ ↓↓ side column.
        self._btn_up_fast = self._make_btn(
            "chevrons-up.svg", "Move up (10 voxels)", _SIDE_BTN_W, _SIDE_BTN_H
        )
        self._btn_up = self._make_btn(
            "chevron-up.svg", "Move up (1 voxel)", _SIDE_BTN_W, _SIDE_BTN_H
        )
        self._btn_down = self._make_btn(
            "chevron-down.svg", "Move down (1 voxel)", _SIDE_BTN_W, _SIDE_BTN_H
        )
        self._btn_down_fast = self._make_btn(
            "chevrons-down.svg", "Move down (10 voxels)", _SIDE_BTN_W, _SIDE_BTN_H
        )
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(2)
        side.addStretch()
        side.addWidget(self._btn_up_fast)
        side.addWidget(self._btn_up)
        side.addWidget(self._btn_down)
        side.addWidget(self._btn_down_fast)
        side.addStretch()
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(_SIDE_BTN_W)
        layout.addWidget(side_widget, 0, 1)

        # Row 1, col 0: «← ← ⟲ ⟳ → →» bottom row, centered under the canvas.
        self._btn_left_fast = self._make_btn(
            "chevrons-left.svg", "Move left (10 voxels)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_left = self._make_btn(
            "chevron-left.svg", "Move left (1 voxel)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_ccw = self._make_btn(
            "rotate-ccw.svg", "Rotate counter-clockwise (1°)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_cw = self._make_btn(
            "rotate-cw.svg", "Rotate clockwise (1°)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_right = self._make_btn(
            "chevron-right.svg", "Move right (1 voxel)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        self._btn_right_fast = self._make_btn(
            "chevrons-right.svg", "Move right (10 voxels)", _BOTTOM_BTN_W, _BOTTOM_BTN_H
        )
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(2)
        bottom.addStretch()
        for b in (
            self._btn_left_fast,
            self._btn_left,
            self._btn_ccw,
            self._btn_cw,
            self._btn_right,
            self._btn_right_fast,
        ):
            bottom.addWidget(b)
        bottom.addStretch()
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom)
        bottom_widget.setFixedHeight(_BOTTOM_BTN_H)
        layout.addWidget(bottom_widget, 1, 0)

        # Wire buttons → helpers.
        col_axis, row_axis = self._TRANSLATE_AXES[axis]
        self._btn_left_fast.clicked.connect(
            lambda: self._translate_step(col_axis, -_MOVE_STEP_FAST)
        )
        self._btn_left.clicked.connect(lambda: self._translate_step(col_axis, -_MOVE_STEP))
        self._btn_right.clicked.connect(lambda: self._translate_step(col_axis, +_MOVE_STEP))
        self._btn_right_fast.clicked.connect(
            lambda: self._translate_step(col_axis, +_MOVE_STEP_FAST)
        )
        self._btn_up_fast.clicked.connect(lambda: self._translate_step(row_axis, -_MOVE_STEP_FAST))
        self._btn_up.clicked.connect(lambda: self._translate_step(row_axis, -_MOVE_STEP))
        self._btn_down.clicked.connect(lambda: self._translate_step(row_axis, +_MOVE_STEP))
        self._btn_down_fast.clicked.connect(
            lambda: self._translate_step(row_axis, +_MOVE_STEP_FAST)
        )
        self._btn_ccw.clicked.connect(lambda: self._rotate_step(-_ROTATE_STEP_DEG))
        self._btn_cw.clicked.connect(lambda: self._rotate_step(+_ROTATE_STEP_DEG))

        self.set_buttons_enabled(False)

        total_w = _VIEW_W + _SIDE_BTN_W + 4
        total_h = self._view_h + _BOTTOM_BTN_H + 6
        self.setFixedSize(total_w, total_h)

        self._base_pixmap: QPixmap | None = None
        # Cut-plane line: two display-space endpoints, each paired with its end letter.
        self._line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._line_labels: tuple[tuple[str, tuple[float, float]], ...] = ()
        self._center_display: tuple[float, float] | None = None

    def _make_btn(self, icon_name: str, tooltip: str, w: int, h: int) -> QPushButton:
        btn = QPushButton()
        btn.setIcon(colored_icon(icon_name, "#ffffff"))
        btn.setIconSize(QSize(w - 2, h - 2))
        btn.setFixedSize(w, h)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(_NAV_BTN_QSS)
        return btn

    # ------------------------------------------------------------------
    # Public update methods

    def set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            self._btn_up_fast,
            self._btn_up,
            self._btn_down,
            self._btn_down_fast,
            self._btn_left_fast,
            self._btn_left,
            self._btn_right,
            self._btn_right_fast,
            self._btn_ccw,
            self._btn_cw,
        ):
            btn.setEnabled(enabled)

    def set_reverse_axis(self, reverse: bool) -> None:
        self._reverse_axis = reverse

    def set_interpolation_axis(self, axis: int) -> None:
        self._interpolation_axis = int(axis)

    def update_dims(self, dims: tuple[int, int, int]) -> None:
        """Update atlas dimensions and resize the widget to preserve proportions."""
        self._dims = dims
        new_h = _view_height(self._axis, dims)
        if new_h != self._view_h:
            self._view_h = new_h
            self._canvas.setFixedSize(_VIEW_W, self._view_h)
            total_w = _VIEW_W + _SIDE_BTN_W + 4
            total_h = self._view_h + _BOTTOM_BTN_H + 6
            self.setFixedSize(total_w, total_h)

    def set_image(self, rgb: np.ndarray) -> None:
        qimg = _ndarray_to_qimage(rgb)
        self._base_pixmap = QPixmap.fromImage(qimg).scaled(
            _VIEW_W,
            self._view_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._redraw()

    def set_cut(self, anchoring: list[float]) -> None:
        self._anchoring = anchoring
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        center = o + u / 2.0 + v / 2.0

        # The cut plane is edge-on in this view, so it reads as a line. Draw the
        # plane's midline along whichever spanning vector lies in the view (the
        # one most perpendicular to the into-screen axis — which is the atlas
        # axis whose index equals this view's axis).
        in_plane = u if abs(u[self._axis]) < abs(v[self._axis]) else v
        p_lo = center - in_plane / 2.0
        p_hi = center + in_plane / 2.0

        # End letters come from the atlas axis the line runs along; order the
        # endpoints by that coordinate so the "low" letter sits at the low end.
        dom = int(np.argmax(np.abs(in_plane)))
        lo_letter, hi_letter = _AXIS_LETTERS[dom]
        if p_lo[dom] > p_hi[dom]:
            p_lo, p_hi = p_hi, p_lo

        d_lo = self._proj(p_lo)
        d_hi = self._proj(p_hi)
        self._line = (d_lo, d_hi)
        self._line_labels = ((lo_letter, d_lo), (hi_letter, d_hi))
        self._center_display = self._proj(center)
        self._redraw()

    # ------------------------------------------------------------------
    # Button-driven translate / rotate helpers

    def _translate_step(self, atlas_axis: int, delta: float) -> None:
        """Translate cut-plane origin along an atlas axis by *delta* voxels."""
        if self._anchoring is None:
            return
        if atlas_axis == self._interpolation_axis and self._reverse_axis:
            delta = -delta
        new_anchoring = list(self._anchoring)
        new_anchoring[atlas_axis] += delta
        self.anchoring_changed.emit(new_anchoring)

    def _rotate_step(self, deg_signed: float) -> None:
        """Rotate cut plane around this view's perpendicular atlas axis."""
        if self._anchoring is None:
            return
        deg = deg_signed * self._ANGLE_SIGNS[self._axis]
        if self._reverse_axis and self._axis != self._interpolation_axis:
            deg = -deg
        deg = clamp_rotation_to_max_tilt(
            self._anchoring, self._axis, deg, self._interpolation_axis, _MAX_TILT_DEG
        )
        new_anchoring = tilt_plane_about_atlas_axis(self._anchoring, self._axis, deg)
        self.anchoring_changed.emit(new_anchoring)

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

        if self._line is not None:
            (x0, y0), (x1, y1) = self._line
            painter.setPen(QPen(_OUTLINE_COLOR, 1.5))
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
            self._draw_end_labels(painter, x0, y0, x1, y1)

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

    def _draw_end_labels(
        self, painter: QPainter, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        """Draw the anatomical end letters just inside each line endpoint.

        The line usually spans the whole frame, so letters are nudged inward
        (toward the line center) and offset perpendicular, then clamped inside
        the canvas so they never clip at the edges.
        """
        if not self._line_labels:
            return
        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(_LABEL_COLOR, 1))

        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        # Perpendicular unit vector for a small sideways offset.
        nx, ny = -dy / length, dx / length

        for letter, (ex, ey) in self._line_labels:
            inx, iny = cx - ex, cy - ey
            inl = math.hypot(inx, iny) or 1.0
            px = ex + inx / inl * 11.0 + nx * 7.0
            py = ey + iny / inl * 11.0 + ny * 7.0
            px = min(max(px, 3.0), _VIEW_W - 9.0)
            py = min(max(py, 11.0), self._view_h - 3.0)
            painter.drawText(QPointF(px, py), letter)

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
        # For translate: apply immediately so a single click moves the plane.
        # For rotate: seed the incremental angle from the press point.
        if self._drag_mode == "translate":
            self._handle_translate(cx, cy)
        elif self._center_display is not None:
            ccx, ccy = self._center_display
            self._drag_last_angle = math.atan2(cy - ccy, cx - ccx)

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
        self._drag_last_angle = None

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
        if self._axis == 0:  # sagittal: move AP and DV
            new_center[1] = cur_atlas[1]
            new_center[2] = cur_atlas[2]
        elif self._axis == 1:  # coronal: move LR and DV
            new_center[0] = cur_atlas[0]
            new_center[2] = cur_atlas[2]
        else:  # horizontal: move LR and AP
            new_center[0] = cur_atlas[0]
            new_center[1] = cur_atlas[1]

        new_o = new_center - u / 2.0 - v / 2.0
        self.anchoring_changed.emit(new_o.tolist() + anchoring[3:])

    def _handle_rotate(self, cx: float, cy: float) -> None:
        if self._anchoring is None or self._center_display is None or self._drag_last_angle is None:
            return
        ccx, ccy = self._center_display
        cur_angle = math.atan2(cy - ccy, cx - ccx)
        # Incremental step since the last cursor position, wrapped into
        # (-180°, 180°]. Applying the small step to the *current* anchoring (not
        # the press-time one) means sweeping the cursor past a half-turn just
        # accumulates smoothly instead of snapping to the opposite rotation, and
        # the tilt clamp simply stops further travel at the limit.
        d = (cur_angle - self._drag_last_angle + math.pi) % (2.0 * math.pi) - math.pi
        self._drag_last_angle = cur_angle
        deg = math.degrees(d) * self._ANGLE_SIGNS[self._axis]
        # The two views that aren't perpendicular to the slicing axis rotate
        # around axes that tilt the plane along the slicing direction —
        # invert when the series is reversed.
        if self._reverse_axis and self._axis != self._interpolation_axis:
            deg = -deg

        deg = clamp_rotation_to_max_tilt(
            self._anchoring, self._axis, deg, self._interpolation_axis, _MAX_TILT_DEG
        )
        new_anchoring = tilt_plane_about_atlas_axis(self._anchoring, self._axis, deg)
        self.anchoring_changed.emit(new_anchoring)


class NavigatorPanel(QWidget):
    """Scrollable vertical stack of three orthogonal atlas slice views."""

    anchoring_changed = pyqtSignal(list)
    scale_requested = pyqtSignal(float, float)  # scale_u, scale_v

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atlas: AtlasVolume | None = None
        self._anchoring: list[float] | None = None
        # Width budget: the fixed-size _SliceView, plus the content-layout and
        # group-box margins around it.  The vertical scrollbar's width is added
        # only while the bar is actually visible (see eventFilter), so the panel
        # widens by exactly the bar when it appears and reclaims that space when
        # it doesn't — instead of permanently reserving an empty slot.
        self._sb_extent = require(QApplication.style()).pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent
        )
        view_w = _VIEW_W + _SIDE_BTN_W + 4  # _SliceView total width
        margins = (2 + 2) + (4 + 4) + 2  # content + group margins + frame
        self._width_no_sb = view_w + margins
        self.setFixedWidth(self._width_no_sb)
        # Panel, scroll area and content inherit the app dark palette so the
        # group boxes match the properties panel.  Only the atlas canvases and
        # their immediate container (_SliceView) keep the darker #1a1a1a.

        # Outer layout holds a scroll area so the tall horizontal view is reachable
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll)

        # Grow the panel by the scrollbar's width only while the bar is shown.
        self._vbar = require(scroll.verticalScrollBar())
        self._vbar.installEventFilter(self)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)

        self._stretch_btns: list[QPushButton] = []
        layout.addWidget(self._make_stretch_section())

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #333; border: none;")
        layout.addWidget(sep)

        dims = (528, 320, 456)
        self._sag = _SliceView(0, dims)
        self._cor = _SliceView(1, dims)
        self._hor = _SliceView(2, dims)

        # Group box per axis, so the view parallel to the slices (the one whose
        # axis is the slicing axis) can be hidden once the project axis is known.
        self._groups: dict[int, QGroupBox] = {}
        self._hidden_axis: int | None = None
        view_titles = ["Sagittal", "Coronal", "Horizontal"]
        for view, title in zip((self._sag, self._cor, self._hor), view_titles, strict=False):
            view.anchoring_changed.connect(self._on_anchoring_changed)
            grp = QGroupBox(title)
            grp.setStyleSheet(_VIEW_GROUP_QSS)
            # Only take as much height as the canvas + buttons need; slack goes
            # to the trailing stretch instead of inflating each group.
            grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            grp_layout = QVBoxLayout(grp)
            grp_layout.setContentsMargins(4, 4, 4, 4)
            grp_layout.setSpacing(0)
            grp_layout.addWidget(view)
            layout.addWidget(grp)
            self._groups[view._axis] = grp

        layout.addStretch()

    def eventFilter(self, obj, event) -> bool:
        # Track the vertical scrollbar's visibility and resize the panel so its
        # width only accounts for the bar while it's on screen.  Changing width
        # never alters content height, so this can't oscillate.
        if obj is self._vbar:
            if event.type() == QEvent.Type.Show:
                self.setFixedWidth(self._width_no_sb + self._sb_extent)
            elif event.type() == QEvent.Type.Hide:
                self.setFixedWidth(self._width_no_sb)
        return super().eventFilter(obj, event)

    def _make_stretch_section(self) -> QGroupBox:
        # Match the navigator button style exactly, adding text colour
        btn_qss = (
            "QPushButton { border-radius: 3px; padding: 0px; color: #ccc;"
            " background: #383838; border: 1px solid #555; }"
            "QPushButton:hover { background: #484848; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #333; }"
            + _TOOLTIP_QSS
        )
        group = QGroupBox("Atlas Stretch")
        group.setStyleSheet(_VIEW_GROUP_QSS)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(4, 6, 4, 6)
        vbox.setSpacing(4)

        rows = [
            (
                "chevrons-left-right-ellipsis.svg",
                [
                    ("−−", "Narrower (10%)", _SCALE_STEP_FAST, 1.0),
                    ("−", "Narrower (2%)", _SCALE_STEP, 1.0),
                    ("+", "Wider (2%)", 1.0 / _SCALE_STEP, 1.0),
                    ("++", "Wider (10%)", 1.0 / _SCALE_STEP_FAST, 1.0),
                ],
            ),
            (
                "chevrons-up-down-ellipsis.svg",
                [
                    ("−−", "Shorter (10%)", 1.0, _SCALE_STEP_FAST),
                    ("−", "Shorter (2%)", 1.0, _SCALE_STEP),
                    ("+", "Taller (2%)", 1.0, 1.0 / _SCALE_STEP),
                    ("++", "Taller (10%)", 1.0, 1.0 / _SCALE_STEP_FAST),
                ],
            ),
        ]

        for icon_name, specs in rows:
            # Stretches on both ends centre the «−− − [icon] + ++» cluster
            # horizontally across the full group width.
            inner = QHBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(3)
            inner.addStretch()

            for sym, tip, su, sv in specs[:2]:
                btn = QPushButton(sym)
                btn.setFixedSize(_BOTTOM_BTN_W + 6, _BOTTOM_BTN_H + 2)
                btn.setToolTip(tip)
                btn.setStyleSheet(btn_qss)
                btn.setEnabled(False)
                btn.clicked.connect(lambda _, s=su, t=sv: self.scale_requested.emit(s, t))
                inner.addWidget(btn)
                self._stretch_btns.append(btn)

            icon_lbl = QLabel()
            icon_lbl.setPixmap(colored_icon(icon_name, "#ffffff").pixmap(QSize(16, 16)))
            icon_lbl.setFixedSize(18, 18)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(icon_lbl)

            for sym, tip, su, sv in specs[2:]:
                btn = QPushButton(sym)
                btn.setFixedSize(_BOTTOM_BTN_W + 8, _BOTTOM_BTN_H + 4)
                btn.setToolTip(tip)
                btn.setStyleSheet(btn_qss)
                btn.setEnabled(False)
                btn.clicked.connect(lambda _, s=su, t=sv: self.scale_requested.emit(s, t))
                inner.addWidget(btn)
                self._stretch_btns.append(btn)

            inner.addStretch()

            row_w = QWidget()
            row_w.setLayout(inner)
            vbox.addWidget(row_w)

        return group

    def set_stretch_enabled(self, enabled: bool) -> None:
        for btn in self._stretch_btns:
            btn.setEnabled(enabled)

    def set_reverse_axis(self, reverse: bool) -> None:
        for view in (self._sag, self._cor, self._hor):
            view.set_reverse_axis(reverse)

    def set_interpolation_axis(self, axis: int) -> None:
        for view in (self._sag, self._cor, self._hor):
            view.set_interpolation_axis(axis)
        # Hide the view parallel to the slices (face-on); the two edge-on views
        # that reveal tilt remain. In-plane rotation is handled on the canvas.
        self._hidden_axis = axis
        for ax, grp in self._groups.items():
            grp.setVisible(ax != axis)

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
        idx_for = {0: round(center[0]), 1: round(center[1]), 2: round(center[2])}
        views_by_axis = {0: self._sag, 1: self._cor, 2: self._hor}

        for ax, view in views_by_axis.items():
            if ax == self._hidden_axis:
                continue
            view.set_image(self._atlas.get_orthogonal_slice(ax, idx_for[ax]))
            view.set_buttons_enabled(True)
            view.set_cut(self._anchoring)

    def _on_anchoring_changed(self, new_anchoring: list[float]) -> None:
        self._anchoring = new_anchoring
        self._refresh_images()
        self.anchoring_changed.emit(new_anchoring)
