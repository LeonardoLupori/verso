"""Interactive editor for the L/R hemisphere separating line.

Owns a pyqtgraph :class:`LineSegmentROI` plus two ``TextItem`` badges ("L"
and "R") that float on each side of the line.  Used by PrepView during the
"Draw separating line" mode.

L/R sides are determined geometrically from the start→end direction of the
line via the cross product ``(p1-p0) x (q-p0)``.  Dragging the start handle
past the end handle reverses the direction and swaps L/R automatically — no
explicit "swap" button needed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PyQt6.QtCore import QObject, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPolygonItem

from verso.engine.preprocessing import line_side_polygons

if TYPE_CHECKING:
    from verso.gui.widgets.canvas import ImageCanvas


_LEFT_COLOR = (220, 60, 60)   # red — matches lr_mask_to_rgba default
_RIGHT_COLOR = (60, 130, 220) # blue — matches lr_mask_to_rgba default
_LINE_COLOR = (255, 220, 60)  # yellow — visible against any tissue colour
_BADGE_OFFSET_PX = 70         # perpendicular distance from line midpoint
_BADGE_FONT_PT = 16
_HOVER_TINT_ALPHA = 80       # 0–255 alpha for the hover-side polygon fill


class LRLineEditor(QObject):
    """Owns the draggable separating line and its L/R badges.

    Lifecycle::

        editor = LRLineEditor(canvas)
        editor.begin((x0, y0), (x1, y1), image_w, image_h)
        # ... user drags handles, line moves, badges follow ...
        eps = editor.endpoints()        # → ((x0, y0), (x1, y1)) in image px
        editor.end()                    # remove items from the plot

    Emits :pyattr:`endpoints_changed` after every drag tick.
    """

    endpoints_changed = pyqtSignal(tuple, tuple)

    def __init__(self, canvas: ImageCanvas) -> None:
        super().__init__()
        self._canvas = canvas
        self._roi: pg.LineSegmentROI | None = None
        self._left_badge: pg.TextItem | None = None
        self._right_badge: pg.TextItem | None = None
        self._hover_poly: QGraphicsPolygonItem | None = None
        self._last_mouse_view: tuple[float, float] | None = None
        self._image_w = 0
        self._image_h = 0

    # ------------------------------------------------------------------
    # Public API

    def begin(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        image_w: int,
        image_h: int,
        left_color: tuple[int, int, int] = _LEFT_COLOR,
        right_color: tuple[int, int, int] = _RIGHT_COLOR,
    ) -> None:
        """Add the line and badges to the canvas at the given endpoints."""
        self._left_color = left_color
        self._right_color = right_color
        if self._roi is not None:
            self.end()
        self._image_w = int(image_w)
        self._image_h = int(image_h)

        self._roi = pg.LineSegmentROI(
            [p0, p1],
            pen=pg.mkPen(_LINE_COLOR, width=2),
            removable=False,
        )
        self._roi.setZValue(25)
        # Lock down extras we don't want (rotation handles, scaling, etc).
        self._roi.translatable = True
        self._roi.rotatable = False
        self._roi.resizable = False

        # Style endpoint handles: yellow circles matching the line colour.
        _hr = 7  # handle radius in pixels
        _circle_path = QPainterPath()
        _circle_path.addEllipse(-_hr, -_hr, 2 * _hr, 2 * _hr)
        handles = self._roi.getHandles()
        if len(handles) >= 2:
            for h in handles[:2]:
                h.pen = pg.mkPen(_LINE_COLOR, width=2)
                h.currentPen = h.pen
                h.brush = QBrush(QColor(*_LINE_COLOR, 160))
                h.path = _circle_path
                h.radius = _hr

        font = QFont()
        font.setPointSize(_BADGE_FONT_PT)
        font.setBold(True)
        self._left_badge = pg.TextItem("L", color=self._left_color, anchor=(0.5, 0.5))
        self._left_badge.setFont(font)
        self._left_badge.setZValue(26)
        self._right_badge = pg.TextItem("R", color=self._right_color, anchor=(0.5, 0.5))
        self._right_badge.setFont(font)
        self._right_badge.setZValue(26)

        # Hover-side tint polygon — single item, recolored on the fly to
        # match whichever side the mouse is over.
        self._hover_poly = QGraphicsPolygonItem()
        self._hover_poly.setZValue(24)  # below the line itself (line z=25)
        self._hover_poly.setPen(QPen(QColor(0, 0, 0, 0)))  # no border
        self._hover_poly.setVisible(False)

        self._canvas.plot.addItem(self._roi)
        self._canvas.plot.addItem(self._hover_poly)
        self._canvas.plot.addItem(self._left_badge)
        self._canvas.plot.addItem(self._right_badge)

        self._roi.sigRegionChanged.connect(self._on_roi_changed)
        self._canvas.plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)
        self._update_badges(p0, p1)

    def end(self) -> None:
        """Remove all editor items from the canvas. Safe to call twice."""
        if self._roi is not None:
            try:
                self._roi.sigRegionChanged.disconnect(self._on_roi_changed)
            except (TypeError, RuntimeError):
                pass
            try:
                self._canvas.plot.scene().sigMouseMoved.disconnect(
                    self._on_scene_mouse_moved
                )
            except (TypeError, RuntimeError):
                pass
            self._canvas.plot.removeItem(self._roi)
            self._roi = None
        for attr in ("_hover_poly", "_left_badge", "_right_badge"):
            item = getattr(self, attr)
            if item is not None:
                self._canvas.plot.removeItem(item)
                setattr(self, attr, None)
        self._last_mouse_view = None

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return the current line endpoints in image-pixel coords."""
        if self._roi is None:
            return None
        handles = self._roi.getHandles()
        if len(handles) < 2:
            return None
        pts: list[tuple[float, float]] = []
        for h in handles[:2]:
            view_pos = self._roi.mapToView(h.pos())
            pts.append((float(view_pos.x()), float(view_pos.y())))
        return pts[0], pts[1]

    # ------------------------------------------------------------------
    # Internal

    def _on_roi_changed(self) -> None:
        eps = self.endpoints()
        if eps is None:
            return
        p0, p1 = eps
        self._update_badges(p0, p1)
        self._refresh_hover_tint()
        self.endpoints_changed.emit(p0, p1)

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        if self._roi is None:
            return
        view = self._canvas.plot.getViewBox()
        view_pt = view.mapSceneToView(scene_pos)
        self._last_mouse_view = (float(view_pt.x()), float(view_pt.y()))
        self._refresh_hover_tint()

    def _refresh_hover_tint(self) -> None:
        if self._hover_poly is None or self._last_mouse_view is None:
            return
        qx, qy = self._last_mouse_view
        # Hide when mouse is outside the image rect.
        if not (0.0 <= qx <= self._image_w and 0.0 <= qy <= self._image_h):
            self._hover_poly.setVisible(False)
            return
        eps = self.endpoints()
        if eps is None:
            self._hover_poly.setVisible(False)
            return
        p0, p1 = eps
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            self._hover_poly.setVisible(False)
            return
        cross = dx * (qy - p0[1]) - dy * (qx - p0[0])
        left_poly, right_poly = line_side_polygons(
            p0, p1, float(self._image_w), float(self._image_h)
        )
        # cross < 0 → left (red); cross ≥ 0 → right (blue)
        if cross < 0.0:
            poly_arr = left_poly
            color = self._left_color
        else:
            poly_arr = right_poly
            color = self._right_color
        if len(poly_arr) < 3:
            self._hover_poly.setVisible(False)
            return
        qpoly = QPolygonF([QPointF(float(x), float(y)) for x, y in poly_arr])
        self._hover_poly.setPolygon(qpoly)
        self._hover_poly.setBrush(
            QBrush(QColor(color[0], color[1], color[2], _HOVER_TINT_ALPHA))
        )
        self._hover_poly.setVisible(True)

    def _update_badges(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
    ) -> None:
        """Position the L/R badges on their respective perpendicular sides.

        The "left" perpendicular direction (cross < 0 side) is ``(dy, -dx)``
        when the line goes p0→p1; the "right" direction is the opposite.
        """
        if self._left_badge is None or self._right_badge is None:
            return
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            # Degenerate (zero-length) line — just stack badges at the midpoint.
            self._left_badge.setPos(mx, my)
            self._right_badge.setPos(mx, my)
            return
        lx = mx + _BADGE_OFFSET_PX * dy / length
        ly = my - _BADGE_OFFSET_PX * dx / length
        rx = mx - _BADGE_OFFSET_PX * dy / length
        ry = my + _BADGE_OFFSET_PX * dx / length
        self._left_badge.setPos(lx, ly)
        self._right_badge.setPos(rx, ry)
