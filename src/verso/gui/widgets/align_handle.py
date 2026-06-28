"""Centre manipulator gizmo for the Align view's atlas overlay.

``AlignHandle`` is a self-contained pyqtgraph item: it paints a centre gizmo
(rotation ring, four N/E/S/W stretch arrows, and a centre dot) at a fixed
on-screen size and classifies cursor positions into drag zones via
:meth:`AlignHandle.zone_at`.  It performs no dragging itself — the canvas's
overlay view-box reads the zone to route a gesture (ring → rotate, arrow grips →
stretch, elsewhere → translate) and feeds the hovered zone back to
:meth:`AlignHandle.set_hover_zone` so the active element brightens.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem

# Align handle geometry (in screen pixels) and opacity.
_HANDLE_GRIP_PX = 30.0  # distance of the arrowhead stretch grips along each axis
_HANDLE_GRIP_HALF = 8.0  # half-size of a stretch grip's square hit box
_HANDLE_ARROW_LEN = 9.0  # arrowhead length (along its axis)
_HANDLE_ARROW_HALF = 7.0  # arrowhead half-width (across its axis)
_HANDLE_CENTER_DOT_PX = 6.0  # inert centre dot radius (visual marker only)
_HANDLE_RING_PX = 42.0  # rotation ring radius
_HANDLE_RING_HALF = 9.0  # half-width of the ring's grab band
_HANDLE_DIM = 0.35  # resting opacity
_HANDLE_OPAQUE = 1.0  # hovered opacity


class AlignHandle(pg.GraphicsObject):
    """Centre manipulator drawn over the atlas overlay in the Align view.

    Painted at the overlay centre at a fixed on-screen size (it ignores the
    view transform), it shows a crosshair inside an outer ring.  It is purely a
    visual + hit-test helper: the canvas's overlay view-box reads :meth:`zone_at`
    to route a drag — inner crosshair → translate, outer ring → rotate — and the
    canvas feeds the hovered zone to :meth:`set_hover_zone` so only the element
    under the cursor (the ring, or a stretch-arrow pair) brightens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(18)
        # Which element is hovered: "rotate", "stretch_x", "stretch_y", or None.
        # Per-element opacity is applied in paint(); the item's own opacity is 1.
        self._hover_zone: str | None = None
        self.setVisible(False)

    def boundingRect(self) -> QRectF:
        r = _HANDLE_RING_PX + 2.0
        return QRectF(-r, -r, 2.0 * r, 2.0 * r)

    def set_center(self, x: float, y: float) -> None:
        """Place the handle at ``(x, y)`` in view/image-pixel coordinates."""
        self.setPos(x, y)

    def set_hover_zone(self, zone: str | None) -> None:
        """Brighten the hovered element only.

        ``zone`` comes from :meth:`zone_at`. Only the rotation ring and the two
        stretch-arrow pairs have a brightened state; ``"translate"`` (the broad
        whole-canvas field) and the inert centre dot are treated as ``None`` so
        nothing brightens and redundant repaints are skipped.
        """
        if zone not in ("rotate", "stretch_x", "stretch_y"):
            zone = None
        if zone == self._hover_zone:
            return
        self._hover_zone = zone
        self.update()

    def set_active(self, active: bool) -> None:
        """Show/hide the handle (Align mode with an overlay present)."""
        self.setVisible(active)
        if not active:
            self.set_hover_zone(None)

    def zone_at(self, view_x: float, view_y: float, view_px: float) -> str | None:
        """Classify a view-space point relative to the handle.

        Returns ``"stretch_x"``/``"stretch_y"`` over an arrowhead grip,
        ``"rotate"`` over the ring's grab band, ``"translate"`` anywhere else
        (the default drag), or ``None`` when the handle is hidden.  ``view_px``
        is view units per screen pixel (``ViewBox.viewPixelSize()[0]``); pixel
        thresholds are scaled by it so hit-testing matches the painted,
        screen-fixed size at any zoom level.
        """
        if not self.isVisible() or view_px <= 0:
            return None
        c = self.pos()
        px = (view_x - c.x()) / view_px
        py = (view_y - c.y()) / view_px
        # Stretch grips sit on each axis at _HANDLE_GRIP_PX; their square hit
        # boxes take priority over the rotation ring band they overlap.
        if abs(py) <= _HANDLE_GRIP_HALF and abs(abs(px) - _HANDLE_GRIP_PX) <= _HANDLE_GRIP_HALF:
            return "stretch_x"
        if abs(px) <= _HANDLE_GRIP_HALF and abs(abs(py) - _HANDLE_GRIP_PX) <= _HANDLE_GRIP_HALF:
            return "stretch_y"
        # Rotate only on the ring line itself; everything else translates.
        if abs(math.hypot(px, py) - _HANDLE_RING_PX) <= _HANDLE_RING_HALF:
            return "rotate"
        return "translate"

    def paint(self, painter: QPainter, *_) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        gray = QColor(200, 200, 200)
        halo = QColor(0, 0, 0, 160)
        g = _HANDLE_GRIP_PX
        hl = _HANDLE_ARROW_LEN
        hw = _HANDLE_ARROW_HALF
        zone = self._hover_zone

        def alpha_for(z: str) -> float:
            # Per-element opacity: the hovered element is opaque, the rest dim.
            return _HANDLE_OPAQUE if zone == z else _HANDLE_DIM

        # Outer rotation ring (gray, with a dark halo for contrast)
        painter.setOpacity(alpha_for("rotate"))
        painter.setPen(QPen(halo, 5.0))
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_RING_PX, _HANDLE_RING_PX)
        painter.setPen(QPen(gray, 3.0))
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_RING_PX, _HANDLE_RING_PX)
        # Arrowhead stretch grips at N/E/S/W, each pointing outward along its
        # axis. The two grips on an axis share a drag, so they brighten together.
        painter.setPen(QPen(halo, 1.5))
        painter.setBrush(gray)
        for axis_zone, dirs in (
            ("stretch_x", ((1.0, 0.0), (-1.0, 0.0))),
            ("stretch_y", ((0.0, 1.0), (0.0, -1.0))),
        ):
            painter.setOpacity(alpha_for(axis_zone))
            for dx, dy in dirs:
                tip = QPointF(g * dx, g * dy)
                bx, by = (g - hl) * dx, (g - hl) * dy
                px, py = -dy, dx  # unit perpendicular to the axis
                c1 = QPointF(bx + hw * px, by + hw * py)
                c2 = QPointF(bx - hw * px, by - hw * py)
                painter.drawPolygon(QPolygonF([tip, c1, c2]))
        # Centre dot — brightens with whichever element is hovered so it always
        # reads as part of the active group; dim when nothing is hovered.
        painter.setOpacity(_HANDLE_OPAQUE if zone is not None else _HANDLE_DIM)
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_CENTER_DOT_PX, _HANDLE_CENTER_DOT_PX)
        painter.setBrush(Qt.BrushStyle.NoBrush)
