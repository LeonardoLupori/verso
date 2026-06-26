"""Shared pyqtgraph image canvas used by Prep and Align/Warp views.

Item stack (low z to high):
  channel_items[i] — one ImageItem per section channel, each holding the raw
                     uint8 plane plus a per-channel 256x4 RGBA LUT.  Composited
                     together with CompositionMode_Lighten (component-wise max),
                     which is the GPU equivalent of np.maximum.reduce.
  overlay_item     — atlas overlay (z=10), normal SourceOver alpha blend.
  disp_halo/disp   — warp displacement lines (z=14, 15).
  cp_item          — warp control points (z=20).
  stroke_item      — live freehand mask preview (z=30).

Align handle: in Align mode a centre gizmo (``_AlignHandle``) is drawn over the
overlay.  Dragging the N/E/S/W arrowhead grips emits
``overlay_scaled(scale_s, scale_t)`` to stretch width/height; dragging the ring
emits ``overlay_rotated(deg)`` to spin it in-plane; dragging anywhere else emits
``overlay_panned(dx, dy)`` to translate the cut plane.  The centre dot is inert.
Holding the spacebar while dragging pans the view instead.  The handle is hidden
outside Align mode and when no overlay is present.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QGraphicsItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from verso.gui.widgets.orientation_overlay import OrientationOverlay

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
# Stretch sensitivity: per-event scale ratios are raised to this power before
# being applied. <1 makes grip dragging slower (the ratios compound across move
# events, so this scales overall sensitivity).
_STRETCH_GAIN = 0.5


def _make_cross_cursor(rgb: tuple[int, int, int], size: int = 21) -> QCursor:
    """Build a 1-px colored crosshair with a 1-px black outline, hotspot at center."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    mid = size // 2
    painter.setPen(QPen(QColor(0, 0, 0), 3))
    painter.drawLine(1, mid, size - 2, mid)
    painter.drawLine(mid, 1, mid, size - 2)
    painter.setPen(QPen(QColor(*rgb), 1))
    painter.drawLine(1, mid, size - 2, mid)
    painter.drawLine(mid, 1, mid, size - 2)
    painter.end()
    return QCursor(pm, mid, mid)


def _make_circle_cursor(rgb: tuple[int, int, int], diameter_px: int) -> QCursor:
    """Build a 1-px colored circle with a 1-px black outline, hotspot at center.

    ``diameter_px`` is the on-screen diameter; it is clamped to a sane range so
    huge brushes at high zoom don't create an unusable pixmap.
    """
    d = int(min(max(diameter_px, 3), 1024))
    pad = 2
    sz = d + 2 * pad + 1
    pm = QPixmap(sz, sz)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(0, 0, 0), 3))
    painter.drawEllipse(pad, pad, d, d)
    painter.setPen(QPen(QColor(*rgb), 1))
    painter.drawEllipse(pad, pad, d, d)
    mid = sz // 2
    painter.setPen(QPen(QColor(0, 0, 0), 3))
    painter.drawPoint(mid, mid)
    painter.setPen(QPen(QColor(*rgb), 1))
    painter.drawPoint(mid, mid)
    painter.end()
    return QCursor(pm, mid, mid)


# lucide "rotate-ccw" glyph (circled arrow), parameterised by render size/colour/stroke.
_ROTATE_CCW_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
    'viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="{w}" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
    '<path d="M3 3v5h5"/></svg>'
)


def _make_rotate_cursor(rgb: tuple[int, int, int], size: int = 24) -> QCursor:
    """Build a circled-arrow rotation cursor from the lucide ``rotate-ccw`` glyph.

    Rendered with a dark halo pass under the coloured stroke for contrast, hotspot
    at the pixmap centre.
    """
    hexc = "#{:02x}{:02x}{:02x}".format(*rgb)
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    for color, width in (("#000000", 3.5), (hexc, 2.0)):
        layer = QPixmap()
        layer.loadFromData(_ROTATE_CCW_SVG.format(s=size, c=color, w=width).encode())
        painter.drawPixmap(0, 0, layer)
    painter.end()
    h = size // 2
    return QCursor(pm, h, h)


# ---------------------------------------------------------------------------
# Application-level space-key tracker (singleton, installed once)
# ---------------------------------------------------------------------------


class _SpaceState:
    held: bool = False
    # ImageCanvas instances that want to be notified on Space change.
    listeners: set = set()


class _ShiftState:
    held: bool = False
    # ImageCanvas instances that want to be notified on Shift change.
    listeners: set = set()


class _SpaceFilter(QObject):
    """Application event filter that tracks whether the spacebar is held.

    Also tracks Shift for prep-mode cursor color so it stays synced even when
    keyboard focus is on another widget (properties panel, main window, etc).
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                _SpaceState.held = True
                for canvas in list(_SpaceState.listeners):
                    canvas._on_space_changed()
                # Consume the event when a button has focus so spacebar doesn't
                # re-trigger the last clicked button while panning.
                if isinstance(QApplication.focusWidget(), QAbstractButton):
                    return True
            elif event.key() == Qt.Key.Key_Shift:
                _ShiftState.held = True
                for canvas in list(_ShiftState.listeners):
                    canvas._on_shift_changed()
        elif t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                _SpaceState.held = False
                for canvas in list(_SpaceState.listeners):
                    canvas._on_space_changed()
            elif event.key() == Qt.Key.Key_Shift:
                _ShiftState.held = False
                for canvas in list(_ShiftState.listeners):
                    canvas._on_shift_changed()
        return False


def _ensure_space_filter() -> None:
    app = QApplication.instance()
    if app is not None and not hasattr(app, "_verso_space_filter"):
        app._verso_space_filter = _SpaceFilter()
        app.installEventFilter(app._verso_space_filter)


# ---------------------------------------------------------------------------
# Custom ViewBox with overlay-pan support
# ---------------------------------------------------------------------------


class _OverlayViewBox(pg.ViewBox):
    """ViewBox that emits overlay_panned(dx, dy) when space is held during drag,
    and canvas_clicked / canvas_dragged for warp control-point interaction."""

    overlay_panned = pyqtSignal(float, float)
    # Emitted (degrees) while dragging the align handle's rotation ring
    overlay_rotated = pyqtSignal(float)
    # Emitted (scale_s, scale_t multipliers) while dragging a stretch grip
    overlay_scaled = pyqtSignal(float, float)
    # Emitted True/False as a spacebar view-pan drag starts/ends (grab cursor)
    space_pan_changed = pyqtSignal(bool)
    # Emitted in image-pixel coordinates
    canvas_clicked = pyqtSignal(float, float)  # single click (no drag)
    canvas_drag_started = pyqtSignal(float, float)  # drag begin
    canvas_dragged = pyqtSignal(float, float)  # drag update
    canvas_drag_ended = pyqtSignal(float, float)  # drag finish

    _InteractionMode = Literal["align", "warp", "prep", "view"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interaction_mode: _OverlayViewBox._InteractionMode = "align"
        self._align_handle: _AlignHandle | None = None

    def set_interaction_mode(self, mode: _InteractionMode) -> None:
        self._interaction_mode = mode

    def set_align_handle(self, handle: _AlignHandle) -> None:
        self._align_handle = handle

    def mouseClickEvent(self, ev) -> None:
        if ev.double() and ev.button() == Qt.MouseButton.LeftButton:
            self.autoRange()
            ev.accept()
            return
        if (
            self._interaction_mode in ("warp", "prep")
            and ev.button() == Qt.MouseButton.LeftButton
            and not _SpaceState.held
        ):
            pos = self.mapSceneToView(ev.scenePos())
            self.canvas_clicked.emit(pos.x(), pos.y())
            ev.accept()
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()  # suppress default right-click context menu
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if self._interaction_mode == "align" and ev.button() == Qt.MouseButton.LeftButton:
            # Spacebar turns the drag into a plain view pan ("padding"); without
            # it the gesture is interpreted by the handle zone it started in —
            # ring rotates, arrowhead grips stretch, anywhere else translates the
            # atlas overlay.
            if _SpaceState.held:
                if ev.isStart():
                    self.space_pan_changed.emit(True)
                elif ev.isFinish():
                    self.space_pan_changed.emit(False)
                super().mouseDragEvent(ev, axis)
                return
            handle = self._align_handle
            zone = None
            if handle is not None:
                start = self.mapSceneToView(ev.buttonDownScenePos())
                view_px = self.viewPixelSize()[0]
                zone = handle.zone_at(start.x(), start.y(), view_px)
            if zone == "rotate":
                ev.accept()
                c = handle.pos()
                p1 = self.mapSceneToView(ev.lastScenePos())
                p2 = self.mapSceneToView(ev.scenePos())
                a1 = math.atan2(p1.y() - c.y(), p1.x() - c.x())
                a2 = math.atan2(p2.y() - c.y(), p2.x() - c.x())
                self.overlay_rotated.emit(math.degrees(a2 - a1))
            elif zone in ("stretch_x", "stretch_y"):
                ev.accept()
                c = handle.pos()
                p1 = self.mapSceneToView(ev.lastScenePos())
                p2 = self.mapSceneToView(ev.scenePos())
                floor = max(view_px * 2.0, 1e-6)  # avoid blow-up near the centre
                if zone == "stretch_x":
                    d1 = max(abs(p1.x() - c.x()), floor)
                    d2 = max(abs(p2.x() - c.x()), floor)
                    # Inverse ratio: pulling the grip outward (d2 > d1) shrinks u,
                    # which widens the sampled overlay (scale_anchoring multiplies u).
                    ratio = (d1 / d2) ** _STRETCH_GAIN
                    self.overlay_scaled.emit(min(2.0, max(0.5, ratio)), 1.0)
                else:
                    d1 = max(abs(p1.y() - c.y()), floor)
                    d2 = max(abs(p2.y() - c.y()), floor)
                    ratio = (d1 / d2) ** _STRETCH_GAIN
                    self.overlay_scaled.emit(1.0, min(2.0, max(0.5, ratio)))
            else:
                # Default: drag anywhere else translates the atlas overlay.
                ev.accept()
                p1 = self.mapSceneToView(ev.lastScenePos())
                p2 = self.mapSceneToView(ev.scenePos())
                self.overlay_panned.emit(p2.x() - p1.x(), p2.y() - p1.y())
        elif (
            self._interaction_mode in ("warp", "prep")
            and not _SpaceState.held
            and ev.button() == Qt.MouseButton.LeftButton
        ):
            ev.accept()
            pos = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                down_pos = self.mapSceneToView(ev.buttonDownScenePos())
                self.canvas_drag_started.emit(down_pos.x(), down_pos.y())
            elif ev.isFinish():
                self.canvas_drag_ended.emit(pos.x(), pos.y())
            else:
                self.canvas_dragged.emit(pos.x(), pos.y())
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()  # suppress default right-drag zoom
        else:
            super().mouseDragEvent(ev, axis)


# ---------------------------------------------------------------------------
# Align centre handle
# ---------------------------------------------------------------------------


class _AlignHandle(pg.GraphicsObject):
    """Centre manipulator drawn over the atlas overlay in the Align view.

    Painted at the overlay centre at a fixed on-screen size (it ignores the
    view transform), it shows a crosshair inside an outer ring.  It is purely a
    visual + hit-test helper: the :class:`_OverlayViewBox` reads :meth:`zone_at`
    to route a drag — inner crosshair → translate, outer ring → rotate — and the
    canvas toggles :meth:`set_hovered` so it brightens when the cursor is over it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(18)
        self._hovered = False
        self.setOpacity(_HANDLE_DIM)
        self.setVisible(False)

    def boundingRect(self) -> QRectF:
        r = _HANDLE_RING_PX + 2.0
        return QRectF(-r, -r, 2.0 * r, 2.0 * r)

    def set_center(self, x: float, y: float) -> None:
        """Place the handle at ``(x, y)`` in view/image-pixel coordinates."""
        self.setPos(x, y)

    def set_hovered(self, hovered: bool) -> None:
        if hovered == self._hovered:
            return
        self._hovered = hovered
        self.setOpacity(_HANDLE_OPAQUE if hovered else _HANDLE_DIM)

    def set_active(self, active: bool) -> None:
        """Show/hide the handle (Align mode with an overlay present)."""
        self.setVisible(active)
        if not active:
            self.set_hovered(False)

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

    def paint(self, painter: QPainter, *args) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        gray = QColor(200, 200, 200)
        halo = QColor(0, 0, 0, 160)
        g = _HANDLE_GRIP_PX
        hl = _HANDLE_ARROW_LEN
        hw = _HANDLE_ARROW_HALF
        # Outer rotation ring (gray, with a dark halo for contrast)
        painter.setPen(QPen(halo, 5.0))
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_RING_PX, _HANDLE_RING_PX)
        painter.setPen(QPen(gray, 3.0))
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_RING_PX, _HANDLE_RING_PX)
        # Arrowhead stretch grips at N/E/S/W, each pointing outward along its axis
        painter.setPen(QPen(halo, 1.5))
        painter.setBrush(gray)
        for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            tip = QPointF(g * dx, g * dy)
            bx, by = (g - hl) * dx, (g - hl) * dy
            px, py = -dy, dx  # unit perpendicular to the axis
            c1 = QPointF(bx + hw * px, by + hw * py)
            c2 = QPointF(bx - hw * px, by - hw * py)
            painter.drawPolygon(QPolygonF([tip, c1, c2]))
        # Translate centre dot
        painter.drawEllipse(QPointF(0.0, 0.0), _HANDLE_CENTER_DOT_PX, _HANDLE_CENTER_DOT_PX)
        painter.setBrush(Qt.BrushStyle.NoBrush)


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------


class ImageCanvas(QWidget):
    """PyQtGraph canvas with a background and an optional semi-transparent overlay."""

    # Emitted while dragging the align handle's inner crosshair (dx, dy in image pixels)
    overlay_panned = pyqtSignal(float, float)
    # Emitted (degrees) while dragging the align handle's rotation ring
    overlay_rotated = pyqtSignal(float)
    # Emitted (scale_s, scale_t multipliers) while dragging an align stretch grip
    overlay_scaled = pyqtSignal(float, float)
    # Emitted on every mouse move over the canvas (x, y in scene/image pixel coords)
    mouse_position_changed = pyqtSignal(float, float)
    # Warp control-point interaction (image pixel coords)
    canvas_clicked = pyqtSignal(float, float)
    canvas_drag_started = pyqtSignal(float, float)
    canvas_dragged = pyqtSignal(float, float)
    canvas_drag_ended = pyqtSignal(float, float)
    # Alt+wheel over the canvas (raw Qt delta, ±120 per tick) for brush resize
    alt_wheel_scrolled = pyqtSignal(int)
    # View range (zoom/pan) changed — lets listeners re-render display-resolution
    # content (e.g. the atlas outline) so it stays ~1 screen-pixel wide.
    view_range_changed = pyqtSignal()

    _InteractionMode = Literal["align", "warp", "prep", "view"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _ensure_space_filter()
        self._channel_items: list[pg.ImageItem] = []
        self._channel_shape: tuple[int, int] | None = None
        self._interaction_mode: ImageCanvas._InteractionMode = "align"
        self._overlay_present: bool = False
        # Align-cursor state: last hovered handle zone + whether a spacebar pan
        # drag is currently in progress (closed-hand vs open-hand cursor).
        self._last_handle_zone: str | None = None
        self._space_panning: bool = False
        # Pre-built cursors swapped in/out by the prep-mode hover filter.
        self._cursor_draw = _make_cross_cursor((120, 200, 255))  # bright sky-blue
        self._cursor_erase = _make_cross_cursor((255, 140, 140))  # bright coral
        # Rotation cursor shown over the align handle's ring.
        self._cursor_rotate = _make_rotate_cursor((230, 230, 230))
        # Brush mode: circular cursor sized to the brush footprint in image px.
        self._brush_mode: bool = False
        self._brush_radius_img: int = 20
        self._build_ui()
        self._install_prep_cursor_filter()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.view)

        # Anatomical edge labels — a mouse-transparent overlay layered above the
        # graphics view so the words stay pinned to the container edges,
        # unaffected by zoom/pan.  Populated via ``set_orientation_labels``.
        self._orientation = OrientationOverlay(self)
        self._orientation.setGeometry(self.rect())
        self._orientation.raise_()

        self._vb = _OverlayViewBox()
        self._vb.setBackgroundColor((0, 0, 0))  # black so Lighten(channel, black)=channel
        self._vb.overlay_panned.connect(self.overlay_panned)
        self._vb.overlay_rotated.connect(self.overlay_rotated)
        self._vb.overlay_scaled.connect(self.overlay_scaled)

        self.plot = self.view.addPlot(viewBox=self._vb)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)  # image coords: row 0 at top
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)

        # Per-channel section ImageItems are created lazily by
        # ``set_channel_planes``; the list lives in ``self._channel_items``.
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(0.5)
        self.overlay_item.setZValue(10)

        # Control-point displacement lines (Warp mode) — drawn below the dots.
        self.disp_halo_item = pg.PlotCurveItem(
            pen=pg.mkPen((0, 0, 0, 220), width=5.0),
            connect="pairs",
        )
        self.disp_halo_item.setZValue(14)
        self.disp_item = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 255), width=2.75),
            connect="pairs",
        )
        self.disp_item.setZValue(15)

        # Control-point scatter (Warp mode)
        self.cp_item = pg.ScatterPlotItem(size=10, pxMode=True)
        self.cp_item.setZValue(20)

        # Live freehand stroke preview (Prep mode)
        self.stroke_item = pg.PlotCurveItem(
            pen=pg.mkPen((80, 160, 255, 220), width=2.0),
        )
        self.stroke_item.setZValue(30)

        # Align centre handle (translate/rotate manipulator), hidden until Align.
        self._align_handle = _AlignHandle()

        self.plot.addItem(self.overlay_item)
        self.plot.addItem(self.disp_halo_item)
        self.plot.addItem(self.disp_item)
        self.plot.addItem(self.cp_item)
        self.plot.addItem(self.stroke_item)
        self.plot.addItem(self._align_handle)
        self._vb.set_align_handle(self._align_handle)

        # Forward scene mouse moves as image-pixel coordinates
        self.plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        # Keep the brush cursor sized to the brush footprint as the user zooms.
        self._vb.sigRangeChanged.connect(self._on_view_range_changed)

        # Forward warp interaction signals from the ViewBox
        self._vb.canvas_clicked.connect(self.canvas_clicked)
        self._vb.canvas_drag_started.connect(self.canvas_drag_started)
        self._vb.canvas_dragged.connect(self.canvas_dragged)
        self._vb.canvas_drag_ended.connect(self.canvas_drag_ended)
        self._vb.space_pan_changed.connect(self._on_space_pan_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        # Keep the orientation overlay covering the full canvas area.
        self._orientation.setGeometry(self.rect())

    def set_orientation_labels(self, labels: dict[str, str] | None) -> None:
        """Set the anatomical edge labels drawn over the canvas.

        Pass ``None`` to clear them (e.g. when no project is loaded).
        """
        self._orientation.set_labels(labels)

    def set_interaction_mode(self, mode: _InteractionMode) -> None:
        """Choose how left-drag gestures are interpreted by the canvas.

        ``align`` shows the centre handle: dragging its ring rotates the overlay,
        its arrowhead grips stretch it, plain left-drag translates it, and
        spacebar+drag pans the view.
        ``prep`` emits plain left-drag signals for mask strokes while allowing
        space+drag to fall through to pyqtgraph.
        ``view`` lets pyqtgraph handle left-drag gestures normally.
        """
        self._vb.set_interaction_mode(mode)
        self._interaction_mode = mode
        self._update_handle_visibility()
        # If the cursor is already over the canvas, refresh immediately;
        # otherwise the next enterEvent will pick up the new mode.
        if self.view.underMouse():
            self._refresh_cursor()
        else:
            self.view.unsetCursor()

    def _update_handle_visibility(self) -> None:
        """Show the align handle only in Align mode with an overlay present."""
        active = self._interaction_mode == "align" and self._overlay_present
        self._align_handle.set_active(active)

    # ------------------------------------------------------------------
    # Prep-mode cursor: blue crosshair while hovering, red while Shift is
    # held. Implemented via an event filter on the inner GraphicsLayoutWidget
    # so we catch Enter/Leave plus Shift press/release without competing
    # with pyqtgraph's own mouse handling.
    # ------------------------------------------------------------------

    def _install_prep_cursor_filter(self) -> None:
        self.view.installEventFilter(self)
        # Wheel events go to the scroll-area viewport, not the view itself.
        self.view.viewport().installEventFilter(self)
        _ShiftState.listeners.add(self)
        _SpaceState.listeners.add(self)

        def _drop_listeners(_=None, s=self) -> None:
            _ShiftState.listeners.discard(s)
            _SpaceState.listeners.discard(s)

        self.destroyed.connect(_drop_listeners)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        # Alt+wheel adjusts the brush size. We intercept the raw QWheelEvent
        # here rather than in the ViewBox because Qt collapses Shift+wheel into
        # a horizontal scroll, and the scene-level QGraphicsSceneWheelEvent then
        # reports delta()==0 — which silently broke brush resizing while erasing
        # (Shift held). angleDelta() exposes both axes, so fall back to the
        # horizontal delta when the vertical one is zero.
        if t == QEvent.Type.Wheel and (event.modifiers() & Qt.KeyboardModifier.AltModifier):
            ad = event.angleDelta()
            delta = ad.y() or ad.x()
            if delta:
                self.alt_wheel_scrolled.emit(int(delta))
            return True
        if obj is self.view:
            if t == QEvent.Type.Enter:
                # Pick up the correct cursor for the current mode (and show the
                # pan grab cursor immediately if entering with space held).
                self._refresh_cursor()
            elif t == QEvent.Type.Leave:
                self.view.unsetCursor()
                self._align_handle.set_hovered(False)
        elif obj is self.view.viewport():
            # Closed-hand feedback the instant a space-pan grab begins (on press,
            # not only once a drag starts) and back to open hand on release. Works
            # in every mode since space+drag pans the view everywhere. The events
            # are observed, never consumed, so pyqtgraph still does the pan.
            if (
                t == QEvent.Type.MouseButtonPress
                and _SpaceState.held
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._space_panning = True
                self._refresh_cursor()
            elif (
                t == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self._space_panning
            ):
                self._space_panning = False
                self._refresh_cursor()
        return super().eventFilter(obj, event)

    def _on_shift_changed(self) -> None:
        """Called by the app-level filter when Shift state changes."""
        if self._interaction_mode == "prep" and self.view.underMouse():
            self._refresh_prep_cursor()

    def set_brush_cursor(self, active: bool, radius_img: int) -> None:
        """Enable the circular brush cursor (``active``) sized to ``radius_img``
        image pixels. When inactive the crosshair cursor is used."""
        self._brush_mode = bool(active)
        self._brush_radius_img = max(int(radius_img), 1)
        if self._interaction_mode == "prep" and self.view.underMouse():
            self._refresh_prep_cursor()

    def _on_view_range_changed(self, *_args: object) -> None:
        if self._brush_mode and self._interaction_mode == "prep" and self.view.underMouse():
            self._refresh_prep_cursor()
        self.view_range_changed.emit()

    def image_to_screen_scale(self) -> float:
        """Return screen pixels per image pixel at the current zoom.

        ``viewPixelSize()[0]`` is the size of one device pixel in image (view)
        coordinates, so its reciprocal converts image px → screen px. Returns
        ``0.0`` if the view has not been laid out / ranged yet, signalling
        callers to fall back to a fixed sampling resolution.
        """
        try:
            vps = float(self._vb.viewPixelSize()[0])
        except Exception:
            return 0.0
        if vps <= 0:
            return 0.0
        return 1.0 / vps

    def _refresh_prep_cursor(self) -> None:
        if self._interaction_mode != "prep":
            self.view.unsetCursor()
            return
        if self._apply_pan_cursor():
            return
        rgb = (255, 140, 140) if _ShiftState.held else (120, 200, 255)
        if self._brush_mode:
            px_per_img = 1.0 / max(self._vb.viewPixelSize()[0], 1e-9)
            diameter = int(round(2 * self._brush_radius_img * px_per_img))
            self.view.setCursor(_make_circle_cursor(rgb, diameter))
            return
        self.view.setCursor(self._cursor_erase if _ShiftState.held else self._cursor_draw)

    def set_channel_planes(self, planes: list[np.ndarray | None]) -> None:
        """Install the per-channel raw uint8 planes that drive the section view.

        Call this once per section load (and again after a flip). Each plane
        must be a contiguous ``(H, W)`` uint8 array, or ``None`` to hide that
        channel. Brightness/color updates do NOT go through here — they use
        :meth:`set_channel_lut`.
        """
        # Reconcile item count with the plane count.
        while len(self._channel_items) < len(planes):
            item = pg.ImageItem()
            item.setZValue(0)
            item.setAutoDownsample(True)
            # CompositionMode_Lighten: dst_rgb = max(src_rgb, dst_rgb) per
            # component. Stacked across channels, this is the GPU equivalent
            # of np.maximum.reduce — matching the old CPU composite output.
            item.setCompositionMode(QPainter.CompositionMode.CompositionMode_Lighten)
            self._channel_items.append(item)
            self.plot.addItem(item)
        while len(self._channel_items) > len(planes):
            item = self._channel_items.pop()
            self.plot.removeItem(item)

        first_shape: tuple[int, int] | None = None
        for item, plane in zip(self._channel_items, planes):
            if plane is None:
                item.clear()
                item.setVisible(False)
                continue
            item.setImage(plane, autoLevels=False, levels=(0, 255))
            item.setVisible(True)
            if first_shape is None:
                first_shape = plane.shape[:2]

        if first_shape is None:
            self._channel_shape = None
            return
        if first_shape != self._channel_shape:
            self.plot.autoRange()
            self._channel_shape = first_shape

    def set_channel_lut(self, index: int, lut: np.ndarray | None) -> None:
        """Apply (or clear) the lookup table for a single channel item.

        ``lut`` is a ``(256, 4)`` uint8 RGBA table — see
        ``verso.engine.preprocessing.channel_lut``. ``None`` hides the channel.
        """
        if not 0 <= index < len(self._channel_items):
            return
        item = self._channel_items[index]
        if lut is None:
            item.setVisible(False)
            return
        item.setLookupTable(lut)
        item.setVisible(True)

    def set_channel_visible(self, index: int, visible: bool) -> None:
        """Show or hide a single channel item without discarding its data."""
        if not 0 <= index < len(self._channel_items):
            return
        self._channel_items[index].setVisible(visible)

    def set_overlay(
        self,
        image: np.ndarray | None,
        display_w: int | None = None,
        display_h: int | None = None,
    ) -> None:
        """Set the atlas overlay (H×W×4 RGBA uint8, or None to hide).

        Args:
            image: RGBA overlay array, or None to clear.
            display_w / display_h: If provided, the overlay is scaled via
                ``setRect`` to fill exactly this region in image-pixel coords
                (allows the overlay to be sampled at a lower resolution for
                performance while still covering the full background).
        """
        if image is None:
            self.overlay_item.clear()
            self._overlay_present = False
            self._update_handle_visibility()
            return
        self.overlay_item.setImage(image)
        if display_w is not None and display_h is not None:
            self.overlay_item.setRect(QRectF(0, 0, display_w, display_h))
            self._align_handle.set_center(display_w / 2.0, display_h / 2.0)
        else:
            h, w = image.shape[:2]
            self._align_handle.set_center(w / 2.0, h / 2.0)
        self._overlay_present = True
        self._update_handle_visibility()

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        vb_pos = self._vb.mapSceneToView(scene_pos)
        if self._interaction_mode == "align":
            view_px = self._vb.viewPixelSize()[0]
            zone = self._align_handle.zone_at(vb_pos.x(), vb_pos.y(), view_px)
            self._last_handle_zone = zone
            # Brighten only over the gizmo itself (ring/grips), not the broad
            # translate field that covers the whole canvas.
            self._align_handle.set_hovered(zone in ("rotate", "stretch_x", "stretch_y"))
            self._refresh_align_cursor()
        self.mouse_position_changed.emit(vb_pos.x(), vb_pos.y())

    _HANDLE_CURSORS = {
        "stretch_x": Qt.CursorShape.SizeHorCursor,
        "stretch_y": Qt.CursorShape.SizeVerCursor,
    }

    def _apply_pan_cursor(self) -> bool:
        """Show the grab cursor whenever space is held, in any interaction mode.

        Open hand while space is merely held; closed hand once a (left) button is
        pressed and a pan is underway. Returns ``True`` when it took over the
        cursor so per-mode refreshers can bail out early.
        """
        if not _SpaceState.held:
            return False
        self.view.setCursor(
            Qt.CursorShape.ClosedHandCursor
            if self._space_panning
            else Qt.CursorShape.OpenHandCursor
        )
        return True

    def _refresh_cursor(self) -> None:
        """Dispatch to the right cursor for the current mode.

        The spacebar pan cursor takes priority in every mode; otherwise each mode
        picks its own (handle zones in Align, draw/erase in Prep, default arrow
        in Warp / View).
        """
        if self._interaction_mode == "align":
            self._refresh_align_cursor()
        elif self._interaction_mode == "prep":
            self._refresh_prep_cursor()
        elif not self._apply_pan_cursor():
            self.view.unsetCursor()

    def _refresh_align_cursor(self) -> None:
        """Set the canvas cursor for the current Align state.

        Spacebar shows the pan grab cursor (open hand, or closed hand while
        dragging); otherwise the hovered handle zone decides (rotate / stretch /
        default translate).
        """
        if self._interaction_mode != "align":
            return
        if self._apply_pan_cursor():
            return
        zone = self._last_handle_zone
        if zone == "rotate":
            self.view.setCursor(self._cursor_rotate)
            return
        shape = self._HANDLE_CURSORS.get(zone)
        if shape is None:
            self.view.unsetCursor()
        else:
            self.view.setCursor(shape)

    def _on_space_changed(self) -> None:
        """React to spacebar press/release (open-hand pan cursor in any mode)."""
        if self.view.underMouse():
            self._refresh_cursor()

    def _on_space_pan_changed(self, panning: bool) -> None:
        """Track an in-progress spacebar pan drag (closed-hand cursor)."""
        self._space_panning = panning
        self._refresh_cursor()

    _CP_SYMBOLS: dict[str, str] = {
        "Circle": "o",
        "Cross": "+",
        "Square": "s",
        "Diamond": "d",
    }
    _CP_COLOR_RGB: dict[str, tuple[int, int, int]] = {
        "Orange": (255, 96, 0),
        "Cyan": (0, 255, 255),
        "Yellow": (255, 245, 0),
        "Red": (255, 32, 32),
        "White": (255, 255, 255),
        "Magenta": (255, 0, 255),
    }
    # Fixed, contrasting colour for automatically-generated control points so
    # they stand apart from the user's manual ones regardless of the CP palette.
    _AUTO_CP_COLOR_RGB: tuple[int, int, int] = (0, 200, 255)

    def set_control_points(
        self,
        dst_pts: list[tuple[float, float]],
        display_w: int,
        display_h: int,
        hovered_idx: int = -1,
        cp_size: int = 10,
        cp_shape: str = "Circle",
        cp_color: str = "Orange",
        src_pts: list[tuple[float, float]] | None = None,
        auto_flags: list[bool] | None = None,
    ) -> None:
        """Draw warp control points and their displacement vectors.

        Args:
            dst_pts: List of (x, y) in normalised [0, 1] section coords (pin position).
            display_w / display_h: Section display dimensions in pixels.
            hovered_idx: Index of the point under the cursor (-1 = none).
            cp_size: Normal point diameter in pixels.
            cp_shape: One of Circle / Cross / Square / Diamond.
            cp_color: Named colour from the properties panel palette.
            src_pts: Atlas-space normalised origins for each CP. When provided,
                a dashed line is drawn from each src to its dst (the displacement
                vector, matching VisuAlign's pin rendering).
            auto_flags: Per-point flags; points marked True are drawn in the
                automatic-CP colour to distinguish them from manual points.
        """
        if not dst_pts:
            self.cp_item.clear()
            self.disp_halo_item.clear()
            self.disp_item.clear()
            return

        symbol = self._CP_SYMBOLS.get(cp_shape, "o")
        if cp_color.startswith("#") and len(cp_color) == 7:
            r, g, b = int(cp_color[1:3], 16), int(cp_color[3:5], 16), int(cp_color[5:7], 16)
        else:
            r, g, b = self._CP_COLOR_RGB.get(cp_color, (255, 80, 0))
        hov_size = cp_size + 4

        # Displacement lines (src → dst)
        if src_pts and len(src_pts) == len(dst_pts):
            xs, ys = [], []
            for (ss, st), (ds, dt) in zip(src_pts, dst_pts):
                xs += [ss * display_w, ds * display_w]
                ys += [st * display_h, dt * display_h]
            self.disp_halo_item.setData(x=xs, y=ys)
            self.disp_item.setPen(pg.mkPen((r, g, b, 255), width=2.75))
            self.disp_item.setData(x=xs, y=ys)
        else:
            self.disp_halo_item.clear()
            self.disp_item.clear()

        ar, ag, ab = self._AUTO_CP_COLOR_RGB
        spots = []
        for i, (s, t) in enumerate(dst_pts):
            px, py = s * display_w, t * display_h
            is_auto = bool(auto_flags[i]) if auto_flags and i < len(auto_flags) else False
            br, bg, bb = (ar, ag, ab) if is_auto else (r, g, b)
            if i == hovered_idx:
                spots.append(
                    {
                        "pos": (px, py),
                        "size": hov_size,
                        "symbol": symbol,
                        "brush": pg.mkBrush(br, bg, bb, 255),
                        "pen": pg.mkPen(255, 255, 255, 255, width=2.5),
                    }
                )
            else:
                spots.append(
                    {
                        "pos": (px, py),
                        "size": cp_size,
                        "symbol": symbol,
                        "brush": pg.mkBrush(br, bg, bb, 255),
                        "pen": pg.mkPen(0, 0, 0, 240, width=1.5),
                    }
                )
        self.cp_item.setData(spots)

    def clear_control_points(self) -> None:
        self.cp_item.clear()
        self.disp_halo_item.clear()
        self.disp_item.clear()

    def set_stroke_preview(
        self,
        points: list[tuple[float, float]],
        color: tuple[int, int, int] = (80, 160, 255),
    ) -> None:
        """Draw a live freehand stroke preview in image-pixel coordinates."""
        if len(points) < 2:
            self.stroke_item.clear()
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        self.stroke_item.setPen(pg.mkPen((*color, 220), width=2.0))
        self.stroke_item.setData(x=xs, y=ys)

    def clear_stroke_preview(self) -> None:
        self.stroke_item.clear()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set overlay opacity in [0, 1]."""
        self.overlay_item.setOpacity(opacity)

    def clear(self) -> None:
        for item in self._channel_items:
            item.clear()
            self.plot.removeItem(item)
        self._channel_items.clear()
        self._channel_shape = None
        self.overlay_item.clear()
        self._overlay_present = False
        self._update_handle_visibility()
        self.cp_item.clear()
        self.disp_halo_item.clear()
        self.disp_item.clear()
        self.stroke_item.clear()
