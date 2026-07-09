"""Shared pyqtgraph image canvas used by Prep and Align/Warp views.

Item stack (low z to high):
  channel_items[i] — one ImageItem per section channel, each holding the raw
                     uint8 plane plus a per-channel 256x4 RGBA LUT.  Composited
                     together with CompositionMode_Lighten (component-wise max),
                     which is the GPU equivalent of np.maximum.reduce.
  overlay_item     — atlas overlay (z=10), normal SourceOver alpha blend.
  _cp_overlay      — warp displacement lines (z=14, 15) + control points (z=20),
                     owned by a ``ControlPointOverlay``.
  stroke_item      — live freehand mask preview (z=30).

Align handle: in Align mode a centre gizmo (``AlignHandle``) is drawn over the
overlay.  Dragging the N/E/S/W arrowhead grips emits
``overlay_scaled(scale_s, scale_t)`` to stretch width/height; dragging the ring
emits ``overlay_rotated(deg)`` to spin it in-plane; dragging anywhere else emits
``overlay_panned(dx, dy)`` to translate the cut plane.  The centre dot is inert.
Holding the spacebar while dragging pans the view instead.  The handle is hidden
outside Align mode and when no overlay is present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Literal

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QMouseEvent,
    QPainter,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from verso.gui.utils import require
from verso.gui.widgets.align_handle import AlignHandle
from verso.gui.widgets.annotation_overlay import AnnotationLayer, AnnotationOverlay
from verso.gui.widgets.area_overlay import AreaLayer, AreaOverlay
from verso.gui.widgets.control_points import ControlPointOverlay
from verso.gui.widgets.cursors import (
    make_circle_cursor,
    make_cross_cursor,
    make_rotate_cursor,
)
from verso.gui.widgets.key_state import (
    ShiftState,
    SpaceState,
    ensure_key_state_filter,
)
from verso.gui.widgets.orientation_overlay import OrientationOverlay

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

    _InteractionMode = Literal["align", "warp", "prep", "annotate", "view"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interaction_mode: _OverlayViewBox._InteractionMode = "align"
        self._align_handle: AlignHandle | None = None

    def set_interaction_mode(self, mode: _InteractionMode) -> None:
        self._interaction_mode = mode

    def set_align_handle(self, handle: AlignHandle) -> None:
        self._align_handle = handle

    def mouseClickEvent(self, ev) -> None:
        if ev.double() and ev.button() == Qt.MouseButton.LeftButton:
            self.autoRange()
            ev.accept()
            return
        if (
            self._interaction_mode in ("warp", "prep", "annotate")
            and ev.button() == Qt.MouseButton.LeftButton
            and not SpaceState.held
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
            if SpaceState.held:
                if ev.isStart():
                    self.space_pan_changed.emit(True)
                elif ev.isFinish():
                    self.space_pan_changed.emit(False)
                super().mouseDragEvent(ev, axis)
                return
            # Classify the gesture by the handle zone it started in, then defer
            # the actual rotate/stretch math to the handle; the viewbox only
            # routes the result to the matching signal.
            view_px = self.viewPixelSize()[0]
            handle = self._align_handle
            start = self.mapSceneToView(ev.buttonDownScenePos())
            zone = handle.zone_at(start.x(), start.y(), view_px) if handle is not None else None
            p1 = self.mapSceneToView(ev.lastScenePos())
            p2 = self.mapSceneToView(ev.scenePos())
            ev.accept()
            if zone == "rotate":
                self.overlay_rotated.emit(
                    require(handle).rotate_delta(p1.x(), p1.y(), p2.x(), p2.y())
                )
            elif zone in ("stretch_x", "stretch_y"):
                scale_s, scale_t = require(handle).stretch_delta(
                    zone, p1.x(), p1.y(), p2.x(), p2.y(), view_px
                )
                self.overlay_scaled.emit(scale_s, scale_t)
            else:
                # Default: drag anywhere else translates the atlas overlay.
                self.overlay_panned.emit(p2.x() - p1.x(), p2.y() - p1.y())
        elif (
            self._interaction_mode in ("warp", "prep", "annotate")
            and not SpaceState.held
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

    _InteractionMode = Literal["align", "warp", "prep", "annotate", "view"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ensure_key_state_filter()
        self._channel_items: list[pg.ImageItem] = []
        self._channel_shape: tuple[int, int] | None = None
        self._interaction_mode: ImageCanvas._InteractionMode = "align"
        self._overlay_present: bool = False
        # Align-cursor state: last hovered handle zone + whether a spacebar pan
        # drag is currently in progress (closed-hand vs open-hand cursor).
        self._last_handle_zone: str | None = None
        self._space_panning: bool = False
        # Pre-built cursors swapped in/out by the prep-mode hover filter.
        self._cursor_draw = make_cross_cursor((120, 200, 255))  # bright sky-blue
        self._cursor_erase = make_cross_cursor((255, 140, 140))  # bright coral
        # Rotation cursor shown over the align handle's ring.
        self._cursor_rotate = make_rotate_cursor((230, 230, 230))
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

        self.plot = self.view.ci.addPlot(viewBox=self._vb)
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)  # image coords: row 0 at top
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)

        # Per-channel section ImageItems are created lazily by
        # ``set_channel_planes``; the list lives in ``self._channel_items``.
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(0.5)
        self.overlay_item.setZValue(10)

        # Control-point dots + displacement vectors (Warp mode). Owns its own
        # graphics items; the canvas adds them to the plot at their z-values.
        self._cp_overlay = ControlPointOverlay()

        # Point-annotation layers (Annotate mode). Manages its own scatter items
        # dynamically, so it needs the plot to add/remove them.
        self._annotation_overlay = AnnotationOverlay(self.plot)

        # Area-mask layers (Annotate mode), rendered below the point overlay.
        self._area_overlay = AreaOverlay(self.plot)

        # Live freehand stroke preview (Prep mode)
        self.stroke_item = pg.PlotCurveItem(
            pen=pg.mkPen((80, 160, 255, 220), width=2.0),
        )
        self.stroke_item.setZValue(30)

        # Align centre handle (translate/rotate manipulator), hidden until Align.
        self._align_handle = AlignHandle()

        self.plot.addItem(self.overlay_item)
        for item in self._cp_overlay.items():
            self.plot.addItem(item)
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

    def resizeEvent(self, event) -> None:
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
        require(self.view.viewport()).installEventFilter(self)
        ShiftState.listeners.add(self)
        SpaceState.listeners.add(self)

        def _drop_listeners(_=None, s=self) -> None:
            ShiftState.listeners.discard(s)
            SpaceState.listeners.discard(s)

        self.destroyed.connect(_drop_listeners)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        # Alt+wheel adjusts the brush size. We intercept the raw QWheelEvent
        # here rather than in the ViewBox because Qt collapses Shift+wheel into
        # a horizontal scroll, and the scene-level QGraphicsSceneWheelEvent then
        # reports delta()==0 — which silently broke brush resizing while erasing
        # (Shift held). angleDelta() exposes both axes, so fall back to the
        # horizontal delta when the vertical one is zero.
        if (
            t == QEvent.Type.Wheel
            and isinstance(event, QWheelEvent)
            and (event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
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
                self._align_handle.set_hover_zone(None)
        elif obj is self.view.viewport() and isinstance(event, QMouseEvent):
            # Closed-hand feedback the instant a space-pan grab begins (on press,
            # not only once a drag starts) and back to open hand on release. Works
            # in every mode since space+drag pans the view everywhere. The events
            # are observed, never consumed, so pyqtgraph still does the pan.
            if (
                t == QEvent.Type.MouseButtonPress
                and SpaceState.held
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

    def _on_view_range_changed(self, *_: object) -> None:
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
        rgb = (255, 140, 140) if ShiftState.held else (120, 200, 255)
        if self._brush_mode:
            px_per_img = 1.0 / max(self._vb.viewPixelSize()[0], 1e-9)
            diameter = round(2 * self._brush_radius_img * px_per_img)
            self.view.setCursor(make_circle_cursor(rgb, diameter))
            return
        self.view.setCursor(self._cursor_erase if ShiftState.held else self._cursor_draw)

    def set_channel_planes(self, planes: Sequence[np.ndarray | None]) -> None:
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
        for item, plane in zip(self._channel_items, planes, strict=False):
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
            self._vb.autoRange()
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
            # Brighten only the hovered element (ring or a stretch-arrow pair);
            # the handle treats the broad translate field as no hover.
            self._align_handle.set_hover_zone(zone)
            self._refresh_align_cursor()
        self.mouse_position_changed.emit(vb_pos.x(), vb_pos.y())

    _HANDLE_CURSORS: ClassVar[dict[str, Qt.CursorShape]] = {
        "stretch_x": Qt.CursorShape.SizeHorCursor,
        "stretch_y": Qt.CursorShape.SizeVerCursor,
    }

    def _apply_pan_cursor(self) -> bool:
        """Show the grab cursor whenever space is held, in any interaction mode.

        Open hand while space is merely held; closed hand once a (left) button is
        pressed and a pan is underway. Returns ``True`` when it took over the
        cursor so per-mode refreshers can bail out early.
        """
        if not SpaceState.held:
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
        elif self._interaction_mode == "annotate":
            if not self._apply_pan_cursor():
                self.view.setCursor(Qt.CursorShape.CrossCursor)
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
        shape = self._HANDLE_CURSORS.get(zone) if zone is not None else None
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

        Thin pass-through to :class:`ControlPointOverlay`; see its ``set`` for the
        full argument semantics.
        """
        self._cp_overlay.set(
            dst_pts,
            display_w,
            display_h,
            hovered_idx=hovered_idx,
            cp_size=cp_size,
            cp_shape=cp_shape,
            cp_color=cp_color,
            src_pts=src_pts,
            auto_flags=auto_flags,
        )

    def clear_control_points(self) -> None:
        self._cp_overlay.clear()

    def set_annotations(self, layers: list[AnnotationLayer]) -> None:
        """Draw point-annotation layers (Annotate mode). See AnnotationOverlay."""
        self._annotation_overlay.set(layers)

    def clear_annotations(self) -> None:
        self._annotation_overlay.clear()

    def set_area_masks(self, layers: list[AreaLayer]) -> None:
        """Draw area-mask layers (Annotate mode). See AreaOverlay."""
        self._area_overlay.set(layers)

    def clear_area_masks(self) -> None:
        self._area_overlay.clear()

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
        self._cp_overlay.clear()
        self._annotation_overlay.clear()
        self._area_overlay.clear()
        self.stroke_item.clear()
