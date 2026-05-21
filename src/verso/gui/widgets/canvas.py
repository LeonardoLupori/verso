"""Shared pyqtgraph image canvas used by Prep and Align/Warp views.

Item stack (low z to high):
  channel_items[i] — one ImageItem per section channel, each holding the raw
                     uint8 plane plus a per-channel 256x4 RGBA LUT.  Composited
                     together with CompositionMode_Lighten (component-wise max),
                     which is the GPU equivalent of np.maximum.reduce.
  overlay_item     — atlas overlay (z=10), normal SourceOver alpha blend.
  lr_overlay_item  — L/R hemisphere mask (z=11), SourceOver.
  disp_halo/disp   — warp displacement lines (z=14, 15).
  cp_item          — warp control points (z=20).
  stroke_item      — live freehand mask preview (z=30).

Space + drag interaction: while the spacebar is held, left-button drag emits
``overlay_panned(dx, dy)`` in scene/data coordinates (image pixels).  The
AlignView connects this to translate the atlas cut plane.  This gesture is
disabled in warp mode — spacebar and overlay movement are no-ops there.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QAbstractButton, QApplication, QSizePolicy, QVBoxLayout, QWidget

# ---------------------------------------------------------------------------
# Application-level space-key tracker (singleton, installed once)
# ---------------------------------------------------------------------------

class _SpaceState:
    held: bool = False


class _SpaceFilter(QObject):
    """Application event filter that tracks whether the spacebar is held."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                _SpaceState.held = True
                # Consume the event when a button has focus so spacebar doesn't
                # re-trigger the last clicked button while panning.
                if isinstance(QApplication.focusWidget(), QAbstractButton):
                    return True
        elif t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                _SpaceState.held = False
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
    # Emitted in image-pixel coordinates
    canvas_clicked = pyqtSignal(float, float)   # single click (no drag)
    canvas_drag_started = pyqtSignal(float, float)   # drag begin
    canvas_dragged = pyqtSignal(float, float)        # drag update
    canvas_drag_ended = pyqtSignal(float, float)     # drag finish

    _InteractionMode = Literal["align", "warp", "prep", "view"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interaction_mode: _OverlayViewBox._InteractionMode = "align"

    def set_interaction_mode(self, mode: _InteractionMode) -> None:
        self._interaction_mode = mode

    def mouseClickEvent(self, ev) -> None:
        if (
            self._interaction_mode in ("warp", "prep")
            and ev.button() == Qt.MouseButton.LeftButton
            and not _SpaceState.held
            and not ev.double()
        ):
            pos = self.mapSceneToView(ev.scenePos())
            self.canvas_clicked.emit(pos.x(), pos.y())
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if (
            self._interaction_mode == "align"
            and _SpaceState.held
            and ev.button() == Qt.MouseButton.LeftButton
        ):
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
        else:
            super().mouseDragEvent(ev, axis)


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    """PyQtGraph canvas with a background and an optional semi-transparent overlay."""

    # Emitted while space+drag panning in the Align view (dx, dy in image pixels)
    overlay_panned = pyqtSignal(float, float)
    # Emitted on every mouse move over the canvas (x, y in scene/image pixel coords)
    mouse_position_changed = pyqtSignal(float, float)
    # Warp control-point interaction (image pixel coords)
    canvas_clicked = pyqtSignal(float, float)
    canvas_drag_started = pyqtSignal(float, float)
    canvas_dragged = pyqtSignal(float, float)
    canvas_drag_ended = pyqtSignal(float, float)

    _InteractionMode = Literal["align", "warp", "prep", "view"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _ensure_space_filter()
        self._channel_items: list[pg.ImageItem] = []
        self._channel_shape: tuple[int, int] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.view)

        self._vb = _OverlayViewBox()
        self._vb.overlay_panned.connect(self.overlay_panned)

        self.plot = self.view.addPlot(viewBox=self._vb)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)          # image coords: row 0 at top
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)

        # Per-channel section ImageItems are created lazily by
        # ``set_channel_planes``; the list lives in ``self._channel_items``.
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(0.5)
        self.overlay_item.setZValue(10)

        # L/R hemisphere overlay (Prep mode) — sits above slice overlay,
        # below the displacement halos / control points.
        self.lr_overlay_item = pg.ImageItem()
        self.lr_overlay_item.setZValue(11)

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

        self.plot.addItem(self.overlay_item)
        self.plot.addItem(self.lr_overlay_item)
        self.plot.addItem(self.disp_halo_item)
        self.plot.addItem(self.disp_item)
        self.plot.addItem(self.cp_item)
        self.plot.addItem(self.stroke_item)

        # Forward scene mouse moves as image-pixel coordinates
        self.plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        # Forward warp interaction signals from the ViewBox
        self._vb.canvas_clicked.connect(self.canvas_clicked)
        self._vb.canvas_drag_started.connect(self.canvas_drag_started)
        self._vb.canvas_dragged.connect(self.canvas_dragged)
        self._vb.canvas_drag_ended.connect(self.canvas_drag_ended)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_interaction_mode(self, mode: _InteractionMode) -> None:
        """Choose how left-drag gestures are interpreted by the canvas.

        ``align`` preserves Align/Warp behavior: space+drag pans the atlas
        overlay and plain left-drag emits CP interaction signals.
        ``prep`` emits plain left-drag signals for mask strokes while allowing
        space+drag to fall through to pyqtgraph.
        ``view`` lets pyqtgraph handle left-drag gestures normally.
        """
        self._vb.set_interaction_mode(mode)

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
            return
        self.overlay_item.setImage(image)
        if display_w is not None and display_h is not None:
            from PyQt6.QtCore import QRectF
            self.overlay_item.setRect(QRectF(0, 0, display_w, display_h))

    def set_lr_overlay(
        self,
        image: np.ndarray | None,
        display_w: int | None = None,
        display_h: int | None = None,
    ) -> None:
        """Set the L/R hemisphere overlay (H×W×4 RGBA uint8, or None to hide).

        Mirrors :meth:`set_overlay` for the dedicated ``lr_overlay_item``
        layer used in Prep mode.
        """
        if image is None:
            self.lr_overlay_item.clear()
            return
        self.lr_overlay_item.setImage(image)
        if display_w is not None and display_h is not None:
            from PyQt6.QtCore import QRectF
            self.lr_overlay_item.setRect(QRectF(0, 0, display_w, display_h))

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        vb_pos = self._vb.mapSceneToView(scene_pos)
        self.mouse_position_changed.emit(vb_pos.x(), vb_pos.y())

    _CP_SYMBOLS: dict[str, str] = {
        "Circle": "o", "Cross": "+", "Square": "s", "Diamond": "d",
    }
    _CP_COLOR_RGB: dict[str, tuple[int, int, int]] = {
        "Orange": (255, 96, 0),
        "Cyan": (0, 255, 255),
        "Yellow": (255, 245, 0),
        "Red": (255, 32, 32),
        "White": (255, 255, 255),
        "Magenta": (255, 0, 255),
    }

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
            self.disp_item.setPen(
                pg.mkPen((r, g, b, 255), width=2.75)
            )
            self.disp_item.setData(x=xs, y=ys)
        else:
            self.disp_halo_item.clear()
            self.disp_item.clear()

        spots = []
        for i, (s, t) in enumerate(dst_pts):
            px, py = s * display_w, t * display_h
            if i == hovered_idx:
                spots.append({
                    "pos": (px, py), "size": hov_size, "symbol": symbol,
                    "brush": pg.mkBrush(r, g, b, 255),
                    "pen": pg.mkPen(255, 255, 255, 255, width=2.5),
                })
            else:
                spots.append({
                    "pos": (px, py), "size": cp_size, "symbol": symbol,
                    "brush": pg.mkBrush(r, g, b, 255),
                    "pen": pg.mkPen(0, 0, 0, 240, width=1.5),
                })
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

    def set_lr_overlay_opacity(self, opacity: float) -> None:
        """Set L/R hemisphere overlay opacity in [0, 1]."""
        self.lr_overlay_item.setOpacity(opacity)

    def set_lr_overlay_visible(self, visible: bool) -> None:
        """Show or hide the L/R hemisphere overlay without discarding its image data."""
        self.lr_overlay_item.setVisible(visible)

    def clear(self) -> None:
        for item in self._channel_items:
            item.clear()
            self.plot.removeItem(item)
        self._channel_items.clear()
        self._channel_shape = None
        self.overlay_item.clear()
        self.lr_overlay_item.clear()
        self.cp_item.clear()
        self.disp_halo_item.clear()
        self.disp_item.clear()
        self.stroke_item.clear()
