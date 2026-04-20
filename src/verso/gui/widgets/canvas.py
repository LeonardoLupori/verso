"""Shared pyqtgraph image canvas used by Prep and Align/Warp views.

Stacks two ImageItems:
  bg_item     — histological section (set once per section load)
  overlay_item — atlas overlay (updated on every warp event)

Space + drag interaction: while the spacebar is held, left-button drag emits
``overlay_panned(dx, dy)`` in scene/data coordinates (image pixels).  The
AlignView connects this to translate the atlas cut plane.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget


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
        elif t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                _SpaceState.held = False
        return False   # never consume events


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

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and not _SpaceState.held and not ev.double():
            pos = self.mapSceneToView(ev.scenePos())
            self.canvas_clicked.emit(pos.x(), pos.y())
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if _SpaceState.held and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            p1 = self.mapSceneToView(ev.lastScenePos())
            p2 = self.mapSceneToView(ev.scenePos())
            self.overlay_panned.emit(p2.x() - p1.x(), p2.y() - p1.y())
        elif ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            pos = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self.canvas_drag_started.emit(pos.x(), pos.y())
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _ensure_space_filter()
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

        self.bg_item = pg.ImageItem()
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(0.5)
        self.overlay_item.setZValue(10)

        # Control-point displacement lines (Warp mode) — drawn below the dots
        self.disp_item = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 160), width=1.5,
                         style=Qt.PenStyle.DashLine),
            connect="pairs",
        )
        self.disp_item.setZValue(15)

        # Control-point scatter (Warp mode)
        self.cp_item = pg.ScatterPlotItem(size=10, pxMode=True)
        self.cp_item.setZValue(20)

        self.plot.addItem(self.bg_item)
        self.plot.addItem(self.overlay_item)
        self.plot.addItem(self.disp_item)
        self.plot.addItem(self.cp_item)

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

    def set_background(self, image: np.ndarray | None) -> None:
        """Set the section image (H×W or H×W×C, uint8)."""
        if image is None:
            self.bg_item.clear()
            return
        self.bg_item.setImage(image)
        self.plot.autoRange()

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

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        vb_pos = self._vb.mapSceneToView(scene_pos)
        self.mouse_position_changed.emit(vb_pos.x(), vb_pos.y())

    _CP_SYMBOLS: dict[str, str] = {
        "Circle": "o", "Cross": "+", "Square": "s", "Diamond": "d",
    }
    _CP_COLOR_RGB: dict[str, tuple[int, int, int]] = {
        "Orange": (255, 80, 0),
        "Cyan": (0, 210, 210),
        "Yellow": (255, 240, 0),
        "Red": (220, 50, 50),
        "White": (255, 255, 255),
        "Magenta": (210, 0, 210),
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
            self.disp_item.clear()
            return

        symbol = self._CP_SYMBOLS.get(cp_shape, "o")
        r, g, b = self._CP_COLOR_RGB.get(cp_color, (255, 80, 0))
        hov_size = cp_size + 4

        # Displacement lines (src → dst)
        if src_pts and len(src_pts) == len(dst_pts):
            xs, ys = [], []
            for (ss, st), (ds, dt) in zip(src_pts, dst_pts):
                xs += [ss * display_w, ds * display_w]
                ys += [st * display_h, dt * display_h]
            self.disp_item.setPen(
                pg.mkPen((r, g, b, 160), width=1.5,
                         style=Qt.PenStyle.DashLine)
            )
            self.disp_item.setData(x=xs, y=ys)
        else:
            self.disp_item.clear()

        spots = []
        for i, (s, t) in enumerate(dst_pts):
            px, py = s * display_w, t * display_h
            if i == hovered_idx:
                spots.append({
                    "pos": (px, py), "size": hov_size, "symbol": symbol,
                    "brush": pg.mkBrush(255, 240, 0, 240),
                    "pen": pg.mkPen("w", width=1.5),
                })
            else:
                spots.append({
                    "pos": (px, py), "size": cp_size, "symbol": symbol,
                    "brush": pg.mkBrush(r, g, b, 200),
                    "pen": pg.mkPen("w", width=1),
                })
        self.cp_item.setData(spots)

    def clear_control_points(self) -> None:
        self.cp_item.clear()
        self.disp_item.clear()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set overlay opacity in [0, 1]."""
        self.overlay_item.setOpacity(opacity)

    def clear(self) -> None:
        self.bg_item.clear()
        self.overlay_item.clear()
        self.cp_item.clear()
        self.disp_item.clear()
