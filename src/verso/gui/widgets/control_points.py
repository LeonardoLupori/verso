"""Warp control-point overlay for the image canvas.

``ControlPointOverlay`` bundles the pyqtgraph items used to render warp
control points — the dot scatter plus the displacement-vector curves (one for
manual CPs, one for auto-generated CPs) — together with the draw logic and the
shape/colour
palettes. It is not a widget: the canvas owns one instance, adds its items to the
plot via :meth:`items` (the canvas keeps ownership of the item stack and its
z-ordering), and delegates :meth:`set` / :meth:`clear`. Only the Warp view feeds
it points; other modes leave it cleared.
"""

from __future__ import annotations

from typing import ClassVar

import pyqtgraph as pg


class ControlPointOverlay:
    """Control-point dots and their displacement vectors drawn on the canvas."""

    _CP_SYMBOLS: ClassVar[dict[str, str]] = {
        "Circle": "o",
        "Cross": "+",
        "Square": "s",
        "Diamond": "d",
    }
    _CP_COLOR_RGB: ClassVar[dict[str, tuple[int, int, int]]] = {
        "Orange": (255, 96, 0),
        "Cyan": (0, 255, 255),
        "Yellow": (255, 245, 0),
        "Red": (255, 32, 32),
        "White": (255, 255, 255),
        "Magenta": (255, 0, 255),
    }
    # Fixed default colour for automatically-generated control points and their
    # displacement line, setting them apart from the user's manual points (which
    # take the palette colour for both dot and line).
    _AUTO_CP_COLOR_RGB: tuple[int, int, int] = (0, 200, 255)

    # z-values in the canvas item stack: displacement line below, dots on top.
    _Z_DISP = 15
    _Z_DOTS = 20

    def __init__(self) -> None:
        # Displacement lines (src → dst), drawn below the dots. Manual and auto
        # CPs use separate curves so each can be coloured to match its dots.
        self.disp_item = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 255), width=2.75),
            connect="pairs",
        )
        self.disp_item.setZValue(self._Z_DISP)
        self.disp_auto_item = pg.PlotCurveItem(
            pen=pg.mkPen((*self._AUTO_CP_COLOR_RGB, 255), width=2.75),
            connect="pairs",
        )
        self.disp_auto_item.setZValue(self._Z_DISP)
        # Control-point scatter.
        self.cp_item = pg.ScatterPlotItem(size=10, pxMode=True)
        self.cp_item.setZValue(self._Z_DOTS)

    def items(self) -> tuple[pg.GraphicsObject, ...]:
        """Graphics items to add to the canvas plot, ordered low z to high."""
        return (self.disp_item, self.disp_auto_item, self.cp_item)

    def set(
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
            auto_flags: Per-point flags; points marked True (and their
                displacement lines) are drawn in the fixed automatic-CP colour,
                setting them apart from the user's manual, palette-coloured points.
        """
        if not dst_pts:
            self.clear()
            return

        symbol = self._CP_SYMBOLS.get(cp_shape, "o")
        if cp_color.startswith("#") and len(cp_color) == 7:
            r, g, b = int(cp_color[1:3], 16), int(cp_color[3:5], 16), int(cp_color[5:7], 16)
        else:
            r, g, b = self._CP_COLOR_RGB.get(cp_color, (255, 80, 0))
        hov_size = cp_size + 4

        ar, ag, ab = self._AUTO_CP_COLOR_RGB

        def _is_auto(i: int) -> bool:
            return bool(auto_flags[i]) if auto_flags and i < len(auto_flags) else False

        # Displacement lines (src → dst): each segment matches its CP's colour, so
        # manual points get the palette line and auto points the fixed auto line.
        if src_pts and len(src_pts) == len(dst_pts):
            man_xs, man_ys, auto_xs, auto_ys = [], [], [], []
            for i, ((ss, st), (ds, dt)) in enumerate(zip(src_pts, dst_pts, strict=False)):
                seg_x = [ss * display_w, ds * display_w]
                seg_y = [st * display_h, dt * display_h]
                if _is_auto(i):
                    auto_xs += seg_x
                    auto_ys += seg_y
                else:
                    man_xs += seg_x
                    man_ys += seg_y
            self.disp_item.setPen(pg.mkPen((r, g, b, 255), width=2.75))
            self.disp_item.setData(x=man_xs, y=man_ys)
            self.disp_auto_item.setData(x=auto_xs, y=auto_ys)
        else:
            self.disp_item.clear()
            self.disp_auto_item.clear()

        spots = []
        for i, (s, t) in enumerate(dst_pts):
            px, py = s * display_w, t * display_h
            is_auto = _is_auto(i)
            br, bg, bb = (ar, ag, ab) if is_auto else (r, g, b)
            if i == hovered_idx:
                spots.append(
                    {
                        "pos": (px, py),
                        "size": hov_size,
                        "symbol": symbol,
                        "brush": pg.mkBrush(br, bg, bb, 255),
                        "pen": pg.mkPen(255, 255, 255, 255, width=1.5),
                    }
                )
            else:
                spots.append(
                    {
                        "pos": (px, py),
                        "size": cp_size,
                        "symbol": symbol,
                        "brush": pg.mkBrush(br, bg, bb, 255),
                        "pen": pg.mkPen(255, 255, 255, 240, width=1.0),
                    }
                )
        self.cp_item.setData(spots)

    def clear(self) -> None:
        """Remove all drawn points and displacement lines."""
        self.cp_item.clear()
        self.disp_item.clear()
        self.disp_auto_item.clear()
