"""Warp control-point overlay for the image canvas.

``ControlPointOverlay`` bundles the three pyqtgraph items used to render warp
control points — the dot scatter plus the two displacement-vector curves (a dark
halo under a coloured line) — together with the draw logic and the shape/colour
palettes. It is not a widget: the canvas owns one instance, adds its items to the
plot via :meth:`items` (the canvas keeps ownership of the item stack and its
z-ordering), and delegates :meth:`set` / :meth:`clear`. Only the Warp view feeds
it points; other modes leave it cleared.
"""

from __future__ import annotations

import pyqtgraph as pg


class ControlPointOverlay:
    """Control-point dots and their displacement vectors drawn on the canvas."""

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

    # z-values in the canvas item stack: halo below the coloured line, dots on top.
    _Z_DISP_HALO = 14
    _Z_DISP = 15
    _Z_DOTS = 20

    def __init__(self) -> None:
        # Displacement lines (src → dst), drawn below the dots.
        self.disp_halo_item = pg.PlotCurveItem(
            pen=pg.mkPen((0, 0, 0, 220), width=5.0),
            connect="pairs",
        )
        self.disp_halo_item.setZValue(self._Z_DISP_HALO)
        self.disp_item = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 255), width=2.75),
            connect="pairs",
        )
        self.disp_item.setZValue(self._Z_DISP)
        # Control-point scatter.
        self.cp_item = pg.ScatterPlotItem(size=10, pxMode=True)
        self.cp_item.setZValue(self._Z_DOTS)

    def items(self) -> tuple[pg.GraphicsObject, ...]:
        """Graphics items to add to the canvas plot, ordered low z to high."""
        return (self.disp_halo_item, self.disp_item, self.cp_item)

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
            auto_flags: Per-point flags; points marked True are drawn in the
                automatic-CP colour to distinguish them from manual points.
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

    def clear(self) -> None:
        """Remove all drawn points and displacement lines."""
        self.cp_item.clear()
        self.disp_halo_item.clear()
        self.disp_item.clear()
