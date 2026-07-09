"""Multi-layer point-annotation overlay for the image canvas.

``AnnotationOverlay`` renders a variable number of annotation layers, one
``pg.ScatterPlotItem`` per visible annotation, each with its own colour and
opacity. It is modelled on
:class:`~verso.gui.widgets.control_points.ControlPointOverlay` but, because the
number of layers is dynamic, it owns a reference to the plot and adds/removes its
scatter items to match the layer count (the same reconcile pattern
:meth:`ImageCanvas.set_channel_planes` uses for channel items).

Only the Annotate view feeds it layers; other modes leave it empty.
"""

from __future__ import annotations

from typing import TypedDict

import pyqtgraph as pg


class AnnotationLayer(TypedDict):
    """One annotation's rendered points (image-pixel coords) plus its style."""

    xs: list[float]
    ys: list[float]
    color: tuple[int, int, int]
    opacity: float
    size: int
    active: bool


class AnnotationOverlay:
    """Point annotations drawn on the canvas, one scatter layer per annotation."""

    #: z above the atlas overlay (10) / control points (20), below stroke preview (30).
    _Z = 22

    def __init__(self, plot: pg.PlotItem) -> None:
        self._plot = plot
        self._items: list[pg.ScatterPlotItem] = []

    def set(self, layers: list[AnnotationLayer]) -> None:
        """Render ``layers``, reconciling the scatter-item count to match."""
        while len(self._items) < len(layers):
            item = pg.ScatterPlotItem(pxMode=True)
            item.setZValue(self._Z)
            self._items.append(item)
            self._plot.addItem(item)
        while len(self._items) > len(layers):
            item = self._items.pop()
            self._plot.removeItem(item)

        for item, layer in zip(self._items, layers, strict=False):
            r, g, b = layer["color"]
            active = layer["active"]
            # The active annotation gets a white ring so the user can see which
            # one edits will target; the rest get a thin dark outline.
            pen = (
                pg.mkPen(255, 255, 255, 255, width=1.5)
                if active
                else pg.mkPen(0, 0, 0, 200, width=1.0)
            )
            item.setOpacity(max(0.0, min(1.0, layer["opacity"])))
            item.setData(
                x=layer["xs"],
                y=layer["ys"],
                size=layer["size"],
                symbol="o",
                brush=pg.mkBrush(r, g, b, 255),
                pen=pen,
            )

    def clear(self) -> None:
        """Remove all layers from the plot."""
        self.set([])
