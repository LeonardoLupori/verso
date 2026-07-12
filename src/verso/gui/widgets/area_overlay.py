"""Multi-layer area-mask overlay for the image canvas.

``AreaOverlay`` renders a variable number of area annotations, one ``pg.ImageItem``
per visible area's mask on the current section, each a coloured semi-transparent
overlay. Like :class:`~verso.gui.widgets.annotation_overlay.AnnotationOverlay` it
owns a reference to the plot and reconciles its items to the layer count; it sits
*below* the point overlay so scatter points stay visible on top.

Only the Annotate view feeds it layers; other modes leave it empty.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF


class AreaLayer(TypedDict):
    """One area's rendered mask (RGBA) plus the section display size + opacity."""

    rgba: np.ndarray
    w: int
    h: int
    opacity: float


class AreaOverlay:
    """Area masks drawn on the canvas, one image layer per visible area."""

    #: z above the background (0) / atlas overlay (10), below points (22).
    _Z = 12

    def __init__(self, plot: pg.PlotItem) -> None:
        self._plot = plot
        self._items: list[pg.ImageItem] = []

    def set(self, layers: list[AreaLayer]) -> None:
        """Render ``layers``, reconciling the image-item count to match."""
        while len(self._items) < len(layers):
            item = pg.ImageItem()
            item.setZValue(self._Z)
            self._items.append(item)
            self._plot.addItem(item)
        while len(self._items) > len(layers):
            item = self._items.pop()
            self._plot.removeItem(item)

        for item, layer in zip(self._items, layers, strict=False):
            item.setImage(layer["rgba"])
            # setRect stretches the mask to the section's working-resolution size,
            # so a mask stored at a slightly different scale still covers it.
            item.setRect(QRectF(0, 0, layer["w"], layer["h"]))
            item.setOpacity(max(0.0, min(1.0, layer["opacity"])))

    def clear(self) -> None:
        """Remove all layers from the plot."""
        self.set([])
