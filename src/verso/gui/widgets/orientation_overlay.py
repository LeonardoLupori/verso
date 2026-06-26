"""Transparent overlay that draws anatomical direction words at the canvas edges.

The overlay is a sibling of the pyqtgraph graphics view (a child of
``ImageCanvas``), so the words stay pinned to the container edges and never move
or scale with zoom/pan.  Words are painted with a bright-gray fill and a dark
outline stroke so they stay legible over any background.

The four words shown depend on the project's interpolation axis; the engine
decides the mapping (see ``verso.engine.orientation_labels``).
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QWidget

# Distance from the container edge to the nearest glyph, in device pixels.
_EDGE_MARGIN = 8
# Glyph appearance.
_FILL = QColor(200, 200, 200)  # relatively bright gray
_STROKE = QColor(0, 0, 0)  # dark outline so it reads over bright backgrounds
_STROKE_WIDTH = 2
_FONT_POINT_SIZE = 12


class OrientationOverlay(QWidget):
    """Paints up to four edge labels (top/bottom/left/right) over its parent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Never intercept canvas interaction; paint only the glyphs.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._labels: dict[str, str] | None = None
        self._font = QFont()
        self._font.setPointSize(_FONT_POINT_SIZE)

    def set_labels(self, labels: dict[str, str] | None) -> None:
        """Set the edge labels (keys ``top``/``bottom``/``left``/``right``).

        Passing ``None`` clears the overlay (nothing is drawn).
        """
        self._labels = dict(labels) if labels else None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if not self._labels:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        fm = QFontMetricsF(self._font)
        ascent = fm.ascent()
        descent = fm.descent()
        # Extent of the text perpendicular to its baseline; the rotated
        # side labels stick out this far from the edge.
        height = ascent + descent

        def draw(word: str, cx: float, cy: float, angle: float = 0.0) -> None:
            """Draw ``word`` centered on ``(cx, cy)``, rotated ``angle`` degrees.

            Qt rotates clockwise for positive angles, so the right label uses
            ``+90`` (CW) and the left label ``-90`` (CCW).
            """
            half = fm.horizontalAdvance(word) / 2.0
            path = QPainterPath()
            # Baseline placed so the glyphs are vertically centered on the origin.
            path.addText(QPointF(-half, (ascent - descent) / 2.0), self._font, word)
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(angle)
            painter.setPen(QPen(_STROKE, _STROKE_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_FILL))
            painter.drawPath(path)
            painter.restore()

        cx = rect.center().x()
        cy = rect.center().y()

        top = self._labels.get("top")
        if top:
            draw(top, cx, _EDGE_MARGIN + height / 2.0)
        bottom = self._labels.get("bottom")
        if bottom:
            draw(bottom, cx, rect.height() - _EDGE_MARGIN - height / 2.0)
        left = self._labels.get("left")
        if left:
            draw(left, _EDGE_MARGIN + height / 2.0, cy, angle=-90.0)
        right = self._labels.get("right")
        if right:
            draw(right, rect.width() - _EDGE_MARGIN - height / 2.0, cy, angle=90.0)

        painter.end()
