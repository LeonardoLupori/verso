"""Factory helpers that build custom ``QCursor`` pixmaps for the image canvas.

Each helper renders a small pixmap (crosshair, circle, or rotation glyph) with a
dark halo for contrast and returns a ``QCursor`` with its hotspot at the centre.
They are pure functions with no canvas state, so they live apart from the canvas
widget.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap


def make_cross_cursor(rgb: tuple[int, int, int], size: int = 21) -> QCursor:
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


def make_circle_cursor(rgb: tuple[int, int, int], diameter_px: int) -> QCursor:
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


def make_rotate_cursor(rgb: tuple[int, int, int], size: int = 24) -> QCursor:
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
