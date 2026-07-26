"""Filmstrip tiles mirror the section's Prep flips, matching the canvas.

The tile pixmap is loaded unflipped (the working copy on disk is never
modified), so the flip is a display-time transform — the same rule
``push_channel_display`` applies to the canvas planes.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap

from verso.engine.model.project import Preprocessing, Section
from verso.gui.widgets.filmstrip import Filmstrip


def _section(sid: str, flip_h: bool = False, flip_v: bool = False) -> Section:
    return Section(
        id=sid,
        slice_index=0,
        original_path=f"{sid}.png",
        thumbnail_path=f"thumbnails/{sid}.tif",
        preprocessing=Preprocessing(flip_horizontal=flip_h, flip_vertical=flip_v),
    )


def _asymmetric_pixmap() -> QPixmap:
    """A 4x4 tile with a single bright pixel in the top-left corner."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0, 0] = 255
    image = QImage(arr.tobytes(), 4, 4, 4 * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _bright_corner(pixmap: QPixmap) -> tuple[int, int]:
    """Return the (x, y) of the tile's bright pixel."""
    image = pixmap.toImage()
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).red() > 128:
                return x, y
    raise AssertionError("no bright pixel found")


def _strip(qtbot, sections: list[Section]) -> Filmstrip:
    strip = Filmstrip()
    qtbot.addWidget(strip)
    strip.populate(sections)
    strip._cancel_loader()  # drop the background loader; we set tiles by hand
    return strip


def test_tile_is_mirrored_for_saved_flips(qtbot):
    strip = _strip(qtbot, [_section("s0", flip_h=True)])
    strip._buttons[0].set_thumbnail(_asymmetric_pixmap())

    assert _bright_corner(strip._buttons[0].thumbnail()) == (3, 0)


def test_tile_follows_a_live_flip_toggle(qtbot):
    strip = _strip(qtbot, [_section("s0")])
    strip._buttons[0].set_thumbnail(_asymmetric_pixmap())
    assert _bright_corner(strip._buttons[0].thumbnail()) == (0, 0)

    strip.set_flip(0, False, True)
    assert _bright_corner(strip._buttons[0].thumbnail()) == (0, 3)

    strip.set_flip(0, True, True)
    assert _bright_corner(strip._buttons[0].thumbnail()) == (3, 3)

    # Toggling back restores the original orientation (no cumulative mirroring).
    strip.set_flip(0, False, False)
    assert _bright_corner(strip._buttons[0].thumbnail()) == (0, 0)


def test_flip_set_before_the_tile_loads_is_applied_on_arrival(qtbot):
    strip = _strip(qtbot, [_section("s0")])
    strip.set_flip(0, True, False)
    strip._buttons[0].set_thumbnail(_asymmetric_pixmap())

    assert _bright_corner(strip._buttons[0].thumbnail()) == (3, 0)
