"""Unit tests for the warp control-point overlay (``ControlPointOverlay``).

Covers the rendering logic that moved off the canvas: how dst/src points map to
scatter spots and displacement-line vertices, palette/shape resolution, the
hovered and auto-CP styling, and clearing.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from verso.gui.widgets.control_points import ControlPointOverlay


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _spots(overlay: ControlPointOverlay) -> list:
    # ScatterPlotItem.data is a structured array; return its records as a list.
    return list(overlay.cp_item.data)


# ---------------------------------------------------------------------------
# items() / z-ordering
# ---------------------------------------------------------------------------


def test_items_ordered_low_to_high_z(_qapp):
    overlay = ControlPointOverlay()
    items = overlay.items()
    assert items == (overlay.disp_item, overlay.disp_auto_item, overlay.cp_item)
    zs = [it.zValue() for it in items]
    assert zs == sorted(zs)  # lines < dots
    assert zs == [15, 15, 20]


# ---------------------------------------------------------------------------
# set(): scatter placement
# ---------------------------------------------------------------------------


def test_set_scales_normalised_points_to_display(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.5, 0.25), (1.0, 1.0)], display_w=200, display_h=80)
    spots = _spots(overlay)
    assert len(spots) == 2
    assert (spots[0]["x"], spots[0]["y"]) == (100.0, 20.0)
    assert (spots[1]["x"], spots[1]["y"]) == (200.0, 80.0)


def test_set_empty_clears(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.5, 0.5)], 100, 100)
    assert len(_spots(overlay)) == 1
    overlay.set([], 100, 100)
    assert len(_spots(overlay)) == 0


# ---------------------------------------------------------------------------
# set(): displacement lines
# ---------------------------------------------------------------------------


def test_displacement_line_drawn_when_src_matches(_qapp):
    overlay = ControlPointOverlay()
    overlay.set(
        [(0.5, 0.5)],
        display_w=100,
        display_h=100,
        src_pts=[(0.1, 0.2)],
    )
    x, y = overlay.disp_item.getData()
    # One src→dst segment: [src, dst] pairs in display pixels.
    assert list(x) == [10.0, 50.0]
    assert list(y) == [20.0, 50.0]


def test_displacement_line_absent_when_src_length_mismatch(_qapp):
    overlay = ControlPointOverlay()
    overlay.set(
        [(0.5, 0.5), (0.6, 0.6)],
        display_w=100,
        display_h=100,
        src_pts=[(0.1, 0.2)],  # only one src for two dst → skipped
    )
    x, _ = overlay.disp_item.getData()
    assert x is None or len(x) == 0


# ---------------------------------------------------------------------------
# set(): styling (colour, shape, hover, auto)
# ---------------------------------------------------------------------------


def test_named_colour_and_shape_resolved(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.5, 0.5)], 100, 100, cp_shape="Square", cp_color="Cyan")
    spot = _spots(overlay)[0]
    assert spot["symbol"] == "s"
    assert spot["brush"].color().getRgb()[:3] == (0, 255, 255)


def test_hex_colour_parsed(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.5, 0.5)], 100, 100, cp_color="#10203f")
    spot = _spots(overlay)[0]
    assert spot["brush"].color().getRgb()[:3] == (16, 32, 63)


def test_hovered_point_is_enlarged(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.2, 0.2), (0.8, 0.8)], 100, 100, hovered_idx=1, cp_size=10)
    spots = _spots(overlay)
    assert spots[0]["size"] == 10
    assert spots[1]["size"] == 14  # cp_size + 4


def test_auto_flagged_point_uses_auto_colour(_qapp):
    overlay = ControlPointOverlay()
    overlay.set(
        [(0.2, 0.2), (0.8, 0.8)],
        100,
        100,
        cp_color="Orange",
        auto_flags=[False, True],
    )
    spots = _spots(overlay)
    assert spots[0]["brush"].color().getRgb()[:3] == (255, 96, 0)  # manual → palette
    assert spots[1]["brush"].color().getRgb()[:3] == (0, 200, 255)  # auto → fixed


def test_displacement_lines_match_their_cp_colour(_qapp):
    overlay = ControlPointOverlay()
    overlay.set(
        [(0.5, 0.5), (0.6, 0.6)],
        display_w=100,
        display_h=100,
        cp_color="Orange",
        src_pts=[(0.1, 0.2), (0.3, 0.4)],
        auto_flags=[False, True],
    )
    # Manual segment goes on the palette-coloured line; auto on the auto line.
    assert overlay.disp_item.opts["pen"].color().getRgb()[:3] == (255, 96, 0)
    assert overlay.disp_auto_item.opts["pen"].color().getRgb()[:3] == (0, 200, 255)
    man_x, _ = overlay.disp_item.getData()
    assert list(man_x) == [10.0, 50.0]  # src→dst of the manual CP
    auto_x, _ = overlay.disp_auto_item.getData()
    assert list(auto_x) == [30.0, 60.0]  # src→dst of the auto CP


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_empties_all_items(_qapp):
    overlay = ControlPointOverlay()
    overlay.set([(0.5, 0.5)], 100, 100, src_pts=[(0.1, 0.1)])
    overlay.clear()
    assert len(_spots(overlay)) == 0
    x, _ = overlay.disp_item.getData()
    assert x is None or len(x) == 0
