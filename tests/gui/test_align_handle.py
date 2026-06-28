"""Unit tests for the Align-view centre manipulator gizmo (``AlignHandle``).

Covers the three responsibilities the handle owns on its own (no canvas needed):
- ``zone_at`` hit-testing of the ring / stretch grips / translate field, including
  zoom scaling via ``view_px`` and the hidden-handle short-circuit;
- ``set_hover_zone`` brightening state (which zones count as a hover);
- the drag math (``rotate_delta`` / ``stretch_delta``) moved off the viewbox.
"""

from __future__ import annotations

import math

import pytest
from PyQt6.QtWidgets import QApplication

from verso.gui.widgets.align_handle import (
    _HANDLE_GRIP_PX,
    _HANDLE_RING_PX,
    _STRETCH_GAIN,
    AlignHandle,
)


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _handle(cx: float = 100.0, cy: float = 80.0) -> AlignHandle:
    h = AlignHandle()
    h.set_center(cx, cy)
    h.set_active(True)  # visible so zone_at classifies
    return h


# ---------------------------------------------------------------------------
# zone_at
# ---------------------------------------------------------------------------


def test_zone_at_classifies_each_region(_qapp):
    h = _handle()
    cx, cy = 100.0, 80.0
    # Centre and ring interior fall through to the default translate field.
    assert h.zone_at(cx, cy, 1.0) == "translate"
    assert h.zone_at(cx + 20.0, cy, 1.0) == "translate"
    # Grips sit on each axis at _HANDLE_GRIP_PX from the centre.
    assert h.zone_at(cx + _HANDLE_GRIP_PX, cy, 1.0) == "stretch_x"
    assert h.zone_at(cx - _HANDLE_GRIP_PX, cy, 1.0) == "stretch_x"
    assert h.zone_at(cx, cy + _HANDLE_GRIP_PX, 1.0) == "stretch_y"
    assert h.zone_at(cx, cy - _HANDLE_GRIP_PX, 1.0) == "stretch_y"
    # The ring line itself rotates; well outside it is translate again.
    assert h.zone_at(cx + _HANDLE_RING_PX, cy, 1.0) == "rotate"
    assert h.zone_at(cx + 500.0, cy, 1.0) == "translate"


def test_zone_at_scales_with_view_px(_qapp):
    """Thresholds are in screen px, so a grip's view-space offset scales by view_px."""
    h = _handle()
    cx, cy = 100.0, 80.0
    view_px = 3.0  # 3 view units per screen pixel (zoomed out)
    assert h.zone_at(cx + _HANDLE_GRIP_PX * view_px, cy, view_px) == "stretch_x"
    # The same view-space offset at view_px=1 lands far outside the grip.
    assert h.zone_at(cx + _HANDLE_GRIP_PX * view_px, cy, 1.0) == "translate"


def test_zone_at_returns_none_when_hidden_or_bad_scale(_qapp):
    h = _handle()
    h.set_active(False)
    assert h.zone_at(100.0, 80.0, 1.0) is None
    h.set_active(True)
    assert h.zone_at(100.0, 80.0, 0.0) is None  # non-positive view_px


# ---------------------------------------------------------------------------
# set_hover_zone
# ---------------------------------------------------------------------------


def test_set_hover_zone_normalizes_non_brightening_zones(_qapp):
    h = _handle()
    h.set_hover_zone("rotate")
    assert h._hover_zone == "rotate"
    h.set_hover_zone("stretch_y")
    assert h._hover_zone == "stretch_y"
    # translate, an unknown string, and None all mean "no element hovered".
    h.set_hover_zone("translate")
    assert h._hover_zone is None
    h.set_hover_zone("rotate")
    h.set_hover_zone(None)
    assert h._hover_zone is None


# ---------------------------------------------------------------------------
# rotate_delta
# ---------------------------------------------------------------------------


def test_rotate_delta_signed_angle_about_centre(_qapp):
    h = _handle(0.0, 0.0)
    # +x axis to +y axis is a quarter turn; sign follows atan2 (image coords).
    assert h.rotate_delta(10.0, 0.0, 0.0, 10.0) == pytest.approx(90.0)
    # Reverse direction → opposite sign.
    assert h.rotate_delta(0.0, 10.0, 10.0, 0.0) == pytest.approx(-90.0)
    # No movement → no rotation.
    assert h.rotate_delta(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)


def test_rotate_delta_uses_handle_centre(_qapp):
    h = _handle(50.0, 50.0)
    # Points straddling the centre on the two axes → quarter turn.
    assert h.rotate_delta(60.0, 50.0, 50.0, 60.0) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# stretch_delta
# ---------------------------------------------------------------------------


def test_stretch_delta_inverse_ratio_and_axis(_qapp):
    h = _handle(0.0, 0.0)
    # Pull the x-grip outward (10 → 40): inverse ratio softened by the gain.
    s, t = h.stretch_delta("stretch_x", 10.0, 0.0, 40.0, 0.0, 1.0)
    assert s == pytest.approx((10.0 / 40.0) ** _STRETCH_GAIN)
    assert t == 1.0
    # stretch_y drives the second factor instead.
    s, t = h.stretch_delta("stretch_y", 0.0, 10.0, 0.0, 40.0, 1.0)
    assert s == 1.0
    assert t == pytest.approx((10.0 / 40.0) ** _STRETCH_GAIN)


def test_stretch_delta_clamped_to_half_two(_qapp):
    h = _handle(0.0, 0.0)
    # Extreme outward pull would undershoot 0.5 without the clamp.
    s, _ = h.stretch_delta("stretch_x", 1.0, 0.0, 1000.0, 0.0, 1.0)
    assert s == pytest.approx(0.5)
    # Extreme inward pull would overshoot 2.0 without the clamp.
    s, _ = h.stretch_delta("stretch_x", 1000.0, 0.0, 1.0, 0.0, 1.0)
    assert s == pytest.approx(2.0)


def test_stretch_delta_centre_floor_avoids_blowup(_qapp):
    h = _handle(0.0, 0.0)
    # Both points at the centre would be 0/0; the view_px floor keeps it finite
    # and (equal distances) yields no scaling.
    s, t = h.stretch_delta("stretch_x", 0.0, 0.0, 0.0, 0.0, 1.0)
    assert math.isfinite(s)
    assert s == pytest.approx(1.0)
    assert t == 1.0
