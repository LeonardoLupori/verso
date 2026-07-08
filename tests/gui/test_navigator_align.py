"""Align-view rework: navigator parallel-view removal, tilt clamp, canvas handle.

Covers:
- NavigatorPanel hides exactly the view parallel to the slices (the one whose
  axis is the project's slicing axis), for every axis.
- ``_SliceView`` tilt rotations are clamped to ±45° from the slicing axis.
- The on-canvas align handle classifies translate/rotate/none zones and is only
  visible in Align mode with an overlay present.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.anchoring import plane_tilt_deg, series_default_anchoring
from verso.gui.widgets.align_handle import _HANDLE_GRIP_PX, _HANDLE_RING_PX
from verso.gui.widgets.canvas import ImageCanvas
from verso.gui.widgets.navigator import NavigatorPanel, _SliceView

_DIMS = (528, 320, 456)  # (AP, DV, LR)


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _default_anchoring(axis: int) -> list[float]:
    return series_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=_DIMS,
        interpolation_axis=axis,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_navigator_hides_only_parallel_view(_qapp, axis):
    panel = NavigatorPanel()
    panel.set_interpolation_axis(axis)
    for ax in (0, 1, 2):
        assert panel._groups[ax].isHidden() == (ax == axis)


@pytest.mark.parametrize("view_axis", [0, 2])  # the two tilt views for a coronal project
def test_slice_view_tilt_clamped_to_44(_qapp, view_axis):
    view = _SliceView(view_axis, _DIMS)
    view.set_interpolation_axis(1)  # coronal project
    view._anchoring = _default_anchoring(1)
    view.anchoring_changed.connect(lambda a: setattr(view, "_anchoring", list(a)))

    # Push well past the limit in 1° steps; tilt must never exceed 44°.
    for _ in range(120):
        view._rotate_step(1.0)
        assert plane_tilt_deg(view._anchoring, 1) <= 44.0 + 1e-6
    assert plane_tilt_deg(view._anchoring, 1) == pytest.approx(44.0, abs=1e-3)

    # Rotating back must be allowed (the lock is not sticky).
    for _ in range(10):
        view._rotate_step(-1.0)
    assert plane_tilt_deg(view._anchoring, 1) == pytest.approx(34.0, abs=1e-3)


def test_align_handle_zones_and_visibility(_qapp):
    canvas = ImageCanvas()
    canvas.set_interaction_mode("align")
    # No overlay yet → hidden.
    assert not canvas._align_handle.isVisible()

    canvas.set_overlay(np.zeros((40, 50, 4), dtype=np.uint8), 50, 40)
    handle = canvas._align_handle
    assert handle.isVisible()  # align + overlay

    # Centre is the overlay centre; classify at 1 view-unit per pixel. Offsets
    # use the handle's own geometry constants so they track tuning changes.
    assert handle.zone_at(25.0, 20.0, 1.0) == "translate"  # inert centre → default
    assert handle.zone_at(25.0 + _HANDLE_GRIP_PX, 20.0, 1.0) == "stretch_x"  # E grip
    assert handle.zone_at(25.0, 20.0 + _HANDLE_GRIP_PX, 1.0) == "stretch_y"  # S grip
    assert handle.zone_at(25.0 + _HANDLE_RING_PX, 20.0, 1.0) == "rotate"  # on the ring line
    assert handle.zone_at(25.0 + 20.0, 20.0, 1.0) == "translate"  # ring interior
    assert handle.zone_at(25.0 + 500.0, 20.0, 1.0) == "translate"  # far field

    # Hidden handle classifies nothing.
    canvas.set_interaction_mode("warp")
    assert not canvas._align_handle.isVisible()
    assert handle.zone_at(25.0, 20.0, 1.0) is None
