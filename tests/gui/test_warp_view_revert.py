"""Warp view "Clear edits" must discard unsaved control-point edits.

Regression test for a bug where re-activating the Warp view while the slice
was dirty re-snapshotted the baseline from the *edited* section, so a later
"Clear edits" (``revert``) restored the edits instead of dropping them and
flipped the status from dirty (yellow) to saved (green).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.model.alignment import ControlPoint, WarpState
from verso.engine.model.project import AtlasRef, Project, Section
from verso.gui.state import AppState
from verso.gui.views.warp_view import WarpView


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _section() -> Section:
    return Section(
        id="s0",
        slice_index=0,
        original_path="s0.png",
        thumbnail_path="thumbnails/s0.tif",
        warp=WarpState(control_points=[]),
    )


def _make_warp_mock(section: Section, state: AppState) -> SimpleNamespace:
    """SimpleNamespace that quacks like WarpView for activate/revert calls.

    Stubs out only the UI-touching members; the dirty / baseline logic runs
    against the real ``AppState``.
    """
    panel = SimpleNamespace(
        section=section,
        canvas=SimpleNamespace(set_interaction_mode=lambda _m: None),
        update_overlay=lambda: None,
        overlay_post_processor=None,
        cursor_to_atlas_mapper=None,
    )
    mock = SimpleNamespace(
        _panel=panel,
        _panel_slot=SimpleNamespace(
            layout=lambda: SimpleNamespace(addWidget=lambda _w: None)
        ),
        _state=state,
        _active=False,
        _cp_hovered=-1,
        _cp_dragging=-1,
        _cp_drag_start_norm=None,
        _cp_drag_start_dst=None,
        _baseline_warp=None,
        _dirty=False,
        _warp_overlay=lambda rgba: rgba,
        _cursor_to_src=lambda s, t: (s, t),
        _reset_undo=lambda: None,
        dirty_changed=SimpleNamespace(emit=lambda _v: None),
    )
    mock._set_dirty = lambda v: WarpView._set_dirty(mock, v)
    return mock


def test_clear_edits_discards_cps_after_navigating_away_and_back(_qapp):
    section = _section()
    state = AppState()
    state.load_project(
        Project(name="p", atlas=AtlasRef(name="allen_mouse_25um"), sections=[section])
    )

    mock = _make_warp_mock(section, state)

    # Enter Warp with a clean (no-CP) slice.
    WarpView.activate(mock)
    assert mock._dirty is False

    # User adds a control point → dirty. _set_dirty stashes the clean baseline.
    section.warp.control_points.append(ControlPoint(0.5, 0.5, 0.6, 0.6))
    state.mark_dirty(section.id, "warp")
    mock._set_dirty(True)

    # Navigate away and back: deactivate is a no-op for the baseline; activate
    # must NOT adopt the edited section.warp as the new baseline.
    WarpView.activate(mock)
    assert mock._dirty is True  # still dirty after re-activation

    # "Clear edits" must drop the unsaved control point and stay clean.
    assert WarpView.revert(mock) is True
    assert section.warp.control_points == []
    assert mock._dirty is False
