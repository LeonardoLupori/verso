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
        STEP="warp",
        _panel=panel,
        _panel_slot=SimpleNamespace(layout=lambda: SimpleNamespace(addWidget=lambda _w: None)),
        _state=state,
        _active=False,
        _cp_hovered=-1,
        _cp_dragging=-1,
        _cp_drag_start_px=None,
        _cp_drag_start_dst=None,
        _warp_overlay=lambda rgba: rgba,
        _cursor_to_src=lambda s, t: (s, t),
        _reset_undo=lambda: None,
    )
    # Dirty flag + baseline are the single source of truth in AppState; the mock
    # drives them through the real BaseCanvasView / WarpView methods (which
    # resolve via WarpView's MRO), so the base save/revert flow runs for real.
    mock._current_section = lambda: WarpView._current_section(mock)
    mock._saved_state = lambda: WarpView._saved_state(mock)
    mock._saved_copy = lambda: WarpView._saved_copy(mock)
    mock._apply_saved = lambda baseline: WarpView._apply_saved(mock, baseline)
    mock._reset_cp_interaction = lambda: WarpView._reset_cp_interaction(mock)
    mock._set_dirty = lambda v: WarpView._set_dirty(mock, v)
    mock._after_revert = lambda: None
    return mock


def test_clear_edits_discards_cps_after_navigating_away_and_back(_qapp):
    section = _section()
    state = AppState()
    state.load_project(
        Project(name="p", atlas=AtlasRef(name="allen_mouse_25um"), sections=[section])
    )

    mock = _make_warp_mock(section, state)

    # Enter Warp with a clean (no-CP) slice; activate syncs the clean baseline.
    WarpView.activate(mock)
    assert state.is_dirty(section.id, "warp") is False

    # User adds a control point → dirty.
    section.warp.control_points.append(ControlPoint(500.0, 400.0, 600.0, 400.0))
    mock._set_dirty(True)
    assert state.is_dirty(section.id, "warp") is True

    # Navigate away and back: re-activation must NOT adopt the edited section.warp
    # as the new baseline (sync_baseline is a no-op while dirty).
    WarpView.activate(mock)
    assert state.is_dirty(section.id, "warp") is True  # still dirty after re-activation

    # "Clear edits" must drop the unsaved control point and stay clean.
    assert WarpView.revert(mock) is True
    assert section.warp.control_points == []
    assert state.is_dirty(section.id, "warp") is False


def test_inactive_view_ignores_shared_panel_section_loaded(_qapp):
    """A section_loaded for another slice must not clobber an inactive view.

    The canvas panel is shared, so its section_loaded signal reaches every
    view.  While Align (or Prep) owns the panel, the Warp view is inactive and
    must keep its dirty flag / baseline untouched — otherwise navigating slices
    or editing the alignment would silently disable Warp's "Clear edits".
    """
    section = _section()
    other = Section(
        id="s1",
        slice_index=1,
        original_path="s1.png",
        thumbnail_path="thumbnails/s1.tif",
        warp=WarpState(control_points=[]),
    )
    state = AppState()
    state.load_project(
        Project(
            name="p",
            atlas=AtlasRef(name="allen_mouse_25um"),
            sections=[section, other],
        )
    )

    mock = _make_warp_mock(section, state)

    # Warp is dirty on `section`.
    WarpView.activate(mock)
    section.warp.control_points.append(ControlPoint(500.0, 400.0, 600.0, 400.0))
    mock._set_dirty(True)
    saved_baseline = state.get_baseline(section.id, "warp")

    # Leave Warp; another view now owns the panel and loads a different slice.
    mock._active = False
    WarpView._on_section_loaded(mock, other)

    # The inactive Warp view must be untouched: `section` stays dirty and its
    # stashed baseline is not overwritten.
    assert state.is_dirty(section.id, "warp") is True
    assert state.get_baseline(section.id, "warp") is saved_baseline
