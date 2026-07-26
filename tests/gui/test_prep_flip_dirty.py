"""Prep's dirty flag tracks the flips against the baseline, it doesn't latch.

Toggling a flip twice puts the image back in its last-saved orientation, so the
slice has nothing left to save and its filmstrip dot must go back to the saved
colour instead of staying yellow.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section
from verso.gui.state import AppState
from verso.gui.views.prep_view import PrepView


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
        preprocessing=Preprocessing(),
    )


def _make_prep_mock(section: Section, state: AppState) -> SimpleNamespace:
    """SimpleNamespace that quacks like PrepView for the dirty-flag calls.

    No widgets are touched: the flip/dirty logic runs for real against the real
    ``AppState``, which is the single source of truth for the save bar and dot.
    """
    mock = SimpleNamespace(STEP="prep", _section=section, _state=state)
    mock._current_section = lambda: PrepView._current_section(mock)
    mock._saved_state = lambda: PrepView._saved_state(mock)
    mock._set_dirty = lambda v: PrepView._set_dirty(mock, v)
    mock._flip_is_dirty = lambda: PrepView._flip_is_dirty(mock)
    mock.mark_flip_changed = lambda: PrepView.mark_flip_changed(mock)
    return mock


def _state_with(section: Section) -> AppState:
    state = AppState()
    state.load_project(
        Project(name="p", atlas=AtlasRef(name="allen_mouse_25um"), sections=[section])
    )
    # What load_section stashes: the last-saved Preprocessing baseline.
    state.sync_baseline(section.id, "prep", Preprocessing())
    return state


def test_flipping_back_clears_the_dirty_flag(_qapp):
    section = _section()
    state = _state_with(section)
    mock = _make_prep_mock(section, state)

    section.preprocessing.flip_horizontal = True
    mock.mark_flip_changed()
    assert state.is_dirty(section.id, "prep") is True

    section.preprocessing.flip_horizontal = False
    mock.mark_flip_changed()
    assert state.is_dirty(section.id, "prep") is False


def test_flipping_back_stays_dirty_while_a_mask_edit_is_pending(_qapp):
    """The two prep sub-edits are one dirty flag — an unsaved mask still counts."""
    section = _section()
    state = _state_with(section)
    mock = _make_prep_mock(section, state)

    state.set_working(section.id, "prep", np.zeros((4, 4), dtype=bool))
    section.preprocessing.flip_vertical = True
    mock.mark_flip_changed()
    section.preprocessing.flip_vertical = False
    mock.mark_flip_changed()

    assert state.is_dirty(section.id, "prep") is True


def test_flip_with_no_baseline_is_dirty(_qapp):
    """Before a baseline is stashed there is nothing to compare to — treat the
    flip as an edit rather than silently dropping it."""
    section = _section()
    state = AppState()
    state.load_project(
        Project(name="p", atlas=AtlasRef(name="allen_mouse_25um"), sections=[section])
    )
    mock = _make_prep_mock(section, state)

    section.preprocessing.flip_horizontal = True
    mock.mark_flip_changed()

    assert state.is_dirty(section.id, "prep") is True


def test_second_flip_axis_keeps_it_dirty(_qapp):
    """H then V is not a revert — both differ from the baseline."""
    section = _section()
    state = _state_with(section)
    mock = _make_prep_mock(section, state)

    section.preprocessing.flip_horizontal = True
    mock.mark_flip_changed()
    section.preprocessing.flip_vertical = True
    mock.mark_flip_changed()

    assert state.is_dirty(section.id, "prep") is True
