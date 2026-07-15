"""AppState edit registry + resident prep-draft store behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.model.project import AtlasRef, Project, Section
from verso.gui.state import AppState


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _project() -> Project:
    sections = [
        Section(
            id=f"s{i}",
            slice_index=i,
            original_path=f"s{i}.png",
            thumbnail_path=f"thumbnails/s{i}.tif",
        )
        for i in range(3)
    ]
    return Project(name="p", atlas=AtlasRef(name="allen_mouse_25um"), sections=sections)


def test_mark_and_clear_dirty_emit_once(_qapp):
    state = AppState()
    state.load_project(_project())
    events: list[tuple[str, str]] = []
    state.dirty_changed.connect(lambda sid, step: events.append((sid, step)))

    state.mark_dirty("s0", "align")
    state.mark_dirty("s0", "align")  # idempotent — no second emit
    assert state.is_dirty("s0", "align")
    assert events == [("s0", "align")]

    state.clear_dirty("s0", "align")
    state.clear_dirty("s0", "align")  # idempotent
    assert not state.is_dirty("s0", "align")
    assert events == [("s0", "align"), ("s0", "align")]


def test_dirty_sections_groups_steps(_qapp):
    state = AppState()
    state.load_project(_project())
    state.mark_dirty("s0", "align")
    state.mark_dirty("s0", "warp")
    state.mark_dirty("s2", "prep")

    grouped = {section.id: steps for section, steps in state.dirty_sections()}
    assert grouped == {"s0": {"align", "warp"}, "s2": {"prep"}}
    assert state.any_dirty()


def test_prep_working_store_roundtrip(_qapp):
    state = AppState()
    state.load_project(_project())
    mask = np.ones((2, 2), dtype=bool)
    state.set_working("s1", "prep", mask)

    assert state.has_working("s1", "prep")
    stored = state.pop_working("s1", "prep")
    assert stored is mask
    assert not state.has_working("s1", "prep")  # popped


def test_load_project_clears_registry_and_drafts(_qapp):
    state = AppState()
    state.load_project(_project())
    state.mark_dirty("s0", "align")
    state.set_working("s1", "prep", np.ones((2, 2), dtype=bool))

    state.load_project(_project())  # fresh load wipes unsaved edits
    assert not state.any_dirty()
    assert not state.has_working("s1", "prep")


def test_sync_baseline_refreshes_while_clean_but_not_while_dirty(_qapp):
    state = AppState()
    state.load_project(_project())

    # Clean: sync stores the supplied snapshot as the baseline.
    state.sync_baseline("s0", "align", "saved-v1")
    assert state.get_baseline("s0", "align") == "saved-v1"

    # Still clean: a later sync refreshes it (e.g. re-loading a clean section).
    state.sync_baseline("s0", "align", "saved-v2")
    assert state.get_baseline("s0", "align") == "saved-v2"

    # Dirty: the stashed last-saved value must survive navigation, so sync is a
    # no-op even though the section object may now hold the unsaved edit.
    state.mark_dirty("s0", "align")
    state.sync_baseline("s0", "align", "dirty-edit")
    assert state.get_baseline("s0", "align") == "saved-v2"


def test_set_baseline_keeps_first_stash(_qapp):
    state = AppState()
    state.load_project(_project())

    # Batch flows (DeepSlice / masks) dirty a section after mutating it and stash
    # the pre-edit snapshot via set_baseline; the first stash wins.
    state.set_baseline("s0", "prep", "saved")
    state.set_baseline("s0", "prep", "later")
    assert state.get_baseline("s0", "prep") == "saved"


def test_clear_all_edits(_qapp):
    state = AppState()
    state.load_project(_project())
    state.mark_dirty("s0", "align")
    state.set_working("s1", "prep", np.ones((2, 2), dtype=bool))

    state.clear_all_edits()
    assert not state.any_dirty()
    assert not state.has_working("s1", "prep")
