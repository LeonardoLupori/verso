"""AppState edit registry + resident prep-draft store behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.drafts import PrepDraft
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
            serial_number=i,
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


def test_prep_draft_store_roundtrip(_qapp):
    state = AppState()
    state.load_project(_project())
    mask = np.ones((2, 2), dtype=bool)
    state.set_prep_draft("s1", PrepDraft(slice_mask=mask, mask_dirty=True))

    assert state.has_prep_draft("s1")
    draft = state.pop_prep_draft("s1")
    assert draft is not None and draft.mask_dirty
    assert not state.has_prep_draft("s1")  # popped


def test_load_project_clears_registry_and_drafts(_qapp):
    state = AppState()
    state.load_project(_project())
    state.mark_dirty("s0", "align")
    state.set_prep_draft("s1", PrepDraft(mask_dirty=True))

    state.load_project(_project())  # fresh load wipes unsaved edits
    assert not state.any_dirty()
    assert not state.has_prep_draft("s1")


def test_clear_all_edits(_qapp):
    state = AppState()
    state.load_project(_project())
    state.mark_dirty("s0", "align")
    state.set_prep_draft("s1", PrepDraft(mask_dirty=True))

    state.clear_all_edits()
    assert not state.any_dirty()
    assert not state.has_prep_draft("s1")
