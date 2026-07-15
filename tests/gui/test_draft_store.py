"""Direct unit tests for DraftStore (the single per-(id, step) edit store)."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from verso.gui.draft_store import DraftStore, SectionDraft


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_mark_clear_dirty_idempotent_and_emit(_qapp):
    store = DraftStore()
    events: list[tuple[str, str]] = []
    store.dirty_changed.connect(lambda sid, step: events.append((sid, step)))

    store.mark_dirty("s0", "align")
    store.mark_dirty("s0", "align")  # idempotent
    assert store.is_dirty("s0", "align")
    assert store.any_dirty()
    assert store.dirty_keys() == [("s0", "align")]

    store.clear_dirty("s0", "align")
    store.clear_dirty("s0", "align")  # idempotent
    assert not store.is_dirty("s0", "align")
    assert not store.any_dirty()
    assert events == [("s0", "align"), ("s0", "align")]


def test_set_saved_first_stash_wins(_qapp):
    store = DraftStore()
    store.set_saved("s0", "prep", "saved")
    store.set_saved("s0", "prep", "later")
    assert store.get_saved("s0", "prep") == "saved"


def test_sync_saved_refreshes_only_while_clean(_qapp):
    store = DraftStore()
    store.sync_saved("s0", "align", "v1")
    assert store.get_saved("s0", "align") == "v1"
    store.sync_saved("s0", "align", "v2")  # still clean → refreshes
    assert store.get_saved("s0", "align") == "v2"

    store.mark_dirty("s0", "align")
    store.sync_saved("s0", "align", "mid-edit")  # dirty → no-op
    assert store.get_saved("s0", "align") == "v2"


def test_working_payload_roundtrip_and_prune(_qapp):
    store = DraftStore()
    store.set_working("s1", "prep", "mask")
    assert store.has_working("s1", "prep")
    assert store.get_working("s1", "prep") == "mask"

    assert store.pop_working("s1", "prep") == "mask"
    assert not store.has_working("s1", "prep")
    # Popping the only field leaves nothing behind: the entry is pruned.
    assert store.dirty_keys() == []
    assert store.get_saved("s1", "prep") is None


def test_pop_saved_keeps_dirty_entry(_qapp):
    store = DraftStore()
    store.sync_saved("s0", "align", "v1")
    store.mark_dirty("s0", "align")

    assert store.pop_saved("s0", "align") == "v1"
    assert store.get_saved("s0", "align") is None
    # Still dirty, so the entry survives the pop (not pruned).
    assert store.is_dirty("s0", "align")


def test_forget_section_drops_all_and_emits_for_dirty(_qapp):
    store = DraftStore()
    events: list[tuple[str, str]] = []
    store.dirty_changed.connect(lambda sid, step: events.append((sid, step)))

    store.mark_dirty("s0", "align")
    store.mark_dirty("s0", "warp")
    store.sync_saved("s0", "prep", "clean-baseline")  # clean entry, no emit
    store.mark_dirty("s2", "align")  # different section, untouched
    events.clear()

    store.forget_section("s0")

    assert not store.is_dirty("s0", "align")
    assert not store.is_dirty("s0", "warp")
    assert store.get_saved("s0", "prep") is None
    assert store.is_dirty("s2", "align")  # other sections untouched
    # Only the two dirty steps of s0 re-emit; the clean prep entry does not.
    assert sorted(events) == [("s0", "align"), ("s0", "warp")]


def test_clear_all_wipes_without_emitting(_qapp):
    store = DraftStore()
    events: list[tuple[str, str]] = []
    store.dirty_changed.connect(lambda sid, step: events.append((sid, step)))
    store.mark_dirty("s0", "align")
    store.set_working("s1", "prep", "mask")
    events.clear()

    store.clear_all()
    assert not store.any_dirty()
    assert not store.has_working("s1", "prep")
    assert events == []


def test_section_draft_defaults():
    entry = SectionDraft()
    assert entry.last_saved is None
    assert entry.working is None
    assert entry.dirty is False
