"""Flip-clears-alignment behaviour under the persistent-edits model.

Toggling a flip flag mutates the section in memory; the alignment is only
wiped when the user commits the prep edit via ``PrepView.save()`` and the
flip differs from the last-saved state (tracked by ``_prep_base_flip``).
Edits are no longer discarded on navigation, so there is no ``discard()``.
"""

from types import SimpleNamespace

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Preprocessing, Section
from verso.gui.views.prep_view import PrepView


class _FakeState:
    """Minimal AppState stand-in for the prep-draft store / dirty registry."""

    def pop_prep_draft(self, _section_id):
        return None

    def set_prep_draft(self, _section_id, _draft):
        pass

    def mark_dirty(self, _section_id, _step):
        pass

    def clear_dirty(self, _section_id, _step):
        pass

    def is_dirty(self, _section_id, _step):
        return False

    def set_baseline(self, _section_id, _step, _snapshot):
        pass

    def get_baseline(self, _section_id, _step):
        return None

    def pop_baseline(self, _section_id, _step):
        return None


def _make_prep_mock(section: Section, base_flip: tuple[bool, bool]) -> SimpleNamespace:
    """SimpleNamespace that quacks like PrepView for save() calls."""
    mock = SimpleNamespace(
        _section=section,
        _state=_FakeState(),
        _baseline_preprocessing=Preprocessing(),
        _prep_base_flip=base_flip,
        _mask_dirty=False,
        _current_mask=None,
        _lr_dirty=False,
        _lr_mask=None,
        _dirty=True,
        dirty_changed=SimpleNamespace(emit=lambda _v: None),
        alignment_invalidated=SimpleNamespace(emit=lambda: None),
    )
    mock._set_dirty = lambda v: PrepView._set_dirty(mock, v)
    return mock


def _stored_anchoring_section() -> Section:
    anchoring = [10.0, 20.0, 30.0, 100.0, 12.0, 0.0, 0.0, 0.0, 80.0]
    return Section(
        id="s001",
        serial_number=1,
        original_path="s001.png",
        thumbnail_path="s001.png",
        alignment=Alignment(
            anchoring=list(anchoring),
            stored_anchoring=list(anchoring),
            status=AlignmentStatus.COMPLETE,
        ),
    )


def test_save_with_flip_change_clears_alignment():
    section = _stored_anchoring_section()
    # User toggled the flip — section now reflects the unsaved state, while the
    # last-saved flip state was "no flip".
    section.preprocessing.flip_horizontal = True
    mock = _make_prep_mock(section, base_flip=(False, False))

    PrepView.save(mock)

    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.stored_anchoring is None
    assert section.alignment.source is None


def test_save_without_flip_change_preserves_alignment():
    section = _stored_anchoring_section()
    mock = _make_prep_mock(section, base_flip=(False, False))

    PrepView.save(mock)

    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring is not None


def test_unsaved_flip_does_not_wipe_alignment_before_save():
    section = _stored_anchoring_section()
    # Flip toggled but not yet saved — alignment must remain intact until save.
    section.preprocessing.flip_horizontal = True
    _make_prep_mock(section, base_flip=(False, False))

    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring is not None
