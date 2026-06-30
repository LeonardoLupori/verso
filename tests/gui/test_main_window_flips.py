"""Flip-invalidates-alignment behaviour under the persistent-edits model.

Toggling a flip changes the image coordinate frame, so the alignment + warp are
wiped the instant the flip is toggled (``MainWindow._invalidate_alignment_for_flip``)
rather than deferred to prep save.  Doing it at toggle time keeps a re-alignment
performed in the new orientation from being clobbered when the flip is later
saved (previously ``PrepView.save()`` / ``Ctrl+S`` wiped it).
"""

from types import SimpleNamespace

from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import Preprocessing, Section
from verso.gui.main_window import MainWindow
from verso.gui.views.prep_view import PrepView


class _FakeState:
    """Minimal AppState stand-in for the prep-draft store / dirty registry."""

    section_index = 0

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
        _dirty=True,
        dirty_changed=SimpleNamespace(emit=lambda _v: None),
        alignment_invalidated=SimpleNamespace(emit=lambda: None),
    )
    mock._set_dirty = lambda v: PrepView._set_dirty(mock, v)
    return mock


def _make_window_mock() -> SimpleNamespace:
    """SimpleNamespace that quacks like MainWindow for the flip-invalidate path."""
    mock = SimpleNamespace(
        _state=_FakeState(),
        _overview=SimpleNamespace(refresh_row=lambda _i: None),
        _refresh_filmstrip_dots=lambda: None,
    )
    mock._clear_alignment_view_state = lambda s: MainWindow._clear_alignment_view_state(mock, s)
    mock._seed_alignment_to_quicknii_default = lambda s: None
    mock._invalidate_alignment_for_flip = lambda s: MainWindow._invalidate_alignment_for_flip(
        mock, s
    )
    return mock


def _stored_anchoring_section() -> Section:
    anchoring = [10.0, 20.0, 30.0, 100.0, 12.0, 0.0, 0.0, 0.0, 80.0]
    return Section(
        id="s001",
        slice_index=1,
        original_path="s001.png",
        thumbnail_path="s001.png",
        alignment=Alignment(
            anchoring=list(anchoring),
            stored_anchoring=list(anchoring),
            status=AlignmentStatus.COMPLETE,
            source="manual",
        ),
        warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)], status=AlignmentStatus.COMPLETE),
    )


def test_flip_toggle_invalidates_alignment_and_warp():
    section = _stored_anchoring_section()
    window = _make_window_mock()

    window._invalidate_alignment_for_flip(section)

    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.stored_anchoring is None
    assert section.alignment.source is None
    assert section.warp.control_points == []
    assert section.warp.status == AlignmentStatus.NOT_STARTED


def test_flip_toggle_noop_when_nothing_aligned():
    section = Section(id="s002", slice_index=2, original_path="s.png", thumbnail_path="s.png")
    window = _make_window_mock()

    # No alignment to wipe — must not raise or fabricate state.
    window._invalidate_alignment_for_flip(section)

    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.anchoring == [0.0] * 9


def test_save_does_not_wipe_alignment_on_flip():
    """Saving a prep draft must never touch the alignment — a re-alignment done
    after the flip survives the save (the bug behind Ctrl+S resetting slices)."""
    section = _stored_anchoring_section()
    # Flip toggled but its invalidation already happened at toggle time; here the
    # user has re-aligned in the new frame, so save must preserve that work.
    section.preprocessing.flip_horizontal = True
    mock = _make_prep_mock(section, base_flip=(False, False))

    PrepView.save(mock)

    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring is not None


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
