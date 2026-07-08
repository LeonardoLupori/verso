"""Flip-invalidates-alignment behaviour under the persistent-edits model.

Toggling a flip changes the image coordinate frame, so the alignment + warp are
wiped the instant the flip is toggled (``MainWindow._invalidate_alignment_for_flip``)
rather than deferred to prep save.  Doing it at toggle time keeps a re-alignment
performed in the new orientation from being clobbered when the flip is later
saved.  (That prep save itself never touches the alignment is covered at the
engine level by ``test_commit_prep_draft_preserves_alignment_through_flip``.)
"""

from types import SimpleNamespace

from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import Section
from verso.gui.main_window import MainWindow


class _FakeState:
    """Minimal AppState stand-in for the dirty registry / baselines."""

    section_index = 0

    def clear_dirty(self, _section_id, _step):
        pass

    def pop_baseline(self, _section_id, _step):
        return None


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
