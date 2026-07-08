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
from verso.engine.model.project import DialogPrefs, Section
from verso.gui.main_window import MainWindow


class _FakeState:
    """Minimal AppState stand-in for the dirty registry / baselines."""

    section_index = 0

    def __init__(self, *, project=None, dirty_steps=()):
        self.project = project
        self._dirty_steps = set(dirty_steps)

    def clear_dirty(self, _section_id, _step):
        pass

    def pop_baseline(self, _section_id, _step):
        return None

    def is_dirty(self, _section_id, step):
        return step in self._dirty_steps


def _make_window_mock(state: _FakeState | None = None) -> SimpleNamespace:
    """SimpleNamespace that quacks like MainWindow for the flip-invalidate path."""
    mock = SimpleNamespace(
        _state=state if state is not None else _FakeState(),
        _overview=SimpleNamespace(refresh_row=lambda _i: None),
        _refresh_filmstrip_dots=lambda: None,
    )
    mock._clear_alignment_view_state = lambda s: MainWindow._clear_alignment_view_state(mock, s)
    mock._seed_alignment_to_quicknii_default = lambda s: None
    mock._invalidate_alignment_for_flip = lambda s: MainWindow._invalidate_alignment_for_flip(
        mock, s
    )
    mock._confirm_flip = lambda s: MainWindow._confirm_flip(mock, s)
    return mock


def _stored_anchoring_section() -> Section:
    anchoring = [10.0, 20.0, 30.0, 100.0, 12.0, 0.0, 0.0, 0.0, 80.0]
    return Section(
        id="s001",
        slice_index=1,
        original_path="s001.png",
        thumbnail_path="s001.png",
        alignment=Alignment(
            current_anchoring=list(anchoring),
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

    assert section.alignment.current_anchoring == [0.0] * 9
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
    assert section.alignment.current_anchoring == [0.0] * 9


def _project_with_dialog_enabled():
    return SimpleNamespace(dialog_prefs=DialogPrefs(show_align_deletion=True))


def test_confirm_flip_skips_dialog_for_auto_interpolated_alignment(monkeypatch):
    """A section only carrying ``interpolate_anchorings``'s default guess.

    ``status`` is IN_PROGRESS and ``anchoring`` is non-zero, but nothing was ever
    saved (``stored_anchoring`` is None) and there is no unsaved edit — so the
    flip must proceed silently, matching "no alignment at all" from the user's
    perspective.
    """
    section = Section(
        id="s003",
        slice_index=3,
        original_path="s.png",
        thumbnail_path="s.png",
        alignment=Alignment(
            current_anchoring=[10.0, 20.0, 30.0, 100.0, 12.0, 0.0, 0.0, 0.0, 80.0],
            status=AlignmentStatus.IN_PROGRESS,
            source="quicknii_default",
        ),
    )
    state = _FakeState(project=_project_with_dialog_enabled())
    window = _make_window_mock(state)

    called = False

    def _fake_dialog(_parent):
        nonlocal called
        called = True
        return True, False

    monkeypatch.setattr(
        "verso.gui.dialogs.flip_warning.confirm_flip_deletes_alignment", _fake_dialog
    )

    assert window._confirm_flip(section) is True
    assert called is False


def test_confirm_flip_shows_dialog_for_saved_alignment(monkeypatch):
    section = _stored_anchoring_section()
    state = _FakeState(project=_project_with_dialog_enabled())
    window = _make_window_mock(state)

    called = False

    def _fake_dialog(_parent):
        nonlocal called
        called = True
        return True, False

    monkeypatch.setattr(
        "verso.gui.dialogs.flip_warning.confirm_flip_deletes_alignment", _fake_dialog
    )

    assert window._confirm_flip(section) is True
    assert called is True


def test_confirm_flip_shows_dialog_for_dirty_unsaved_alignment(monkeypatch):
    """No saved alignment yet, but the align step has an in-progress unsaved edit."""
    section = Section(id="s004", slice_index=4, original_path="s.png", thumbnail_path="s.png")
    state = _FakeState(project=_project_with_dialog_enabled(), dirty_steps={"align"})
    window = _make_window_mock(state)

    called = False

    def _fake_dialog(_parent):
        nonlocal called
        called = True
        return True, False

    monkeypatch.setattr(
        "verso.gui.dialogs.flip_warning.confirm_flip_deletes_alignment", _fake_dialog
    )

    assert window._confirm_flip(section) is True
    assert called is True
