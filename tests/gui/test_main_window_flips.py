"""Flip-clears-alignment behaviour under the new draft model.

Toggling a flip flag is now a draft mutation: the alignment is only
wiped when the user commits the draft via PrepView.save().  Discarding
the draft (e.g., switching slice/view) rolls the flip back and leaves
the alignment intact.
"""

from types import SimpleNamespace

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Preprocessing, Section
from verso.gui.views.prep_view import PrepView


def _make_prep_mock(section: Section, baseline: Preprocessing) -> SimpleNamespace:
    """SimpleNamespace that quacks like PrepView for save()/discard() calls."""
    mock = SimpleNamespace(
        _section=section,
        _baseline_preprocessing=baseline,
        _mask_dirty=False,
        _current_mask=None,
        _lr_dirty=False,
        _lr_mask=None,
        _undo_stack=[],
        _stroke_points=[],
        _stroke_active=False,
        _lr_draw_mode=False,
        _raw_image=None,
        _dirty=True,
        dirty_changed=SimpleNamespace(emit=lambda _v: None),
        lr_status_changed=SimpleNamespace(emit=lambda: None),
        cancel_lr_draw_if_active=lambda: False,
    )
    mock._wipe_alignment_for_flip = lambda: PrepView._wipe_alignment_for_flip(mock)
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
    baseline = Preprocessing()  # flip flags both False
    # User toggled the flip — section now reflects the draft state.
    section.preprocessing.flip_horizontal = True
    mock = _make_prep_mock(section, baseline)

    PrepView.save(mock)

    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.stored_anchoring is None
    assert section.alignment.source is None


def test_save_without_flip_change_preserves_alignment():
    section = _stored_anchoring_section()
    baseline = Preprocessing()
    mock = _make_prep_mock(section, baseline)

    PrepView.save(mock)

    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring is not None


def test_discard_reverts_flip_and_preserves_alignment():
    section = _stored_anchoring_section()
    baseline = Preprocessing()
    # User toggled the flip without saving — discard should undo it.
    section.preprocessing.flip_horizontal = True
    mock = _make_prep_mock(section, baseline)

    PrepView.discard(mock)

    assert section.preprocessing.flip_horizontal is False
    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring is not None
