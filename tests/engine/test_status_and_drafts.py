"""Unit tests for the per-step status helper and draft persistence ops."""

from __future__ import annotations

import numpy as np

from verso.engine.drafts import (
    PrepDraft,
    commit_alignment,
    commit_warp,
    lr_mask_path_for,
    persist_prep_draft,
    slice_mask_path_for,
    wipe_alignment_for_flip,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import Preprocessing, Section
from verso.engine.model.status import STATUS_COLOR, section_step_status


def _section(**kw) -> Section:
    return Section(
        id=kw.get("id", "s1"),
        slice_index=1,
        original_path=kw.get("original_path", "img.png"),
        thumbnail_path=kw.get("thumbnail_path", "thumbnails/img.tif"),
        preprocessing=kw.get("preprocessing", Preprocessing()),
        alignment=kw.get("alignment", Alignment()),
        warp=kw.get("warp", WarpState()),
    )


# ---------------------------------------------------------------------------
# section_step_status
# ---------------------------------------------------------------------------


def test_dirty_is_in_progress_for_prep_and_align():
    section = _section()
    for step in ("prep", "align"):
        assert section_step_status(section, step, dirty=True) == AlignmentStatus.IN_PROGRESS


def test_warp_empty_is_gray_even_when_dirty():
    # Removing the last control point keeps the step dirty but it must read gray.
    empty = _section()
    assert section_step_status(empty, "warp", dirty=True) == AlignmentStatus.NOT_STARTED

    with_cps = _section(warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)]))
    assert section_step_status(with_cps, "warp", dirty=True) == AlignmentStatus.IN_PROGRESS


def test_prep_status_from_saved_state():
    blank = _section()
    assert section_step_status(blank, "prep", dirty=False) == AlignmentStatus.NOT_STARTED

    with_mask = _section(preprocessing=Preprocessing(slice_mask_path="m.png"))
    assert section_step_status(with_mask, "prep", dirty=False) == AlignmentStatus.COMPLETE

    flipped = _section(preprocessing=Preprocessing(flip_horizontal=True))
    assert section_step_status(flipped, "prep", dirty=False) == AlignmentStatus.COMPLETE


def test_align_status_green_only_when_complete():
    interp = _section(alignment=Alignment(anchoring=[1.0] * 9, status=AlignmentStatus.IN_PROGRESS))
    # An interpolated/proposed plane is not a user save → gray, not green.
    assert section_step_status(interp, "align", dirty=False) == AlignmentStatus.NOT_STARTED

    saved = _section(alignment=Alignment(anchoring=[1.0] * 9, status=AlignmentStatus.COMPLETE))
    assert section_step_status(saved, "align", dirty=False) == AlignmentStatus.COMPLETE


def test_warp_status_green_with_control_points():
    cps = _section(warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)]))
    assert section_step_status(cps, "warp", dirty=False) == AlignmentStatus.COMPLETE

    none = _section()
    assert section_step_status(none, "warp", dirty=False) == AlignmentStatus.NOT_STARTED


def test_status_color_covers_all_states():
    for status in AlignmentStatus:
        assert status in STATUS_COLOR


# ---------------------------------------------------------------------------
# commit_alignment / commit_warp / wipe
# ---------------------------------------------------------------------------


def test_commit_alignment_promotes_to_stored_complete():
    section = _section(alignment=Alignment(anchoring=[1.0, 2.0] + [0.0] * 7))
    assert commit_alignment(section) is True
    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring == section.alignment.anchoring


def test_commit_alignment_noop_on_zero_plane():
    section = _section()
    assert commit_alignment(section) is False
    assert section.alignment.status == AlignmentStatus.NOT_STARTED


def test_commit_warp_requires_complete_alignment():
    section = _section()
    assert commit_warp(section) is False
    section.alignment.status = AlignmentStatus.COMPLETE
    assert commit_warp(section) is True
    assert section.warp.status == AlignmentStatus.COMPLETE


def test_wipe_alignment_for_flip_resets_everything():
    section = _section(
        alignment=Alignment(
            anchoring=[1.0] * 9,
            stored_anchoring=[1.0] * 9,
            status=AlignmentStatus.COMPLETE,
            source="manual",
        ),
        warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)], status=AlignmentStatus.COMPLETE),
    )
    wipe_alignment_for_flip(section)
    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.stored_anchoring is None
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.warp.control_points == []
    assert section.warp.status == AlignmentStatus.NOT_STARTED


# ---------------------------------------------------------------------------
# persist_prep_draft
# ---------------------------------------------------------------------------


def test_persist_prep_draft_writes_mask_and_sets_path(tmp_path):
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    draft = PrepDraft(slice_mask=mask, mask_dirty=True)

    flip_changed = persist_prep_draft(section, draft)

    assert flip_changed is False
    expected = slice_mask_path_for(section)
    assert expected.exists()
    assert section.preprocessing.slice_mask_path == str(expected)


def test_persist_prep_draft_flip_change_wipes_alignment(tmp_path):
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
        alignment=Alignment(
            anchoring=[1.0] * 9, stored_anchoring=[1.0] * 9, status=AlignmentStatus.COMPLETE
        ),
    )
    section.preprocessing.flip_horizontal = True  # current differs from base (False)
    draft = PrepDraft(base_flip_h=False, base_flip_v=False)

    assert persist_prep_draft(section, draft) is True
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.stored_anchoring is None


def test_lr_mask_path_distinct_from_slice(tmp_path):
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
    )
    assert slice_mask_path_for(section) != lr_mask_path_for(section)
