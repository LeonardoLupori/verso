"""Unit tests for the per-step status helper and draft persistence ops."""

from __future__ import annotations

import numpy as np

from verso.engine.drafts import (
    commit_alignment,
    commit_prep_draft,
    commit_warp,
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
    interp = _section(
        alignment=Alignment(current_anchoring=[1.0] * 9, status=AlignmentStatus.IN_PROGRESS)
    )
    # An interpolated/proposed plane is not a user save → gray, not green.
    assert section_step_status(interp, "align", dirty=False) == AlignmentStatus.NOT_STARTED

    saved = _section(
        alignment=Alignment(current_anchoring=[1.0] * 9, status=AlignmentStatus.COMPLETE)
    )
    assert section_step_status(saved, "align", dirty=False) == AlignmentStatus.COMPLETE


def test_warp_status_green_with_control_points():
    cps = _section(warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)]))
    assert section_step_status(cps, "warp", dirty=False) == AlignmentStatus.COMPLETE

    none = _section()
    assert section_step_status(none, "warp", dirty=False) == AlignmentStatus.NOT_STARTED


def test_warp_auto_only_is_yellow():
    # A purely auto-generated (elastix) warp is a proposal awaiting review.
    auto = _section(warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0, auto=True)]))
    assert section_step_status(auto, "warp", dirty=False) == AlignmentStatus.IN_PROGRESS

    # Saving accepts the proposal (status → COMPLETE) → green.
    accepted = _section(
        warp=WarpState(
            control_points=[ControlPoint(0, 0, 0, 0, auto=True)],
            status=AlignmentStatus.COMPLETE,
        )
    )
    assert section_step_status(accepted, "warp", dirty=False) == AlignmentStatus.COMPLETE

    # One hand-placed point makes the warp user-owned → green when saved.
    mixed = _section(
        warp=WarpState(
            control_points=[
                ControlPoint(0, 0, 0, 0, auto=True),
                ControlPoint(1, 1, 1, 1, auto=False),
            ]
        )
    )
    assert section_step_status(mixed, "warp", dirty=False) == AlignmentStatus.COMPLETE


def test_status_color_covers_all_states():
    for status in AlignmentStatus:
        assert status in STATUS_COLOR


# ---------------------------------------------------------------------------
# commit_alignment / commit_warp / wipe
# ---------------------------------------------------------------------------


def test_commit_alignment_promotes_to_stored_complete():
    section = _section(alignment=Alignment(current_anchoring=[1.0, 2.0] + [0.0] * 7))
    assert commit_alignment(section) is True
    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring == section.alignment.current_anchoring


def test_commit_alignment_noop_on_zero_plane():
    section = _section()
    assert commit_alignment(section) is False
    assert section.alignment.status == AlignmentStatus.NOT_STARTED


def test_commit_warp_empty_resets_to_not_started():
    # A warp with no control points is not a finished warp: it resets to
    # NOT_STARTED even when a usable plane exists, matching the per-view save.
    section = _section(
        alignment=Alignment(current_anchoring=[1.0] * 9, status=AlignmentStatus.COMPLETE)
    )
    section.warp.status = AlignmentStatus.COMPLETE
    assert commit_warp(section) is False
    assert section.warp.status == AlignmentStatus.NOT_STARTED


def test_commit_warp_requires_usable_plane():
    section = _section(warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)]))
    assert commit_warp(section) is False
    section.alignment.status = AlignmentStatus.COMPLETE
    assert commit_warp(section) is True
    assert section.warp.status == AlignmentStatus.COMPLETE


def test_commit_warp_promotes_proposal_plane():
    # A warp placed on an uncommitted (e.g. quicknii_default) plane must promote
    # that plane to COMPLETE so the next save's interpolation can't overwrite it.
    section = _section(
        alignment=Alignment(
            current_anchoring=[1.0, 2.0] + [0.0] * 7,
            status=AlignmentStatus.IN_PROGRESS,
            source="quicknii_default",
        ),
        warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)]),
    )
    assert commit_warp(section) is True
    assert section.warp.status == AlignmentStatus.COMPLETE
    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring == section.alignment.current_anchoring


def test_wipe_alignment_for_flip_resets_everything():
    section = _section(
        alignment=Alignment(
            current_anchoring=[1.0] * 9,
            stored_anchoring=[1.0] * 9,
            status=AlignmentStatus.COMPLETE,
            source="manual",
        ),
        warp=WarpState(control_points=[ControlPoint(0, 0, 0, 0)], status=AlignmentStatus.COMPLETE),
    )
    wipe_alignment_for_flip(section)
    assert section.alignment.current_anchoring == [0.0] * 9
    assert section.alignment.stored_anchoring is None
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.warp.control_points == []
    assert section.warp.status == AlignmentStatus.NOT_STARTED


# ---------------------------------------------------------------------------
# commit_prep_draft
# ---------------------------------------------------------------------------


def test_commit_prep_draft_writes_mask_and_sets_path(tmp_path):
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    commit_prep_draft(section, mask)

    expected = slice_mask_path_for(section)
    assert expected.exists()
    assert section.preprocessing.slice_mask_path == str(expected)


def test_commit_prep_draft_none_mask_is_noop(tmp_path):
    """A flip-only save passes mask=None: nothing is written, path untouched."""
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
    )
    commit_prep_draft(section, None)
    assert section.preprocessing.slice_mask_path is None


def test_commit_prep_draft_preserves_alignment_through_flip(tmp_path):
    """Flips are invalidated at toggle time, so persisting a prep draft must
    never wipe an alignment the user (re)did after flipping."""
    section = _section(
        original_path=str(tmp_path / "img.png"),
        thumbnail_path=str(tmp_path / "thumbnails" / "img.tif"),
        alignment=Alignment(
            current_anchoring=[1.0] * 9, stored_anchoring=[1.0] * 9, status=AlignmentStatus.COMPLETE
        ),
    )
    section.preprocessing.flip_horizontal = True  # current differs from base (False)

    commit_prep_draft(section, None)

    assert section.alignment.status == AlignmentStatus.COMPLETE
    assert section.alignment.stored_anchoring == [1.0] * 9
