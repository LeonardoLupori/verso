"""Unsaved-edit (draft) data and the operations that persist them.

This module holds the pure-engine side of the persistent unsaved-edits model:

- :func:`commit_prep_draft` — write an unsaved slice mask to disk.
- :func:`commit_alignment` / :func:`commit_warp` — promote in-memory align/warp
  edits to their saved state.
- :func:`reset_alignment` — reset a section's alignment + warp to default.
- mask path helpers shared by the GUI and the save path.

None of this imports Qt, so it stays usable from scripts and tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Section
from verso.engine.preprocessing import save_mask


def slice_mask_path_for(section: Section) -> Path:
    """Canonical on-disk path for a section's slice mask PNG."""
    masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
    return masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"


def reset_alignment(section: Section) -> None:
    """Reset a section's alignment + warp back to the un-registered default.

    Clears both the live and saved planes and the dependent warp. Used wherever
    a registration must be discarded: a flip changing the image coordinate frame
    (so the old plane no longer applies), the Align view's Reset, and reversing
    the proposal series before any alignment is stored.
    """
    section.alignment.current_anchoring = [0.0] * 9
    section.alignment.position_mm = None
    section.alignment.status = AlignmentStatus.NOT_STARTED
    section.alignment.source = None
    section.alignment.stored_anchoring = None
    section.warp.control_points.clear()
    section.warp.status = AlignmentStatus.NOT_STARTED


def commit_prep_draft(section: Section, mask: np.ndarray | None) -> None:
    """Write an unsaved slice *mask* to disk and update the preprocessing path.

    ``mask`` is the section's in-progress slice mask (``None`` when only flips
    changed, in which case there is nothing to write — flips live on
    ``section.preprocessing`` and are persisted with the project).

    A flip invalidates the alignment **at the moment the user toggles it** (the
    GUI wipes the alignment + warp then), not here — so committing a prep draft
    only writes the mask and never touches the alignment.  This keeps an
    alignment the user (re)did *after* a flip from being clobbered when the flip
    is later saved.
    """
    if mask is not None:
        path = slice_mask_path_for(section)
        save_mask(mask, path)
        section.preprocessing.slice_mask_path = str(path)


def commit_alignment(section: Section) -> bool:
    """Promote the live anchoring to the saved plane (status COMPLETE).

    No-op (returns False) when the section has no usable anchoring yet.
    """
    if not section.alignment.is_anchored:
        return False
    section.alignment.stored_anchoring = list(section.alignment.current_anchoring)
    section.alignment.status = AlignmentStatus.COMPLETE
    return True


def commit_warp(section: Section) -> bool:
    """Mark the section's warp saved, committing its alignment plane too.

    An **empty** warp (no control points) is not a finished warp, so it resets to
    NOT_STARTED (matching the per-view Warp save) and returns False.

    Placing control points means the user accepted the section's affine plane,
    so the alignment is promoted to COMPLETE via :func:`commit_alignment` when it
    isn't already.  Without this the next save's auto-interpolation would treat
    the plane as unfinished, re-guess it, and leave the warp sitting on a
    different plane.  Returns False without promoting to COMPLETE when there are
    no control points, or when there is no usable plane to commit (a zero/empty
    anchoring).
    """
    if not section.warp.control_points:
        section.warp.status = AlignmentStatus.NOT_STARTED
        return False
    if section.alignment.status != AlignmentStatus.COMPLETE and not commit_alignment(section):
        return False
    section.warp.status = AlignmentStatus.COMPLETE
    return True


__all__ = [
    "commit_alignment",
    "commit_prep_draft",
    "commit_warp",
    "reset_alignment",
    "slice_mask_path_for",
]
