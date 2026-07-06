"""Unsaved-edit (draft) data and the operations that persist them.

This module holds the pure-engine side of the persistent unsaved-edits model:

- :class:`PrepDraft` — an in-memory slice-mask edit for one section, kept
  resident until the user saves.
- :func:`persist_prep_draft` — write a prep draft's masks to disk.
- :func:`commit_alignment` / :func:`commit_warp` — promote in-memory align/warp
  edits to their saved state.
- :func:`wipe_alignment_for_flip` — invalidate alignment + warp after a flip.
- mask path helpers shared by the GUI and the save path.

None of this imports Qt, so it stays usable from scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Section
from verso.engine.preprocessing import save_mask


@dataclass
class PrepDraft:
    """A resident, unsaved prep edit for one section (slice mask only).

    The mask is stored unflipped (matching the on-disk convention); flips live on
    ``section.preprocessing`` and need no draft.  ``slice_mask`` is ``None`` when
    unset; a ``None`` mask with ``mask_dirty`` True means the user cleared the
    mask (the saved file, if any, should be removed).

    ``base_flip_*`` capture the section's last-saved flip flags at the moment the
    draft was created so the GUI can carry them across navigation for its
    Clear/Reset logic.  They no longer drive alignment invalidation: a flip wipes
    the alignment the instant it is toggled, not when the draft is persisted.
    """

    slice_mask: np.ndarray | None = None
    mask_dirty: bool = False
    base_flip_h: bool = False
    base_flip_v: bool = False


def slice_mask_path_for(section: Section) -> Path:
    """Canonical on-disk path for a section's slice mask PNG."""
    masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
    return masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"


def wipe_alignment_for_flip(section: Section) -> None:
    """Reset a section's alignment + warp because its flip changed.

    A horizontal/vertical flip changes the image coordinate frame, so any
    existing registration no longer applies.
    """
    section.alignment.anchoring = [0.0] * 9
    section.alignment.position_mm = None
    section.alignment.status = AlignmentStatus.NOT_STARTED
    section.alignment.source = None
    section.alignment.stored_anchoring = None
    section.alignment.proposal_anchoring = None
    section.alignment.proposal_confidence = None
    section.alignment.proposal_run_id = None
    section.warp.control_points.clear()
    section.warp.status = AlignmentStatus.NOT_STARTED


def persist_prep_draft(section: Section, draft: PrepDraft) -> None:
    """Write *draft*'s masks to disk and update ``section.preprocessing`` paths.

    A flip invalidates the alignment **at the moment the user toggles it** (the
    GUI wipes the alignment + warp then), not here — so persisting a prep draft
    only writes masks and never touches the alignment.  This keeps an alignment
    the user (re)did *after* a flip from being clobbered when the flip is later
    saved.  ``draft.base_flip_*`` are retained only so the GUI can carry the
    last-saved flip across navigation for its Clear/Reset logic.
    """
    if draft.mask_dirty and draft.slice_mask is not None:
        path = slice_mask_path_for(section)
        save_mask(draft.slice_mask, path)
        section.preprocessing.slice_mask_path = str(path)


def commit_alignment(section: Section) -> bool:
    """Promote the live anchoring to the saved plane (status COMPLETE).

    No-op (returns False) when the section has no usable anchoring yet.
    """
    anchoring = section.alignment.anchoring
    if not anchoring or all(v == 0.0 for v in anchoring):
        return False
    section.alignment.stored_anchoring = list(anchoring)
    section.alignment.status = AlignmentStatus.COMPLETE
    return True


def commit_warp(section: Section) -> bool:
    """Mark the section's warp saved, committing its alignment plane too.

    Placing control points means the user accepted the section's affine plane,
    so the alignment is promoted to COMPLETE via :func:`commit_alignment` when it
    isn't already.  Without this the next save's auto-interpolation would treat
    the plane as unfinished, re-guess it, and leave the warp sitting on a
    different plane.  Returns False without changing state only when there is no
    usable plane to commit (a zero/empty anchoring).
    """
    if section.alignment.status != AlignmentStatus.COMPLETE and not commit_alignment(section):
        return False
    section.warp.status = AlignmentStatus.COMPLETE
    return True


__all__ = [
    "PrepDraft",
    "commit_alignment",
    "commit_warp",
    "persist_prep_draft",
    "slice_mask_path_for",
    "wipe_alignment_for_flip",
]
