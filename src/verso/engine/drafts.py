"""Unsaved-edit (draft) data and the operations that persist them.

This module holds the pure-engine side of the persistent unsaved-edits model:

- :class:`PrepDraft` — an in-memory slice/L-R mask edit for one section, kept
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
from verso.engine.preprocessing import save_lr_mask, save_mask


@dataclass
class PrepDraft:
    """A resident, unsaved prep edit for one section (masks only).

    Masks are stored unflipped (matching the on-disk convention); flips live on
    ``section.preprocessing`` and need no draft.  ``slice_mask``/``lr_mask`` are
    ``None`` when unset; a ``None`` mask with its ``*_dirty`` flag True means the
    user cleared that mask (the saved file, if any, should be removed).

    ``base_flip_*`` capture the section's last-saved flip flags at the moment the
    draft was created, so :func:`persist_prep_draft` can detect a flip change and
    invalidate the alignment exactly like the interactive save path.
    """

    slice_mask: np.ndarray | None = None
    lr_mask: np.ndarray | None = None
    mask_dirty: bool = False
    lr_dirty: bool = False
    base_flip_h: bool = False
    base_flip_v: bool = False


def slice_mask_path_for(section: Section) -> Path:
    """Canonical on-disk path for a section's slice mask PNG."""
    masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
    return masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"


def lr_mask_path_for(section: Section) -> Path:
    """Canonical on-disk path for a section's L/R hemisphere mask PNG."""
    lr_dir = Path(section.thumbnail_path).parent.parent / "lr_masks"
    return lr_dir / f"{Path(section.original_path).stem}_lr.png"


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


def persist_prep_draft(section: Section, draft: PrepDraft) -> bool:
    """Write *draft*'s masks to disk and update ``section.preprocessing`` paths.

    Returns:
        True iff the section's flip changed relative to ``draft.base_flip_*`` —
        in which case the caller must treat the alignment + warp as invalidated
        (this function already wiped them).
    """
    if draft.mask_dirty and draft.slice_mask is not None:
        path = slice_mask_path_for(section)
        save_mask(draft.slice_mask, path)
        section.preprocessing.slice_mask_path = str(path)

    if draft.lr_dirty:
        if draft.lr_mask is None:
            old = section.preprocessing.lr_mask_path
            if old:
                try:
                    Path(old).unlink(missing_ok=True)
                except OSError:
                    pass
            section.preprocessing.lr_mask_path = None
            section.preprocessing.lr_line = None
        else:
            path = lr_mask_path_for(section)
            save_lr_mask(draft.lr_mask, path)
            section.preprocessing.lr_mask_path = str(path)

    flip_changed = (
        draft.base_flip_h != section.preprocessing.flip_horizontal
        or draft.base_flip_v != section.preprocessing.flip_vertical
    )
    if flip_changed:
        wipe_alignment_for_flip(section)
    return flip_changed


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
    """Mark the section's warp saved.  Requires a COMPLETE alignment."""
    if section.alignment.status != AlignmentStatus.COMPLETE:
        return False
    section.warp.status = AlignmentStatus.COMPLETE
    return True


__all__ = [
    "PrepDraft",
    "slice_mask_path_for",
    "lr_mask_path_for",
    "wipe_alignment_for_flip",
    "persist_prep_draft",
    "commit_alignment",
    "commit_warp",
]
