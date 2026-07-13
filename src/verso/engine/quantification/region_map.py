"""Shared backbone: full-resolution region labels + reconciled scope mask.

Every quantification reduces to, per section, a full-resolution on-disk map of
atlas region IDs plus a boolean scope mask. This module is the single place the
coordinate frames are reconciled (see ``.claude/quantification.md`` §3): raw
pixels, region labels, and masks all live in the **on-disk (un-flipped)** frame,
so the only reconciliation is a nearest-neighbour resolution rescale of the
(working-resolution) masks — no flips.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume
    from verso.engine.model.project import Section
    from verso.engine.registration import VersoRegistration


def upsample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to ``shape`` ``(H, W)``.

    Masks are integer maps, so nearest-neighbour is mandatory (no interpolation
    across region/foreground boundaries).
    """
    from PIL import Image

    arr = np.asarray(mask, dtype=bool)
    target_h, target_w = int(shape[0]), int(shape[1])
    if arr.shape == (target_h, target_w):
        return arr
    im = Image.fromarray(arr.view(np.uint8) * 255, mode="L")
    im = im.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def full_res_labels(reg: VersoRegistration, atlas: AtlasVolume, section: Section) -> np.ndarray:
    """Full-resolution ``(H, W)`` atlas region-ID map for ``section``.

    Delegates to :meth:`VersoRegistration.image_to_atlas` (which already applies
    anchoring, Delaunay warp, and preprocessing flips, and streams in row-tiles).
    The pre-built ``atlas`` is injected into ``reg`` so it isn't reconstructed.
    """
    reg._atlas_volume = atlas  # reuse the already-loaded atlas (avoids re-download)
    labels = reg.image_to_atlas(section.id, kind="annotation", space="full")
    return np.asarray(labels, dtype=np.int32)


def full_res_hemispheres(
    reg: VersoRegistration, atlas: AtlasVolume, section: Section
) -> np.ndarray:
    """Full-resolution ``(H, W)`` per-pixel hemisphere map for ``section``.

    Samples brainglobe's ``hemispheres`` volume through the identical warp used by
    :func:`full_res_labels` (via ``image_to_atlas(kind="hemisphere")``), so the
    returned map is pixel-matched 1:1 with the region-label map. Values are the
    atlas hemisphere codes (``1``/``2`` in-brain, ``0`` out-of-atlas).
    """
    reg._atlas_volume = atlas  # reuse the already-loaded atlas (avoids re-download)
    hemi = reg.image_to_atlas(section.id, kind="hemisphere", space="full")
    return np.asarray(hemi, dtype=np.uint8)


def slice_scope(section: Section, shape: tuple[int, int]) -> np.ndarray:
    """Boolean scope mask for a section at full-resolution ``shape`` ``(H, W)``.

    The Prep **slice mask** is the only silent filter (see plan §2). If the
    section has a saved slice mask it is loaded and nearest-upsampled to full
    resolution; otherwise the scope is the whole frame (all True) — the caller's
    precondition gate decides whether an unmasked section is allowed at all.
    """
    h, w = int(shape[0]), int(shape[1])
    mask_path = section.preprocessing.slice_mask_path
    if mask_path and Path(mask_path).exists():
        from verso.engine.preprocessing import load_mask

        return load_mask(mask_path, (h, w))
    return np.ones((h, w), dtype=bool)


def region_map(
    reg: VersoRegistration,
    atlas: AtlasVolume,
    section: Section,
    *,
    split_hemispheres: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(labels, scope, hemi)`` for a section, all full-resolution ``(H, W)``.

    ``labels`` is the int32 atlas region-ID map (0 = out-of-brain/out-of-atlas,
    kept as a real bucket). ``scope`` is the slice-mask scope (whole frame when the
    section has no slice mask). Callers intersect ``scope`` with an area mask as
    needed.

    ``hemi`` is the per-pixel hemisphere map (uint8, ``1``/``2`` in-brain, ``0``
    out-of-atlas) when ``split_hemispheres`` is True, else ``None``. It is sampled
    through the same warp as ``labels`` so the two are pixel-matched.
    """
    labels = full_res_labels(reg, atlas, section)
    scope = slice_scope(section, labels.shape)
    hemi = full_res_hemispheres(reg, atlas, section) if split_hemispheres else None
    return labels, scope, hemi
