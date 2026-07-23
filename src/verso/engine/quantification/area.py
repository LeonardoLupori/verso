"""Area-annotation intensity: same stats as intensity, restricted to an area.

The scope is ``slice_mask ∧ area_mask`` (plan §4.2). Only the scope differs from
the plain intensity analysis, so the accumulation reuses
:class:`~verso.engine.quantification.intensity.IntensityAccumulator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from verso.engine.quantification.region_map import upsample_mask

if TYPE_CHECKING:
    from verso.engine.model.annotation import AreaAnnotation
    from verso.engine.model.project import Section


def area_scope(
    section: Section,
    slice_scope: np.ndarray,
    area: AreaAnnotation,
) -> np.ndarray:
    """Return ``slice_scope ∧ area_mask`` at the slice-scope resolution.

    The area's per-section mask (working resolution, on-disk frame, keyed by image
    basename) is nearest-upsampled to the scope shape and intersected. Sections the
    area does not cover yield an all-False scope (nothing to quantify).

    Args:
        section: The section being quantified.
        slice_scope: ``(H, W)`` bool slice-mask scope (from ``region_map``).
        area: The selected area annotation.

    Returns:
        ``(H, W)`` bool intersected scope.
    """
    key = section.image_key.lower()
    mask = None
    for name, m in area.masks.items():
        if name.lower() == key:
            mask = m
            break
    if mask is None or not np.any(mask):
        return np.zeros(slice_scope.shape, dtype=bool)
    return slice_scope & upsample_mask(mask, slice_scope.shape)
