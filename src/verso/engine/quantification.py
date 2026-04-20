"""Region quantification: map detected objects to atlas regions."""

from __future__ import annotations

import numpy as np


def quantify_points(
    points_px: np.ndarray,
    anchoring: list[float],
    atlas_labels: np.ndarray,
    section_shape: tuple[int, int],
) -> dict[int, int]:
    """Count how many *points_px* fall in each atlas region.

    Args:
        points_px: Float array of shape (N, 2) in section pixel space (row, col).
        anchoring: 9-element QuickNII anchoring vector for this section.
        atlas_labels: 3-D integer label volume (AP × DV × LR).
        section_shape: (height, width) of the working-resolution section image.

    Returns:
        Mapping of region label → point count.
    """
    raise NotImplementedError


def quantify_area(
    mask: np.ndarray,
    anchoring: list[float],
    atlas_labels: np.ndarray,
) -> dict[int, float]:
    """Compute the fraction of *mask* area attributable to each atlas region.

    Args:
        mask: Boolean H×W array; True = foreground pixels.
        anchoring: 9-element QuickNII anchoring vector for this section.
        atlas_labels: 3-D integer label volume (AP × DV × LR).

    Returns:
        Mapping of region label → pixel count.
    """
    raise NotImplementedError
