"""Geometry helpers for annotation editing.

Kept in the engine (Qt-free) so the lasso-removal hit test is unit-testable and
reusable from scripts. Coordinates are plain (x, y) pairs in whatever space the
caller works in; :func:`points_in_polygon` is space-agnostic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def points_in_polygon(points: ArrayLike, polygon: ArrayLike) -> np.ndarray:
    """Return a boolean mask of which ``points`` fall inside ``polygon``.

    Uses the even-odd crossing-number rule, vectorised over all points. Points
    exactly on an edge are not guaranteed either way (typical for a lasso).

    Args:
        points: ``(N, 2)`` array of (x, y) coordinates to test.
        polygon: ``(M, 2)`` array of the polygon's vertices in order (open or
            closed; the closing edge is implied).

    Returns:
        A length-``N`` boolean array; ``True`` where the point is inside.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(pts) == 0 or len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)

    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        x_at_y = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= crosses & (x < x_at_y)
        j = i
    return inside
