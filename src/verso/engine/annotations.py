"""Geometry helpers for annotation editing.

Kept in the engine (Qt-free) so the lasso-removal hit test is unit-testable and
reusable from scripts. Coordinates are plain (x, y) pairs in whatever space the
caller works in; :func:`points_in_polygon` is space-agnostic.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike

from verso.engine.model.annotation import Annotation, AreaAnnotation, PointSeries


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


def annotation_images(annotation: Annotation) -> set[str]:
    """Return the section image basenames a single annotation covers.

    Used to tell, per section, whether a given annotation appears on it — e.g.
    to flag the sections that carry the selected annotation in the filmstrip.

    Args:
        annotation: A point series or area annotation.

    Returns:
        The set of image basenames the annotation touches, lower-cased so callers
        can match a section's ``Path(original_path).name`` regardless of disk
        casing. For a point series, the images its points reference; for an area,
        the images with a non-empty mask (an all-``False`` mask does not count).
    """
    if isinstance(annotation, PointSeries):
        # A point series can hold hundreds of thousands of points but only ever
        # spans a few dozen images. Collapse to the distinct raw image strings
        # first (a cheap set insert per point), then run the costly basename +
        # lower-case on just those uniques — ~50x faster than transforming every
        # point (357 ms -> 7 ms for 200k points).
        return {os.path.basename(img).lower() for img in {p.image for p in annotation.points}}
    if isinstance(annotation, AreaAnnotation):
        return {name.lower() for name, mask in annotation.masks.items() if bool(np.any(mask))}
    return set()
