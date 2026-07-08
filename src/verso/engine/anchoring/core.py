"""Anchoring vector algebra — decomposition, transforms, and the sample grid.

Pure vector math on the 9-element anchoring vector, with no dependency on the
data model or any atlas library. See :mod:`verso.engine.anchoring` for the
vector-format spec.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Anchoring vector decomposition
# ---------------------------------------------------------------------------


def is_anchored(anchoring: Sequence[float] | None) -> bool:
    """Return True when an anchoring vector is present and not all zeros.

    A missing or all-zero anchoring means the section has no usable plane yet.
    This is the single definition of "anchored" for the whole codebase, applied
    to live and stored anchorings alike. Code holding an
    :class:`~verso.engine.model.alignment.Alignment` should prefer its
    ``is_anchored`` property, which delegates here; call this directly only when
    all you have is a raw vector (the pure series algorithm, JSON parsing).
    """
    return bool(anchoring) and any(v != 0.0 for v in anchoring)


def anchoring_to_vectors(
    anchoring: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 9-element anchoring list into origin and direction vectors.

    Args:
        anchoring: [ox, oy, oz, ux, uy, uz, vx, vy, vz]

    Returns:
        Tuple ``(o, u, v)`` each as shape-(3,) float64 arrays.
    """
    a = np.asarray(anchoring, dtype=np.float64)
    if a.shape != (9,):
        raise ValueError(f"anchoring must have 9 elements, got {len(anchoring)}")
    return a[0:3], a[3:6], a[6:9]


def vectors_to_anchoring(o: np.ndarray, u: np.ndarray, v: np.ndarray) -> list[float]:
    """Pack origin and direction vectors back into a 9-element anchoring list."""
    return np.concatenate([o, u, v]).tolist()


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def normalized_to_atlas(s: float, t: float, anchoring: list[float]) -> np.ndarray:
    """Map normalized section coords (s, t) → atlas voxel coords (x, y, z).

    Args:
        s: Normalized x-coordinate along section width, in [0, 1].
        t: Normalized y-coordinate along section height, in [0, 1].
        anchoring: 9-element anchoring vector.

    Returns:
        Shape-(3,) array of atlas voxel coordinates.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    return o + s * u + t * v


def atlas_to_normalized(
    xyz: np.ndarray | list[float], anchoring: list[float]
) -> tuple[float, float]:
    """Map atlas voxel coords → normalized section coords (s, t).

    Solves the least-squares system  xyz − o = s·u + t·v.

    Args:
        xyz: Atlas voxel position, shape (3,).
        anchoring: 9-element anchoring vector.

    Returns:
        Tuple (s, t).  Values outside [0, 1] indicate the point is outside the
        section boundary.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    A = np.column_stack([u, v])  # (3, 2)
    b = np.asarray(xyz, dtype=np.float64) - o
    result, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1])


def anchoring_center(anchoring: list[float]) -> np.ndarray:
    """Return the center point of an anchoring plane in atlas voxel space."""
    o, u, v = anchoring_to_vectors(anchoring)
    return o + (u + v) / 2.0


# ---------------------------------------------------------------------------
# Pixel ↔ Normalised (convenience wrappers for the GUI)
# ---------------------------------------------------------------------------


def pixel_to_normalized(px: float, py: float, width: int, height: int) -> tuple[float, float]:
    """Convert working-resolution pixel coords to normalised section coords."""
    return px / width, py / height


def normalized_to_pixel(s: float, t: float, width: int, height: int) -> tuple[float, float]:
    """Convert normalised section coords to working-resolution pixel coords."""
    return s * width, t * height


# ---------------------------------------------------------------------------
# Atlas slice sampling grid
# ---------------------------------------------------------------------------


def make_atlas_sample_grid(
    anchoring: list[float],
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Build a (H, W, 3) array of atlas voxel coordinates for a 2D slice.

    The grid covers the full section plane at the given output resolution.
    Each cell ``grid[row, col]`` contains the atlas voxel (x, y, z) that
    corresponds to the normalized section coordinate
    ``(col / (W-1), row / (H-1))``.

    This grid is passed to the atlas volume sampler (in ``atlas.py``) to
    extract a 2D slice image.

    Args:
        anchoring: 9-element anchoring vector.
        out_width: Width of the desired output slice image in pixels.
        out_height: Height of the desired output slice image in pixels.

    Returns:
        Float64 array of shape (out_height, out_width, 3).
    """
    o, u, v = anchoring_to_vectors(anchoring)
    s = np.linspace(0.0, 1.0, out_width)
    t = np.linspace(0.0, 1.0, out_height)
    ss, tt = np.meshgrid(s, t)  # (H, W) each
    grid = o + ss[..., np.newaxis] * u + tt[..., np.newaxis] * v  # (H, W, 3)
    return grid
