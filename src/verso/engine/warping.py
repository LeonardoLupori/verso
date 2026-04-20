"""Piecewise-affine Delaunay warp for atlas overlay refinement.

Algorithm (VisuAlign-compatible backward mapping)
-------------------------------------------------
Control points define pairs (src, dst) where:
  src_x/y : normalised [0,1] position in the *affine* atlas overlay
  dst_x/y : normalised [0,1] position in the section image

Warp steps:
  1. Add four invisible corner anchors (src == dst, identity) so the
     convex hull covers the entire image.
  2. Build a Delaunay triangulation on the DST (section) points.
  3. For every pixel in the output (atlas overlay resolution):
       a. Normalise its pixel coords to [0, 1] — these are "section space"
          fractions since the overlay is displayed stretched to section size.
       b. Find the enclosing Delaunay triangle in DST space.
       c. Compute barycentric coordinates inside that triangle.
       d. Interpolate the corresponding SRC (atlas) normalised coords.
       e. Convert to pixel coords and record in the remap array.
  4. Apply the remap with cv2.remap (bilinear) to the affine atlas overlay.

This matches VisuAlign's sample(x, y) approach: for each section pixel,
find its atlas location via barycentric interpolation, then sample the atlas.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import Delaunay

# Corner anchors in normalised [0, 1] space — always identity (src == dst)
_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _with_corners(pts: np.ndarray) -> np.ndarray:
    """Prepend four corner anchors to a (N, 2) normalised point array."""
    return np.vstack([_CORNERS, pts])


def find_atlas_position(
    s: float,
    t: float,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
) -> tuple[float, float]:
    """Given a section position (s, t) in [0,1], return the atlas (u, v) [0,1].

    Uses the current triangulation (with corner anchors) to map from section
    space into atlas overlay space.  If (s, t) is outside the convex hull the
    nearest corner is returned.

    Args:
        s: Normalised section x in [0, 1].
        t: Normalised section y in [0, 1].
        src_norm: (N, 2) existing control-point atlas positions (without corners).
        dst_norm: (N, 2) existing control-point section positions (without corners).

    Returns:
        (u, v) atlas normalised position clipped to [0, 1].
    """
    src_all = _with_corners(src_norm) if len(src_norm) else _CORNERS
    dst_all = _with_corners(dst_norm) if len(dst_norm) else _CORNERS

    tri = Delaunay(dst_all)
    pt = np.array([[s, t]])
    si = tri.find_simplex(pt)

    if si[0] < 0:
        return float(np.clip(s, 0.0, 1.0)), float(np.clip(t, 0.0, 1.0))

    T = tri.transform[si[0], :2]           # (2, 2)
    r = pt[0] - tri.transform[si[0], 2]    # (2,)
    b2 = T @ r                             # first two barycentric coords
    bary = np.array([b2[0], b2[1], 1.0 - b2[0] - b2[1]])

    idx = tri.simplices[si[0]]             # (3,) vertex indices
    u = float(np.clip((bary * src_all[idx, 0]).sum(), 0.0, 1.0))
    v = float(np.clip((bary * src_all[idx, 1]).sum(), 0.0, 1.0))
    return u, v


def warp_overlay(
    overlay: np.ndarray,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
) -> np.ndarray:
    """Warp the affine atlas overlay using piecewise-affine Delaunay interpolation.

    Args:
        overlay: (H, W) or (H, W, C) uint8 array — the affine atlas overlay.
        src_norm: (N, 2) float64 — control-point positions in atlas overlay
            normalised space [0, 1].  Do *not* include corner anchors.
        dst_norm: (N, 2) float64 — control-point positions in section image
            normalised space [0, 1].  Do *not* include corner anchors.

    Returns:
        Warped overlay, same shape and dtype as ``overlay``.
    """
    h, w = overlay.shape[:2]
    src_norm = np.asarray(src_norm, dtype=np.float64).reshape(-1, 2)
    dst_norm = np.asarray(dst_norm, dtype=np.float64).reshape(-1, 2)
    if len(src_norm) == 0 or np.allclose(src_norm, dst_norm):
        return overlay.copy()

    src_all = _with_corners(src_norm) if len(src_norm) else _CORNERS.copy()
    dst_all = _with_corners(dst_norm) if len(dst_norm) else _CORNERS.copy()

    # Triangulate in DST (section) normalised space
    tri = Delaunay(dst_all)

    # Normalised coords for each output pixel (overlay resolution)
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W) each
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (H*W, 2)

    simplices = tri.find_simplex(pixels)
    valid = simplices >= 0

    # Vectorised barycentric computation via scipy's precomputed transform:
    #   transform[s, :2] = 2×2 inverse of the triangle's edge matrix
    #   transform[s, 2]  = triangle origin vertex
    T = tri.transform[simplices[valid], :2]                      # (M, 2, 2)
    r = pixels[valid] - tri.transform[simplices[valid], 2]       # (M, 2)
    b = np.einsum("ijk,ik->ij", T, r)                            # (M, 2)
    bary = np.column_stack([b, 1.0 - b.sum(axis=1)])             # (M, 3)

    idx = tri.simplices[simplices[valid]]                         # (M, 3)
    atlas_x = (bary * src_all[idx, 0]).sum(axis=1)               # normalised [0,1]
    atlas_y = (bary * src_all[idx, 1]).sum(axis=1)

    # Build backward remap: output pixel → atlas pixel (in overlay pixel coords)
    # Default: identity (no warp outside convex hull)
    map_x = grid_x.astype(np.float32) * w
    map_y = grid_y.astype(np.float32) * h

    flat = np.where(valid)[0]
    rows, cols = flat // w, flat % w
    map_x[rows, cols] = (atlas_x * w).astype(np.float32)
    map_y[rows, cols] = (atlas_y * h).astype(np.float32)

    return cv2.remap(
        np.ascontiguousarray(overlay),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
