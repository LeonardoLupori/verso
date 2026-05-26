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
  4. Apply the remap with cv2.remap to the affine atlas overlay.  RGBA atlas
     overlays use nearest-neighbour sampling so outline/fill opacity stays
     constant instead of being averaged with transparent pixels.

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


def warp_points_atlas_to_section(
    points_norm: np.ndarray,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
) -> np.ndarray:
    """Map atlas-space points into section-space via the Delaunay warp.

    Mirror of the backward map in :func:`warp_overlay`: triangulates on the
    *src* (atlas) anchors and interpolates *dst* (section) coords for each
    input point. Points outside the convex hull pass through unchanged
    (clipped to ``[0, 1]``).

    Args:
        points_norm: (M, 2) normalised atlas-space points in ``[0, 1]``.
        src_norm: (N, 2) atlas-space control points (no corner anchors).
        dst_norm: (N, 2) section-space control points (no corner anchors).

    Returns:
        (M, 2) normalised section-space points.
    """
    pts = np.asarray(points_norm, dtype=np.float64).reshape(-1, 2)
    src_norm = np.asarray(src_norm, dtype=np.float64).reshape(-1, 2)
    dst_norm = np.asarray(dst_norm, dtype=np.float64).reshape(-1, 2)

    if len(src_norm) == 0 or np.allclose(src_norm, dst_norm):
        return np.clip(pts, 0.0, 1.0)

    src_all = _with_corners(src_norm)
    dst_all = _with_corners(dst_norm)

    tri = Delaunay(src_all)
    simplices = tri.find_simplex(pts)

    out = np.clip(pts.copy(), 0.0, 1.0)
    valid = simplices >= 0
    if not np.any(valid):
        return out

    T = tri.transform[simplices[valid], :2]
    r = pts[valid] - tri.transform[simplices[valid], 2]
    b = np.einsum("ijk,ik->ij", T, r)
    bary = np.column_stack([b, 1.0 - b.sum(axis=1)])

    idx = tri.simplices[simplices[valid]]
    out_x = (bary * dst_all[idx, 0]).sum(axis=1)
    out_y = (bary * dst_all[idx, 1]).sum(axis=1)
    out[valid, 0] = np.clip(out_x, 0.0, 1.0)
    out[valid, 1] = np.clip(out_y, 0.0, 1.0)
    return out


def build_backward_remap(
    h: int,
    w: int,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (map_x, map_y) float32 arrays for cv2.remap using the backward Delaunay warp.

    Triangulates on *dst* (section space) and interpolates *src* (atlas space),
    matching the warp direction used by :func:`warp_overlay`.  The returned
    arrays are ready to pass directly to ``cv2.remap`` with any interpolation
    mode.  Pixels outside the convex hull of *dst* default to identity (no warp).

    Args:
        h: Output image height in pixels.
        w: Output image width in pixels.
        src_norm: (N, 2) float64 — atlas-space control points (no corner anchors).
        dst_norm: (N, 2) float64 — section-space control points (no corner anchors).

    Returns:
        ``(map_x, map_y)`` each of shape ``(h, w)`` float32.
    """
    src_all = _with_corners(src_norm) if len(src_norm) else _CORNERS.copy()
    dst_all = _with_corners(dst_norm) if len(dst_norm) else _CORNERS.copy()

    tri = Delaunay(dst_all)

    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W) each
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (H*W, 2)

    simplices = tri.find_simplex(pixels)
    valid = simplices >= 0

    T = tri.transform[simplices[valid], :2]                      # (M, 2, 2)
    r = pixels[valid] - tri.transform[simplices[valid], 2]       # (M, 2)
    b = np.einsum("ijk,ik->ij", T, r)                            # (M, 2)
    bary = np.column_stack([b, 1.0 - b.sum(axis=1)])             # (M, 3)

    idx = tri.simplices[simplices[valid]]                         # (M, 3)
    atlas_x = (bary * src_all[idx, 0]).sum(axis=1)               # normalised [0,1]
    atlas_y = (bary * src_all[idx, 1]).sum(axis=1)

    map_x = grid_x.astype(np.float32) * w
    map_y = grid_y.astype(np.float32) * h

    flat = np.where(valid)[0]
    rows, cols = flat // w, flat % w
    map_x[rows, cols] = (atlas_x * w).astype(np.float32)
    map_y[rows, cols] = (atlas_y * h).astype(np.float32)

    return map_x, map_y


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

    map_x, map_y = build_backward_remap(h, w, src_norm, dst_norm)

    interpolation = (
        cv2.INTER_NEAREST
        if overlay.ndim == 3 and overlay.shape[2] == 4
        else cv2.INTER_LINEAR
    )

    return cv2.remap(
        np.ascontiguousarray(overlay),
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
