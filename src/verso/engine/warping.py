"""Piecewise-affine Delaunay warp for atlas overlay refinement.

Algorithm (VisuAlign-compatible backward mapping)
-------------------------------------------------
Control points define pairs (src, dst) where:
  src_x/y : normalised [0,1] position in the *affine* atlas overlay
  dst_x/y : normalised [0,1] position in the section image

Warp steps:
  1. Add four invisible corner anchors (src == dst, identity) positioned 10%
     outside the image on every side, matching VisuAlign's triangulation
     (data/Slice.java). The convex hull then covers the whole frame with a
     margin, so every in-image pixel falls strictly inside a triangle and is
     interpolated rather than clamped at the border.
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

# Corner anchors in normalised section space — always identity (src == dst).
# Placed 10% outside the image on every side, mirroring VisuAlign's
# Slice.triangulate() (data/Slice.java), which seeds the triangulation with
# markers at (-0.1W, -0.1H), (1.1W, -0.1H), (-0.1W, 1.1H), (1.1W, 1.1H).
# Anchoring outside the frame (rather than at the image corners (0,0)…(1,1))
# keeps every in-image pixel strictly inside the convex hull, so border
# triangles are interpolated, not clamped — reproducing VisuAlign's warp
# exactly inside the frame.
_CORNERS = np.array([[-0.1, -0.1], [1.1, -0.1], [-0.1, 1.1], [1.1, 1.1]])


def _with_corners(pts: np.ndarray) -> np.ndarray:
    """Prepend four corner anchors to a (N, 2) normalised point array."""
    return np.vstack([_CORNERS, pts])


def _tri_scale(aspect: float) -> np.ndarray:
    """Anisotropy factor that puts normalised points into VisuAlign's space.

    VERSO stores control points in normalised ``[0, 1]²`` (x divided by the
    section *width*, y by its *height*).  VisuAlign builds its Delaunay
    triangulation in raw section **pixel** space (``width``×``height``).  A
    Delaunay triangulation is invariant under *similarity* transforms but **not**
    under the anisotropic ``(x/W, y/H)`` scaling VERSO's normalisation applies
    when ``W != H`` — so triangulating the normalised points directly yields a
    different triangle topology, and thus a different piecewise-affine warp, from
    VisuAlign's inside the frame.

    Scaling x by ``aspect = W / H`` (leaving y) restores the section's true
    pixel aspect ratio up to a uniform factor (``[W, H] = H · [aspect, 1]``),
    which Delaunay *is* invariant to.  The triangulation then matches VisuAlign
    exactly.  Barycentric interpolation is affine-invariant, so the interpolated
    src/dst coordinates are unchanged in value — only *which* triangle a point
    falls in changes.  ``aspect = 1.0`` is the identity (square section).
    """
    return np.array([float(aspect), 1.0])


def find_atlas_position(
    s: float,
    t: float,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
    aspect: float = 1.0,
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
        aspect: Section ``width / height``.  Triangulation is performed in this
            aspect-corrected space for VisuAlign parity (see :func:`_tri_scale`).

    Returns:
        (u, v) atlas normalised position clipped to [0, 1].
    """
    src_all = _with_corners(src_norm) if len(src_norm) else _CORNERS
    dst_all = _with_corners(dst_norm) if len(dst_norm) else _CORNERS

    scale = _tri_scale(aspect)
    tri = Delaunay(dst_all * scale)
    pt = np.array([[s, t]]) * scale
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
    aspect: float = 1.0,
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
        aspect: Section ``width / height``.  Triangulation is performed in this
            aspect-corrected space for VisuAlign parity (see :func:`_tri_scale`).

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

    scale = _tri_scale(aspect)
    tri = Delaunay(src_all * scale)
    simplices = tri.find_simplex(pts * scale)

    out = np.clip(pts.copy(), 0.0, 1.0)
    valid = simplices >= 0
    if not np.any(valid):
        return out

    T = tri.transform[simplices[valid], :2]
    r = pts[valid] * scale - tri.transform[simplices[valid], 2]
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
    aspect: float = 1.0,
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
        aspect: Section ``width / height``.  Triangulation is performed in this
            aspect-corrected space for VisuAlign parity (see :func:`_tri_scale`).

    Returns:
        ``(map_x, map_y)`` each of shape ``(h, w)`` float32.
    """
    src_all = _with_corners(src_norm) if len(src_norm) else _CORNERS.copy()
    dst_all = _with_corners(dst_norm) if len(dst_norm) else _CORNERS.copy()

    # Triangulate in the section's true aspect ratio so the topology matches
    # VisuAlign (see :func:`_tri_scale`).  The dst anchors and the per-pixel
    # query points are scaled by the same factor; *src* stays in normalised atlas
    # space because it is the interpolation output, not part of the geometry.
    scale = _tri_scale(aspect)
    tri = Delaunay(dst_all * scale)

    # Collapse each dst triangle to a single affine pixel→atlas map.  The warp is
    # piecewise affine, so inside a triangle the atlas position is an affine
    # function of the (scaled) pixel: ``atlas = coef · p_scaled + bias``.
    # Precomputing one such affine per triangle (reusing qhull's barycentric
    # ``tri.transform``) lets the per-pixel step run as a few flat multiplies
    # instead of building (M,2,2) and (M,3) barycentric temporaries — the latter
    # dominated the cost during live warp drags.  Result is bit-identical to the
    # barycentric formulation.
    bary_T = tri.transform[:, :2, :]            # (T, 2, 2) barycentric matrix
    offset = tri.transform[:, 2, :]             # (T, 2) triangle origin (scaled space)
    verts = tri.simplices                       # (T, 3) vertex indices
    s2 = src_all[verts[:, 2]]
    edges = np.stack(                           # (T, 2, 2) src edge vectors
        [src_all[verts[:, 0]] - s2, src_all[verts[:, 1]] - s2], axis=2
    )
    coef = edges @ bary_T                       # (T, 2, 2) maps scaled-pixel → atlas
    bias = s2 - np.einsum("tij,tj->ti", coef, offset)  # (T, 2)
    # Contiguous 1-D component arrays so the per-pixel gather stays cheap.
    a00 = np.ascontiguousarray(coef[:, 0, 0])
    a01 = np.ascontiguousarray(coef[:, 0, 1])
    a10 = np.ascontiguousarray(coef[:, 1, 0])
    a11 = np.ascontiguousarray(coef[:, 1, 1])
    b0 = np.ascontiguousarray(bias[:, 0])
    b1 = np.ascontiguousarray(bias[:, 1])

    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W) each
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (H*W, 2) normalised
    pixels_scaled = pixels * scale                              # triangulation space

    simplices = tri.find_simplex(pixels_scaled)
    valid = simplices >= 0

    # Identity (no warp) outside the convex hull; warped pixels overwritten below.
    map_x = (pixels[:, 0] * w).astype(np.float32)
    map_y = (pixels[:, 1] * h).astype(np.float32)

    s = simplices[valid]
    px = pixels_scaled[valid, 0]
    py = pixels_scaled[valid, 1]
    atlas_x = a00[s] * px + a01[s] * py + b0[s]   # normalised [0, 1]
    atlas_y = a10[s] * px + a11[s] * py + b1[s]
    map_x[valid] = (atlas_x * w).astype(np.float32)
    map_y[valid] = (atlas_y * h).astype(np.float32)

    return map_x.reshape(h, w), map_y.reshape(h, w)


def warp_overlay(
    overlay: np.ndarray,
    src_norm: np.ndarray,
    dst_norm: np.ndarray,
    aspect: float = 1.0,
) -> np.ndarray:
    """Warp the affine atlas overlay using piecewise-affine Delaunay interpolation.

    Args:
        overlay: (H, W) or (H, W, C) uint8 array — the affine atlas overlay.
        src_norm: (N, 2) float64 — control-point positions in atlas overlay
            normalised space [0, 1].  Do *not* include corner anchors.
        dst_norm: (N, 2) float64 — control-point positions in section image
            normalised space [0, 1].  Do *not* include corner anchors.
        aspect: Section ``width / height``.  Triangulation is performed in this
            aspect-corrected space for VisuAlign parity (see :func:`_tri_scale`).
            Pass the section's working ``width / height``; the default ``1.0``
            assumes a square section.

    Returns:
        Warped overlay, same shape and dtype as ``overlay``.
    """
    h, w = overlay.shape[:2]
    src_norm = np.asarray(src_norm, dtype=np.float64).reshape(-1, 2)
    dst_norm = np.asarray(dst_norm, dtype=np.float64).reshape(-1, 2)
    if len(src_norm) == 0 or np.allclose(src_norm, dst_norm):
        return overlay.copy()

    map_x, map_y = build_backward_remap(h, w, src_norm, dst_norm, aspect=aspect)

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
