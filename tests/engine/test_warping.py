"""Tests for engine/warping.py."""

import cv2
import numpy as np
from scipy.spatial import Delaunay

from verso.engine.warping import (
    _CORNERS,
    build_backward_remap,
    find_atlas_position,
    warp_overlay,
    warp_points_atlas_to_section,
)


def _visualign_atlas_norm(s, t, src_norm, dst_norm, width, height):
    """Reference VisuAlign warp: section (s, t) → atlas (u, v), both normalised.

    Faithful port of ``data/Slice.java`` + ``nonlin/Triangle.java``: the
    triangulation is built in raw section **pixel** space on the marker's
    ``(nx, ny)`` (section) components with identity corner anchors 10% outside
    the frame, and ``transform`` barycentrically interpolates the ``(ox, oy)``
    (atlas) components. VERSO must reproduce this exactly inside the frame.
    """
    corners = np.array([[-0.1, -0.1], [1.1, -0.1], [-0.1, 1.1], [1.1, 1.1]])
    dst_px = np.vstack([corners, dst_norm]) * [width, height]  # (nx, ny)
    src_px = np.vstack([corners, src_norm]) * [width, height]  # (ox, oy)
    tri = Delaunay(dst_px)
    q = np.array([[s * width, t * height]])
    si = int(tri.find_simplex(q)[0])
    if si < 0:
        return float(np.clip(s, 0, 1)), float(np.clip(t, 0, 1))
    T = tri.transform[si, :2]
    r = q[0] - tri.transform[si, 2]
    b = T @ r
    bary = np.array([b[0], b[1], 1.0 - b[0] - b[1]])
    idx = tri.simplices[si]
    u = float(np.clip((bary * src_px[idx, 0]).sum() / width, 0.0, 1.0))
    v = float(np.clip((bary * src_px[idx, 1]).sum() / height, 0.0, 1.0))
    return u, v


def test_find_atlas_position_matches_visualign_when_aspect_set():
    """With aspect = width/height the warp must equal VisuAlign pixel-for-pixel.

    VisuAlign triangulates in section pixel space; VERSO normalises to [0,1]².
    Delaunay is not invariant under the anisotropic x/W, y/H scaling, so the two
    only agree when VERSO triangulates in the section's true aspect ratio.
    """
    width, height = 1140, 800  # aspect 1.425 — the case that exposed the bug
    aspect = width / height
    rng = np.random.default_rng(3)
    src = rng.uniform(0.15, 0.85, (10, 2))
    dst = src + rng.normal(0, 0.04, src.shape)
    queries = rng.uniform(0.08, 0.92, (60, 2))

    max_fixed = 0.0
    max_unfixed = 0.0
    for s, t in queries:
        ref_u, ref_v = _visualign_atlas_norm(s, t, src, dst, width, height)
        # Fixed path: aspect-correct triangulation must match VisuAlign exactly.
        fu, fv = find_atlas_position(float(s), float(t), src, dst, aspect=aspect)
        max_fixed = max(max_fixed, abs(fu - ref_u), abs(fv - ref_v))
        # Unfixed path (aspect=1, the old behaviour) diverges from VisuAlign.
        uu, uv = find_atlas_position(float(s), float(t), src, dst)
        max_unfixed = max(max_unfixed, abs(uu - ref_u), abs(uv - ref_v))

    assert max_fixed < 1e-9, f"aspect-correct warp must match VisuAlign (got {max_fixed})"
    # Sanity: the old normalised triangulation really did differ (regression guard).
    assert max_unfixed > 1e-3, "expected normalised-space warp to diverge from VisuAlign"


def test_build_backward_remap_matches_visualign_when_aspect_set():
    """The dense remap (display/export path) must also match VisuAlign per pixel."""
    width, height = 1280, 720  # aspect 1.778
    aspect = width / height
    h, w = 72, 128  # overlay grid at the same aspect
    rng = np.random.default_rng(7)
    src = rng.uniform(0.2, 0.8, (8, 2))
    dst = src + rng.normal(0, 0.05, src.shape)

    map_x, map_y = build_backward_remap(h, w, src, dst, aspect=aspect)

    # Spot-check pixel centres against the VisuAlign reference (atlas coords).
    max_err = 0.0
    for j in range(2, h, 11):
        for i in range(2, w, 11):
            s = (i + 0.5) / w
            t = (j + 0.5) / h
            ref_u, ref_v = _visualign_atlas_norm(s, t, src, dst, width, height)
            max_err = max(max_err, abs(map_x[j, i] / w - ref_u), abs(map_y[j, i] / h - ref_v))
    assert max_err < 1e-5, f"backward remap must match VisuAlign (got {max_err})"


def test_corner_anchors_match_visualign_convention():
    """Corner anchors must sit 10% outside the image, as VisuAlign does.

    VisuAlign's ``data/Slice.java`` seeds its triangulation with identity
    markers at (-0.1W, -0.1H), (1.1W, -0.1H), (-0.1W, 1.1H), (1.1W, 1.1H).
    verso uses the same anchors (in normalised space) so warped exports
    reproduce VisuAlign's deformation exactly inside the frame. Reverting to
    image-corner anchors (0,0)…(1,1) reintroduces the border mismatch.
    """
    expected = np.array([[-0.1, -0.1], [1.1, -0.1], [-0.1, 1.1], [1.1, 1.1]])
    np.testing.assert_allclose(_CORNERS, expected)


def test_warp_overlay_identity_preserves_image():
    """With src == dst the overlay must come back unchanged."""
    h, w = 60, 80
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    overlay[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))

    src = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    dst = src.copy()

    warped = warp_overlay(overlay, src, dst)

    assert warped.shape == overlay.shape
    assert warped.dtype == overlay.dtype
    np.testing.assert_array_equal(warped, overlay)


def test_warp_overlay_shift():
    """Shifting dst right of src should shift the sampled content left."""
    h, w = 60, 80
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, 35:45] = 255

    src = np.array([[0.125, 0.16], [0.875, 0.16], [0.125, 0.83], [0.875, 0.83]])
    dst = src + np.array([5.0 / w, 0.0])

    warped = warp_overlay(overlay, src, dst)

    assert warped.shape == overlay.shape
    assert warped.dtype == overlay.dtype


def test_warp_overlay_grayscale():
    overlay = np.full((40, 60), 128, dtype=np.uint8)
    src = np.array([[0.2, 0.25], [0.8, 0.25]])
    dst = src.copy()
    warped = warp_overlay(overlay, src, dst)
    assert warped.shape == (40, 60)


def test_warp_overlay_rgba_preserves_discrete_opacity_and_brightness():
    """Atlas RGBA overlays should not fade when remapped by control points."""
    h, w = 40, 60
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[8:32, 30] = [255, 255, 255, 220]

    src = np.array([[0.2, 0.25], [0.8, 0.25], [0.2, 0.75], [0.8, 0.75]])
    dst = src + np.array([0.5 / w, 0.0])

    warped = warp_overlay(overlay, src, dst)
    visible = warped[..., 3] > 0

    assert visible.any()
    assert set(np.unique(warped[..., 3])) <= {0, 220}
    np.testing.assert_array_equal(warped[visible, :3], np.full((visible.sum(), 3), 255))


def test_warp_overlay_no_control_points():
    """With no user control points the image is unchanged."""
    h, w = 40, 60
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    warped = warp_overlay(overlay, np.zeros((0, 2)), np.zeros((0, 2)))
    np.testing.assert_array_equal(warped, overlay)


def test_warp_points_atlas_to_section_identity():
    """With src == dst the forward warp is the identity."""
    src = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    dst = src.copy()
    pts = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    out = warp_points_atlas_to_section(pts, src, dst)
    np.testing.assert_allclose(out, pts, atol=1e-9)


def test_warp_points_atlas_to_section_translation():
    """A pure translation of the section anchors shifts atlas points by the same amount."""
    src = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
    dst = src + np.array([0.1, 0.0])
    pts = np.array([[0.5, 0.5], [0.3, 0.4]])
    out = warp_points_atlas_to_section(pts, src, dst)
    np.testing.assert_allclose(out, pts + np.array([0.1, 0.0]), atol=1e-9)


def test_warp_points_atlas_to_section_no_control_points():
    pts = np.array([[0.1, 0.2], [0.9, 0.4]])
    out = warp_points_atlas_to_section(pts, np.zeros((0, 2)), np.zeros((0, 2)))
    np.testing.assert_array_equal(out, pts)


def test_export_and_display_warp_agree():
    """build_backward_remap on labels must match warp_overlay on RGBA — export ≡ display.

    This encodes the key invariant broken by the original export bug: the export
    pipeline used a *forward* Delaunay map (warp_points_atlas_to_section) while
    the display used a *backward* remap (warp_overlay).  Both paths now use the
    same build_backward_remap + INTER_NEAREST, so they must agree pixel-for-pixel.
    """
    h, w = 100, 150
    # Four quadrant label regions.
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:50, :75] = 1
    labels[:50, 75:] = 2
    labels[50:, :75] = 3
    labels[50:, 75:] = 4

    # Build RGBA from labels the same way slice_annotation does.
    palette: dict[int, tuple[int, int, int, int]] = {
        0: (0, 0, 0, 0),
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (0, 0, 255, 255),
        4: (255, 255, 0, 255),
    }
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for lbl, color in palette.items():
        rgba[labels == lbl] = color

    # Control point: nudge atlas (0.25, 0.25) to section (0.40, 0.40).
    src = np.array([[0.25, 0.25]])
    dst = np.array([[0.40, 0.40]])

    # Export path: build_backward_remap + NEAREST directly on the label array.
    map_x, map_y = build_backward_remap(h, w, src, dst)
    labels_export = cv2.remap(
        labels.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)

    # Display path: warp_overlay on RGBA (also uses NEAREST for 4-channel).
    rgba_display = warp_overlay(rgba, src, dst)

    # Reconstruct integer labels from the warped RGBA.
    labels_display = np.zeros((h, w), dtype=np.int32)
    for lbl, color in palette.items():
        if lbl == 0:
            continue
        mask = np.all(rgba_display[:, :, :3] == np.array(color[:3]), axis=2)
        labels_display[mask] = lbl

    np.testing.assert_array_equal(labels_export, labels_display)


def test_warp_overlay_output_shape_preserved():
    """warp_overlay must return the same shape and dtype as the input."""
    for dtype in (np.uint8, np.float32):
        overlay = np.random.rand(50, 70, 3).astype(dtype)
        if dtype == np.uint8:
            overlay = (overlay * 255).astype(np.uint8)

        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dst = src.copy()
        warped = warp_overlay(overlay, src, dst)

        assert warped.shape == overlay.shape, f"shape mismatch for dtype={dtype}"
        assert warped.dtype == overlay.dtype, f"dtype mismatch for dtype={dtype}"
