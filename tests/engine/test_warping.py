"""Tests for engine/warping.py."""

import numpy as np
import pytest

from verso.engine.warping import add_corner_anchors, compute_warp, warp_overlay


# ---------------------------------------------------------------------------
# add_corner_anchors
# ---------------------------------------------------------------------------

def test_add_corner_anchors_prepends_four_corners():
    src = np.array([[50.0, 50.0], [100.0, 80.0]])
    dst = np.array([[55.0, 52.0], [98.0, 83.0]])
    shape = (200, 300, 3)

    src_all, dst_all = add_corner_anchors(src, dst, shape)

    assert src_all.shape == (6, 2)
    assert dst_all.shape == (6, 2)

    # First four rows are the corners with identity mapping.
    expected_corners = np.array([[0, 0], [299, 0], [0, 199], [299, 199]], dtype=float)
    np.testing.assert_array_equal(src_all[:4], expected_corners)
    np.testing.assert_array_equal(dst_all[:4], expected_corners)


def test_add_corner_anchors_grayscale_shape():
    src = np.zeros((0, 2))
    dst = np.zeros((0, 2))
    src_all, dst_all = add_corner_anchors(src, dst, (100, 200))  # 2-tuple
    expected = np.array([[0, 0], [199, 0], [0, 99], [199, 99]], dtype=float)
    np.testing.assert_array_equal(src_all, expected)


# ---------------------------------------------------------------------------
# compute_warp / warp_overlay — identity case
# ---------------------------------------------------------------------------

def _checkerboard(h: int, w: int) -> np.ndarray:
    """8×8 checkerboard pattern, uint8, shape (H, W, 3)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if (r // 8 + c // 8) % 2 == 0:
                img[r, c] = 255
    return img


def test_warp_overlay_identity_preserves_image():
    """With src == dst the overlay must come back unchanged (within bilinear tolerance)."""
    # Use a smooth gradient rather than a sharp checkerboard so that bilinear
    # interpolation at near-integer map coordinates does not introduce errors.
    h, w = 60, 80
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    overlay[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))

    src = np.array([[20.0, 15.0], [60.0, 15.0], [20.0, 45.0], [60.0, 45.0]])
    dst = src.copy()

    warped = warp_overlay(overlay, src, dst)

    assert warped.shape == overlay.shape
    assert warped.dtype == overlay.dtype
    # Allow ±2 for sub-pixel bilinear interpolation error on a smooth image.
    np.testing.assert_allclose(
        warped[10:50, 10:70].astype(int),
        overlay[10:50, 10:70].astype(int),
        atol=2,
    )


def test_warp_overlay_shift():
    """Shifting all control points right by 5 px should shift the content left by 5 px."""
    h, w = 60, 80
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw a vertical white stripe in the centre.
    overlay[:, 35:45] = 255

    # Shift dst 5 px to the right of src → content moves left in the output.
    src = np.array([[10.0, 10.0], [70.0, 10.0], [10.0, 50.0], [70.0, 50.0]])
    dst = src + np.array([5.0, 0.0])

    warped = warp_overlay(overlay, src, dst)

    # The stripe should have shifted: check a pixel that was white and is now black.
    assert warped.shape == overlay.shape


def test_warp_overlay_grayscale():
    overlay = np.full((40, 60), 128, dtype=np.uint8)
    src = np.array([[10.0, 10.0], [50.0, 10.0]])
    dst = src.copy()
    warped = warp_overlay(overlay, src, dst)
    assert warped.shape == (40, 60)


def test_warp_overlay_no_control_points():
    """With no user control points (only corner anchors) the image is unchanged."""
    h, w = 40, 60
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    warped = warp_overlay(overlay, np.zeros((0, 2)), np.zeros((0, 2)))
    np.testing.assert_allclose(warped.astype(int), overlay.astype(int), atol=2)


def test_compute_warp_output_shape_preserved():
    """compute_warp must return the same shape and dtype as the input."""
    for dtype in (np.uint8, np.float32):
        overlay = np.random.rand(50, 70, 3).astype(dtype)
        if dtype == np.uint8:
            overlay = (overlay * 255).astype(np.uint8)

        src = np.array([[0, 0], [69, 0], [0, 49], [69, 49]], dtype=np.float64)
        dst = src.copy()
        warped = compute_warp(overlay, src, dst)

        assert warped.shape == overlay.shape, f"shape mismatch for dtype={dtype}"
        assert warped.dtype == overlay.dtype, f"dtype mismatch for dtype={dtype}"
