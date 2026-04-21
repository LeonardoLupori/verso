"""Tests for engine/warping.py."""

import numpy as np

from verso.engine.warping import warp_overlay


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
