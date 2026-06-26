from __future__ import annotations

import numpy as np
import pytest

from verso.engine.model.project import ChannelSpec, Preprocessing
from verso.engine.preprocessing import (
    apply_brush_stroke,
    apply_flip,
    apply_freehand_stroke,
    apply_mask,
    channel_lut,
    composite_channels,
    detect_foreground,
    load_mask,
    mask_to_rgba,
    morph_mask,
    save_mask,
)


def test_apply_flip_respects_horizontal_and_vertical_flags() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessing = Preprocessing(flip_horizontal=True, flip_vertical=True)

    flipped = apply_flip(image, preprocessing)

    np.testing.assert_array_equal(flipped, np.flipud(np.fliplr(image)))
    assert flipped.flags.c_contiguous


def test_apply_mask_zeros_background_pixels() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
    mask = np.array([[True, False], [False, True], [True, True]])

    masked = apply_mask(image, mask)

    np.testing.assert_array_equal(masked[0, 0], image[0, 0])
    np.testing.assert_array_equal(masked[0, 1], [0, 0, 0])
    np.testing.assert_array_equal(masked[1, 0], [0, 0, 0])


def test_composite_channels_max_blends_visible_channels() -> None:
    image = np.zeros((1, 3, 2), dtype=np.uint8)
    image[0, :, 0] = [200, 0, 0]  # bright in channel 0
    image[0, :, 1] = [0, 200, 0]  # bright in channel 1
    channels = [
        ChannelSpec(name="ch0", color=(255, 0, 0), scale=1.0, visible=True),
        ChannelSpec(name="ch1", color=(0, 255, 0), scale=1.0, visible=True),
    ]

    rgb = composite_channels(image, channels)

    assert rgb.shape == (1, 3, 3)
    # First pixel: channel 0 is bright → red contribution.
    assert rgb[0, 0, 0] > 100 and rgb[0, 0, 1] == 0
    # Second pixel: channel 1 is bright → green contribution.
    assert rgb[0, 1, 1] > 100 and rgb[0, 1, 0] == 0
    # Third pixel: nothing bright in either channel.
    assert rgb[0, 2, 0] == 0 and rgb[0, 2, 1] == 0


def test_composite_channels_invisible_channel_contributes_nothing() -> None:
    image = np.full((1, 1, 2), 200, dtype=np.uint8)
    channels = [
        ChannelSpec(name="ch0", color=(255, 0, 0), scale=1.0, visible=False),
        ChannelSpec(name="ch1", color=(0, 255, 0), scale=1.0, visible=True),
    ]
    rgb = composite_channels(image, channels)
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 0, 1] > 100


def test_composite_channels_scale_brightens_channel() -> None:
    image = np.full((1, 1, 1), 50, dtype=np.uint8)
    full_scale = composite_channels(image, [ChannelSpec(name="x", color=(255, 0, 0), scale=1.0)])
    boosted = composite_channels(image, [ChannelSpec(name="x", color=(255, 0, 0), scale=0.5)])
    assert boosted[0, 0, 0] > full_scale[0, 0, 0]


def test_composite_channels_handles_2d_input() -> None:
    image = np.full((2, 2), 128, dtype=np.uint8)
    rgb = composite_channels(image, [ChannelSpec(name="x", color=(255, 255, 255), scale=1.0)])
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0, 0] == 128


def test_composite_channels_empty_specs_returns_black() -> None:
    image = np.full((2, 2, 3), 200, dtype=np.uint8)
    rgb = composite_channels(image, [])
    assert rgb.shape == (2, 2, 3)
    assert np.all(rgb == 0)


def test_mask_save_load_roundtrip_and_resize(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]])
    path = tmp_path / "mask.png"

    save_mask(mask, path)
    loaded = load_mask(path, shape=(4, 4))

    assert loaded.dtype == bool
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    np.testing.assert_array_equal(loaded, expected)


def test_mask_to_rgba_positive_and_negative_polarity() -> None:
    mask = np.array([[True, False]])

    positive = mask_to_rgba(mask, negative=False, opacity=0.5, color=(1, 2, 3))
    negative = mask_to_rgba(mask, negative=True, opacity=0.5, color=(1, 2, 3))

    np.testing.assert_array_equal(positive[0, 0], [1, 2, 3, 128])
    np.testing.assert_array_equal(positive[0, 1], [1, 2, 3, 0])
    np.testing.assert_array_equal(negative[0, 0], [1, 2, 3, 0])
    np.testing.assert_array_equal(negative[0, 1], [1, 2, 3, 128])


def test_apply_freehand_stroke_adds_and_erases_polygon() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    polygon = np.array([[2, 2], [7, 2], [7, 7], [2, 7]])

    added = apply_freehand_stroke(mask, polygon, add=True)
    erased = apply_freehand_stroke(added, polygon, add=False)

    assert added[4, 4]
    assert not added[0, 0]
    assert not erased[4, 4]


def test_apply_brush_stroke_single_point_paints_disk() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    out = apply_brush_stroke(mask, np.array([[25, 25]]), radius=5, add=True)

    assert out[25, 25]
    assert out[25, 29]  # within radius
    assert not out[25, 40]  # outside radius
    assert not mask[25, 25]  # original untouched


def test_apply_brush_stroke_connects_sparse_points() -> None:
    mask = np.zeros((20, 60), dtype=bool)
    out = apply_brush_stroke(mask, np.array([[5, 10], [55, 10]]), radius=3, add=True)

    # The midpoint between the two stamps must be filled (no gap).
    assert out[10, 30]


def test_apply_brush_stroke_erases() -> None:
    mask = np.ones((30, 30), dtype=bool)
    out = apply_brush_stroke(mask, np.array([[15, 15]]), radius=4, add=False)

    assert not out[15, 15]
    assert out[0, 0]  # outside the brush untouched


def test_detect_foreground_dark_tissue_on_bright_background() -> None:
    rgb = np.full((80, 80, 3), 240, dtype=np.uint8)
    rgb[20:60, 25:55] = 40

    mask = detect_foreground(rgb)

    assert mask[40, 40]
    assert not mask[5, 5]


def test_detect_foreground_bright_tissue_on_dark_background() -> None:
    rgb = np.full((80, 80, 3), 10, dtype=np.uint8)
    rgb[20:60, 25:55] = 230

    mask = detect_foreground(rgb)

    assert mask[40, 40]
    assert not mask[5, 5]


def test_channel_lut_shape_and_dtype() -> None:
    spec = ChannelSpec(name="ch0", color=(255, 255, 255), scale=1.0, visible=True)
    lut = channel_lut(spec)
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8
    assert (lut[:, 3] == 255).all()


def test_channel_lut_identity_at_full_scale() -> None:
    # scale=1.0 + white tint → output luminance equals input intensity per channel.
    spec = ChannelSpec(name="ch0", color=(255, 255, 255), scale=1.0, visible=True)
    lut = channel_lut(spec)
    np.testing.assert_array_equal(lut[:, 0], np.arange(256, dtype=np.uint8))
    np.testing.assert_array_equal(lut[:, 1], np.arange(256, dtype=np.uint8))
    np.testing.assert_array_equal(lut[:, 2], np.arange(256, dtype=np.uint8))


def test_channel_lut_brightness_boost_clips_at_255() -> None:
    # scale=0.5 doubles intensity; pixels ≥ 128 must saturate to 255.
    spec = ChannelSpec(name="ch0", color=(255, 255, 255), scale=0.5, visible=True)
    lut = channel_lut(spec)
    assert lut[64, 0] == 128
    assert lut[128, 0] == 255
    assert lut[200, 0] == 255


def test_channel_lut_tints_to_color() -> None:
    # Pure red tint → green/blue channels stay zero, red follows luminance.
    spec = ChannelSpec(name="ch0", color=(255, 0, 0), scale=1.0, visible=True)
    lut = channel_lut(spec)
    assert (lut[:, 1] == 0).all()
    assert (lut[:, 2] == 0).all()
    np.testing.assert_array_equal(lut[:, 0], np.arange(256, dtype=np.uint8))


def test_channel_lut_matches_composite_channels_for_single_channel() -> None:
    # The LUT path must produce the same RGB image as composite_channels does
    # for a single-channel input — that is the property the GUI relies on.
    plane = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = plane[..., np.newaxis]
    spec = ChannelSpec(name="ch0", color=(0, 200, 100), scale=0.6, visible=True)

    expected = composite_channels(image, [spec])
    lut = channel_lut(spec)
    via_lut = lut[plane][..., :3]
    np.testing.assert_array_equal(via_lut, expected)


# ---------------------------------------------------------------------------
# apply_flip — individual flags and 3-D images
# ---------------------------------------------------------------------------


def test_apply_flip_horizontal_only() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessing = Preprocessing(flip_horizontal=True, flip_vertical=False)
    flipped = apply_flip(image, preprocessing)
    np.testing.assert_array_equal(flipped, np.fliplr(image))
    assert flipped.flags.c_contiguous


def test_apply_flip_vertical_only() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessing = Preprocessing(flip_horizontal=False, flip_vertical=True)
    flipped = apply_flip(image, preprocessing)
    np.testing.assert_array_equal(flipped, np.flipud(image))


def test_apply_flip_no_flags_returns_identical_array() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessing = Preprocessing(flip_horizontal=False, flip_vertical=False)
    out = apply_flip(image, preprocessing)
    np.testing.assert_array_equal(out, image)


def test_apply_flip_3d_image() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    preprocessing = Preprocessing(flip_horizontal=True, flip_vertical=True)
    out = apply_flip(image, preprocessing)
    np.testing.assert_array_equal(out, np.flipud(np.fliplr(image)))
    assert out.flags.c_contiguous


# ---------------------------------------------------------------------------
# apply_mask — 2-D image, shape mismatch, copy semantics
# ---------------------------------------------------------------------------


def test_apply_mask_2d_image() -> None:
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    mask = np.array([[True, False, True], [False, True, False]])
    out = apply_mask(image, mask)
    assert out[0, 0] == image[0, 0]
    assert out[0, 1] == 0
    assert out[1, 2] == 0


def test_apply_mask_raises_on_shape_mismatch() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.ones((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        apply_mask(image, mask)


def test_apply_mask_does_not_mutate_original() -> None:
    image = np.ones((3, 3), dtype=np.uint8) * 100
    mask = np.zeros((3, 3), dtype=bool)
    apply_mask(image, mask)
    assert (image == 100).all()


# ---------------------------------------------------------------------------
# composite_channels — edge cases
# ---------------------------------------------------------------------------


def test_composite_channels_zero_scale_contributes_nothing() -> None:
    image = np.full((1, 1, 1), 200, dtype=np.uint8)
    channels = [ChannelSpec(name="x", color=(255, 0, 0), scale=0.0, visible=True)]
    rgb = composite_channels(image, channels)
    assert np.all(rgb == 0)


def test_composite_channels_more_specs_than_planes_ignores_excess() -> None:
    image = np.full((1, 1, 1), 100, dtype=np.uint8)
    channels = [
        ChannelSpec(name="ch0", color=(255, 0, 0), scale=1.0, visible=True),
        ChannelSpec(name="ch1", color=(0, 255, 0), scale=1.0, visible=True),
    ]
    rgb = composite_channels(image, channels)
    assert rgb.shape == (1, 1, 3)
    assert rgb[0, 0, 0] > 0
    assert rgb[0, 0, 1] == 0


def test_composite_channels_invalid_ndim_raises() -> None:
    image = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        composite_channels(image, [])


# ---------------------------------------------------------------------------
# detect_foreground — 2-D input and uniform-image fallback
# ---------------------------------------------------------------------------


def test_detect_foreground_2d_grayscale_input() -> None:
    gray = np.full((80, 80), 240, dtype=np.uint8)
    gray[20:60, 25:55] = 40
    mask = detect_foreground(gray)
    assert mask[40, 40]
    assert not mask[5, 5]


def test_detect_foreground_uniform_image_returns_all_ones() -> None:
    # Completely uniform image: _usable_mask will fail → fallback to all-True.
    image = np.full((50, 50), 128, dtype=np.uint8)
    mask = detect_foreground(image)
    assert mask.shape == (50, 50)
    assert mask.all()


# ---------------------------------------------------------------------------
# apply_freehand_stroke — degenerate polygon inputs
# ---------------------------------------------------------------------------


def test_apply_freehand_stroke_fewer_than_3_points_is_noop() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    out = apply_freehand_stroke(mask, np.array([[2, 2], [7, 7]]), add=True)
    np.testing.assert_array_equal(out, mask)


def test_apply_freehand_stroke_empty_polygon_is_noop() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    out = apply_freehand_stroke(mask, np.empty((0, 2)), add=True)
    np.testing.assert_array_equal(out, mask)


# ---------------------------------------------------------------------------
# apply_brush_stroke — empty input and zero radius clamp
# ---------------------------------------------------------------------------


def test_apply_brush_stroke_empty_points_is_noop() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    out = apply_brush_stroke(mask, np.empty((0, 2)), radius=5, add=True)
    np.testing.assert_array_equal(out, mask)


def test_apply_brush_stroke_zero_radius_clamped_to_1() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    out = apply_brush_stroke(mask, np.array([[10, 10]]), radius=0, add=True)
    assert out[10, 10]


# ---------------------------------------------------------------------------
# morph_mask — erode and expand (completely untested previously)
# ---------------------------------------------------------------------------


def test_morph_mask_expand_grows_foreground() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[14:16, 14:16] = True
    expanded = morph_mask(mask, pixels=4, operation="expand")
    assert expanded.dtype == bool
    assert expanded[10, 14]  # 4 px above seed, within radius
    assert not expanded[0, 0]


def test_morph_mask_erode_shrinks_foreground() -> None:
    mask = np.ones((30, 30), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    eroded = morph_mask(mask, pixels=3, operation="erode")
    assert eroded.dtype == bool
    assert not eroded[1, 1]  # border area should be eroded away
    assert eroded[15, 15]  # centre should survive


def test_morph_mask_expand_then_erode_approximates_original() -> None:
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    expanded = morph_mask(mask, pixels=5, operation="expand")
    restored = morph_mask(expanded, pixels=5, operation="erode")
    # After expand+erode the original interior must still be foreground.
    assert restored[25:35, 25:35].all()


def test_morph_mask_radius_1_is_minimum() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    out = morph_mask(mask, pixels=0, operation="expand")
    assert out[5, 5]


def test_morph_mask_returns_bool_array() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    assert morph_mask(mask, pixels=2, operation="expand").dtype == bool
    assert morph_mask(mask, pixels=2, operation="erode").dtype == bool


# ---------------------------------------------------------------------------
# mask_to_rgba — opacity clamping
# ---------------------------------------------------------------------------


def test_mask_to_rgba_opacity_clamped_above_1() -> None:
    mask = np.array([[True]])
    rgba = mask_to_rgba(mask, negative=False, opacity=5.0)
    assert rgba[0, 0, 3] == 255


def test_mask_to_rgba_opacity_clamped_below_0() -> None:
    mask = np.array([[True]])
    rgba = mask_to_rgba(mask, negative=False, opacity=-1.0)
    assert rgba[0, 0, 3] == 0
