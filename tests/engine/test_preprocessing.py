from __future__ import annotations

import numpy as np

from verso.engine.model.project import ChannelSpec, Preprocessing
from verso.engine.preprocessing import (
    _sensitive_threshold,
    apply_brush_stroke,
    apply_flip,
    apply_freehand_stroke,
    apply_mask,
    channel_lut,
    composite_channels,
    detect_foreground,
    flip_lr_mask,
    line_side_polygons,
    load_lr_mask,
    load_mask,
    lr_mask_to_rgba,
    mask_to_rgba,
    rasterize_lr_line,
    save_lr_mask,
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


def test_sensitive_threshold_includes_more_dim_foreground() -> None:
    gray = np.full((40, 40), 0.85, dtype=np.float32)
    gray[10:30, 10:30] = 0.35
    gray[14:26, 14:26] = 0.1

    base = _sensitive_threshold(
        gray,
        bright_background=True,
        background_level=0.85,
        sensitivity=0.0,
    )
    sensitive = _sensitive_threshold(
        gray,
        bright_background=True,
        background_level=0.85,
        sensitivity=0.25,
    )

    assert sensitive > base


def test_sensitive_threshold_includes_more_faint_bright_foreground() -> None:
    gray = np.full((40, 40), 0.05, dtype=np.float32)
    gray[10:30, 10:30] = 0.45
    gray[14:26, 14:26] = 0.9

    base = _sensitive_threshold(
        gray,
        bright_background=False,
        background_level=0.05,
        sensitivity=0.0,
    )
    sensitive = _sensitive_threshold(
        gray,
        bright_background=False,
        background_level=0.05,
        sensitivity=0.25,
    )

    assert sensitive < base


# ---------------------------------------------------------------------------
# L/R hemisphere masks
# ---------------------------------------------------------------------------


def test_rasterize_lr_line_vertical_default_splits_left_right() -> None:
    # Vertical line down the centre (p0 above p1). Cross product convention:
    # negative side is "left", which for a downward-pointing line is the +x side.
    mask = rasterize_lr_line((50.0, 0.0), (50.0, 99.0), shape=(100, 100))
    # p0 → p1 points downward (dy>0, dx=0); cross = dx*(y-y0) - dy*(x-x0) = -dy*(x-x0).
    # x < 50 → cross > 0 → right (2); x > 50 → cross < 0 → left (1).
    assert (mask[:, 0:49] == 2).all()
    assert (mask[:, 51:] == 1).all()
    # On the line itself (x=50): cross == 0 → assigned 2 by convention.
    assert (mask[:, 50] == 2).all()


def test_rasterize_lr_line_reversed_endpoints_swaps_sides() -> None:
    a = rasterize_lr_line((50.0, 0.0), (50.0, 99.0), shape=(40, 100))
    b = rasterize_lr_line((50.0, 99.0), (50.0, 0.0), shape=(40, 100))
    # Reversing direction flips 1↔2 for every off-line pixel.
    swap = np.where(a == 1, 2, np.where(a == 2, 1, 0)).astype(np.uint8)
    # On-line pixels are 2 in both rasters by convention (cross == 0 → 2).
    on_line = (a == 2) & (b == 2) & (np.arange(100)[np.newaxis, :] == 50)
    off_line = ~on_line
    np.testing.assert_array_equal(swap[off_line], b[off_line])


def test_flip_lr_mask_horizontal_swaps_values_and_mirrors() -> None:
    m = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    out = flip_lr_mask(m, horizontal=True, vertical=False)
    # np.fliplr → [2, 2, 1, 1]; then 1↔2 swap → [1, 1, 2, 2]
    np.testing.assert_array_equal(out, [[1, 1, 2, 2]])


def test_flip_lr_mask_horizontal_is_involutive() -> None:
    m = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.uint8)
    once = flip_lr_mask(m, horizontal=True, vertical=False)
    twice = flip_lr_mask(once, horizontal=True, vertical=False)
    np.testing.assert_array_equal(twice, m)


def test_flip_lr_mask_vertical_mirrors_without_value_swap() -> None:
    m = np.array([[1, 1, 2, 2], [1, 0, 0, 2]], dtype=np.uint8)
    out = flip_lr_mask(m, horizontal=False, vertical=True)
    np.testing.assert_array_equal(out, [[1, 0, 0, 2], [1, 1, 2, 2]])


def test_save_load_lr_mask_round_trip(tmp_path) -> None:
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    path = tmp_path / "lr.png"
    save_lr_mask(mask, path)
    loaded = load_lr_mask(path, shape=mask.shape)
    np.testing.assert_array_equal(loaded, mask)


def test_lr_mask_to_rgba_assigns_distinct_colors_and_alpha() -> None:
    mask = np.array([[0, 1, 2]], dtype=np.uint8)
    rgba = lr_mask_to_rgba(
        mask,
        opacity=0.5,
        left_color=(10, 20, 30),
        right_color=(40, 50, 60),
    )
    # Unlabeled pixels: fully transparent.
    np.testing.assert_array_equal(rgba[0, 0], [0, 0, 0, 0])
    # Left pixels: red-ish tint at 50% alpha = 128 (round(0.5 * 255)).
    np.testing.assert_array_equal(rgba[0, 1], [10, 20, 30, 128])
    np.testing.assert_array_equal(rgba[0, 2], [40, 50, 60, 128])


def test_line_side_polygons_vertical_midline_splits_rect() -> None:
    left, right = line_side_polygons((50.0, 0.0), (50.0, 100.0), 100.0, 100.0)
    # Each side should be a 4-vertex polygon (left/right halves of the rect).
    assert len(left) == 4
    assert len(right) == 4
    # Left side (negative cross product) is +x side for a downward line.
    assert (left[:, 0] >= 49.999).all()
    assert (right[:, 0] <= 50.001).all()


def test_line_side_polygons_horizontal_midline() -> None:
    left, right = line_side_polygons((0.0, 50.0), (100.0, 50.0), 100.0, 100.0)
    # Line points rightward (dx>0, dy=0); cross = dx*(y-y0) → negative for y<50.
    assert len(left) == 4
    assert len(right) == 4
    assert (left[:, 1] <= 50.001).all()
    assert (right[:, 1] >= 49.999).all()


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
