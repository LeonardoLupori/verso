from __future__ import annotations

import numpy as np

from verso.engine.model.project import ChannelSpec, Preprocessing
from verso.engine.preprocessing import (
    _sensitive_threshold,
    apply_flip,
    apply_freehand_stroke,
    apply_mask,
    composite_channels,
    detect_foreground,
    load_mask,
    mask_to_rgba,
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
    image[0, :, 0] = [200, 0, 0]      # bright in channel 0
    image[0, :, 1] = [0, 200, 0]      # bright in channel 1
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
    full_scale = composite_channels(
        image, [ChannelSpec(name="x", color=(255, 0, 0), scale=1.0)]
    )
    boosted = composite_channels(
        image, [ChannelSpec(name="x", color=(255, 0, 0), scale=0.5)]
    )
    assert boosted[0, 0, 0] > full_scale[0, 0, 0]


def test_composite_channels_handles_2d_input() -> None:
    image = np.full((2, 2), 128, dtype=np.uint8)
    rgb = composite_channels(
        image, [ChannelSpec(name="x", color=(255, 255, 255), scale=1.0)]
    )
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
