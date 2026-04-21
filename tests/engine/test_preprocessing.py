from __future__ import annotations

import numpy as np

from verso.engine.model.project import Preprocessing
from verso.engine.preprocessing import (
    apply_channel_luminance,
    apply_flip,
    apply_freehand_stroke,
    apply_mask,
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


def test_channel_luminance_matches_imadjust_direction() -> None:
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, :, 0] = [25, 50, 100]
    rgb[0, :, 1] = [10, 40, 80]
    rgb[0, :, 2] = [7, 8, 9]

    adjusted = apply_channel_luminance(rgb, red=0.5, green=0.25)

    np.testing.assert_array_equal(adjusted[0, :, 0], [50, 100, 200])
    np.testing.assert_array_equal(adjusted[0, :, 1], [40, 160, 255])
    np.testing.assert_array_equal(adjusted[0, :, 2], [7, 8, 9])


def test_channel_luminance_zero_hides_channel() -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)

    adjusted = apply_channel_luminance(rgb, red=0.0, green=1.0)

    assert np.all(adjusted[:, :, 0] == 0)
    assert np.all(adjusted[:, :, 1] == 100)
    assert np.all(adjusted[:, :, 2] == 100)


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
