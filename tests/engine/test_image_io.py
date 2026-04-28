"""Tests for image IO helpers."""

import numpy as np

from verso.engine.io.image_io import imadjust, parse_section_serial_number, thumbnail_filename


def test_parse_section_serial_number_from_mouse_name():
    assert parse_section_serial_number("MOUSE_0042_CODEs.tif", fallback=1) == 42


def test_parse_section_serial_number_falls_back_to_list_order():
    assert parse_section_serial_number("section_without_number.tif", fallback=7) == 7


def test_thumbnail_filename_uses_source_stem():
    assert thumbnail_filename("MOUSE_0042_CODEs.tif") == "MOUSE_0042_CODEs-thumb.png"


def test_imadjust_keeps_blue_channel_unchanged_for_color_images():
    rgb = np.array(
        [
            [[10, 20, 1], [30, 60, 2], [50, 100, 3]],
            [[70, 140, 4], [90, 180, 5], [110, 220, 6]],
        ],
        dtype=np.uint8,
    )

    adjusted = imadjust(rgb)

    assert not np.array_equal(adjusted[:, :, 0], rgb[:, :, 0])
    assert not np.array_equal(adjusted[:, :, 1], rgb[:, :, 1])
    np.testing.assert_array_equal(adjusted[:, :, 2], rgb[:, :, 2])


def test_imadjust_keeps_grayscale_rgb_neutral():
    gray = np.array([[10, 20, 40], [80, 120, 200]], dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    adjusted = imadjust(rgb)

    np.testing.assert_array_equal(adjusted[:, :, 0], adjusted[:, :, 1])
    np.testing.assert_array_equal(adjusted[:, :, 0], adjusted[:, :, 2])
