"""Tests for :mod:`verso.engine.colormaps`."""

from __future__ import annotations

import numpy as np
import pytest

from verso.engine.colormaps import NONE, apply_colormap, colormap_lut, colormap_names


def test_none_is_first_and_maps_are_listed():
    names = colormap_names()
    assert names[0] == NONE
    assert "Viridis" in names and "Magma" in names


def test_lut_shape_and_dtype():
    lut = colormap_lut("Viridis")
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8


def test_invert_reverses_the_lut():
    lut = colormap_lut("Magma")
    inverted = colormap_lut("Magma", invert=True)
    np.testing.assert_array_equal(inverted, lut[::-1])


def test_name_lookup_is_case_insensitive():
    np.testing.assert_array_equal(colormap_lut("viridis"), colormap_lut("Viridis"))


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        colormap_lut("not-a-colormap")


def test_none_passes_the_image_through_unchanged():
    rgb = np.random.default_rng(0).integers(0, 256, (4, 5, 3), dtype=np.uint8)
    assert apply_colormap(rgb, NONE) is rgb


def test_apply_maps_luminance_through_the_lut():
    # A pure-gray ramp: luminance equals the input value, so each output pixel
    # must be exactly that LUT entry.
    ramp = np.tile(np.arange(256, dtype=np.uint8)[:, None, None], (1, 2, 3))
    out = apply_colormap(ramp, "Turbo")
    assert out.shape == ramp.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, 0, :], colormap_lut("Turbo"))
