"""Tests for :mod:`verso.engine.levels`."""

from __future__ import annotations

import numpy as np
import pytest

from verso.engine.levels import (
    MAX_GAMMA,
    MIN_GAMMA,
    Levels,
    apply_levels,
    is_bright_background,
    levels_lut,
    sample_background_level,
)


def test_default_is_the_identity():
    assert Levels().is_identity
    assert not Levels(black=10).is_identity
    assert not Levels(white=200).is_identity
    assert not Levels(gamma=2.0).is_identity


def test_identity_lut_is_a_straight_ramp():
    np.testing.assert_array_equal(levels_lut(Levels()), np.arange(256, dtype=np.uint8))


def test_black_and_white_points_clip_and_stretch():
    lut = levels_lut(Levels(black=50, white=200))
    assert lut[0] == 0
    assert lut[50] == 0
    assert lut[200] == 255
    assert lut[255] == 255
    # The midpoint of the window lands near mid-gray.
    assert abs(int(lut[125]) - 128) <= 2


def test_gamma_below_one_brightens_midtones():
    assert levels_lut(Levels(gamma=0.5))[128] > 128
    assert levels_lut(Levels(gamma=2.0))[128] < 128


def test_lut_is_monotonic():
    lut = levels_lut(Levels(black=20, white=230, gamma=1.7))
    assert np.all(np.diff(lut.astype(np.int16)) >= 0)


def test_normalized_clamps_out_of_range_values():
    lv = Levels(black=-30, white=999, gamma=50.0).normalized()
    assert lv.black == 0
    assert lv.white == 255
    assert lv.gamma == MAX_GAMMA
    assert Levels(gamma=0.0).normalized().gamma == MIN_GAMMA


def test_normalized_keeps_a_degenerate_window_usable():
    """white <= black would divide by zero; the window is widened instead."""
    lv = Levels(black=100, white=100).normalized()
    assert lv.white > lv.black
    lv = Levels(black=255, white=0).normalized()
    assert lv.white > lv.black
    # And the LUT for it is still well-formed (no NaN / no exception).
    assert levels_lut(Levels(black=100, white=100)).dtype == np.uint8


def test_apply_is_a_no_op_for_identity_and_none():
    rgb = np.random.default_rng(0).integers(0, 256, (4, 5, 3), dtype=np.uint8)
    assert apply_levels(rgb, None) is rgb
    assert apply_levels(rgb, Levels()) is rgb


def test_apply_maps_every_channel_through_the_lut():
    rgb = np.random.default_rng(1).integers(0, 256, (6, 7, 3), dtype=np.uint8)
    lv = Levels(black=30, white=210, gamma=1.4)
    out = apply_levels(rgb, lv)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, levels_lut(lv)[rgb])


def test_sample_ignores_a_single_outlier():
    rgb = np.full((20, 20, 3), 12, dtype=np.uint8)
    rgb[10, 10] = 255  # one hot pixel must not drag the median
    assert sample_background_level(rgb, 10, 10) == 12


def test_sample_uses_the_brightest_channel_not_the_average():
    """A tinted section's background lives in one channel; the LUT hits them all.

    Averaging (0, 34, 0) would report 11 and leave the background visible after
    the black point is applied.
    """
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:, :, 1] = 34  # green-tinted background
    assert sample_background_level(rgb, 10, 10) == 34
    # Applying that level does what the user asked: the background clips away.
    assert apply_levels(rgb, Levels(black=34, white=255)).max() == 0


def test_sample_clamps_at_the_edges():
    rgb = np.full((10, 10, 3), 40, dtype=np.uint8)
    assert sample_background_level(rgb, 0, 0) == 40
    assert sample_background_level(rgb, 9, 9) == 40


def test_sample_rejects_out_of_bounds():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(IndexError):
        sample_background_level(rgb, 10, 5)
    with pytest.raises(IndexError):
        sample_background_level(rgb, -1, 5)


def test_dark_background_is_not_reported_bright():
    """Fluorescence: dark background, bright signal."""
    rgb = np.full((60, 60, 3), 20, dtype=np.uint8)
    rgb[20:40, 20:40] = 230
    assert not is_bright_background(rgb, 20)


def test_bright_background_is_detected():
    """Brightfield: bright background, dark stain — the background is the maximum."""
    rgb = np.full((60, 60, 3), 240, dtype=np.uint8)
    rgb[20:40, 20:40] = 40
    assert is_bright_background(rgb, 240)


def test_bright_background_verdict_survives_hot_pixels():
    """The midpoint uses percentiles, so a few stray extremes cannot move it."""
    rgb = np.full((60, 60, 3), 240, dtype=np.uint8)
    rgb[20:40, 20:40] = 40
    rgb[0, 0] = 0  # dead pixel
    assert is_bright_background(rgb, 240)
