"""Tests for the lasso point-in-polygon hit test."""

from __future__ import annotations

import numpy as np

from verso.engine.annotations import points_in_polygon

# A unit square (0,0)-(10,10).
_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_inside_and_outside():
    pts = [(5.0, 5.0), (15.0, 5.0), (-1.0, 5.0), (2.0, 8.0)]
    mask = points_in_polygon(pts, _SQUARE)
    assert list(mask) == [True, False, False, True]


def test_empty_points_returns_empty_mask():
    mask = points_in_polygon(np.empty((0, 2)), _SQUARE)
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_degenerate_polygon_is_all_false():
    pts = [(5.0, 5.0), (1.0, 1.0)]
    mask = points_in_polygon(pts, [(0.0, 0.0), (10.0, 0.0)])  # 2 vertices
    assert list(mask) == [False, False]


def test_concave_polygon():
    # An arrow/chevron: the notch at the top excludes points near (5, 9).
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 4.0), (0.0, 10.0)]
    mask = points_in_polygon([(5.0, 9.0), (2.0, 2.0), (8.0, 2.0)], poly)
    assert list(mask) == [False, True, True]


def test_matches_shape_of_input():
    pts = np.random.default_rng(0).uniform(-5, 15, size=(50, 2))
    mask = points_in_polygon(pts, _SQUARE)
    assert mask.shape == (50,)
    # Cross-check a couple against the square bounds (interior points only, so
    # edge ambiguity does not bite).
    for p, inside in zip(pts, mask, strict=False):
        if 0.5 < p[0] < 9.5 and 0.5 < p[1] < 9.5:
            assert inside
