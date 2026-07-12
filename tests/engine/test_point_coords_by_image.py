"""Tests for grouping a point series' coordinates by image."""

from __future__ import annotations

import numpy as np

from verso.engine.annotations import point_coords_by_image
from verso.engine.model.annotation import AnnotationPoint, PointSeries


def _series(points):
    return PointSeries(title="s", points=[AnnotationPoint(*p) for p in points])


def test_empty_series_returns_empty_dict():
    assert point_coords_by_image(PointSeries(title="s")) == {}


def test_groups_by_image_with_parallel_coords():
    series = _series(
        [
            (1.0, 2.0, "a.tif"),
            (3.0, 4.0, "b.tif"),
            (5.0, 6.0, "a.tif"),
        ]
    )
    out = point_coords_by_image(series)
    assert set(out) == {"a.tif", "b.tif"}
    xa, ya = out["a.tif"]
    assert list(xa) == [1.0, 5.0]
    assert list(ya) == [2.0, 6.0]
    xb, yb = out["b.tif"]
    assert list(xb) == [3.0] and list(yb) == [4.0]


def test_keys_are_basename_lowercased():
    series = _series([(1.0, 1.0, "/data/Sub/Section_01.TIF")])
    out = point_coords_by_image(series)
    assert set(out) == {"section_01.tif"}


def test_paths_and_basenames_collapse_to_one_bucket():
    # A raw path and a bare basename that normalise the same must merge, not
    # overwrite, keeping every point.
    series = _series(
        [
            (1.0, 1.0, "img.tif"),
            (2.0, 2.0, "/somewhere/IMG.tif"),
        ]
    )
    out = point_coords_by_image(series)
    assert set(out) == {"img.tif"}
    xs, ys = out["img.tif"]
    assert sorted(xs.tolist()) == [1.0, 2.0]
    assert sorted(ys.tolist()) == [1.0, 2.0]


def test_arrays_are_float():
    out = point_coords_by_image(_series([(1, 2, "a.tif")]))
    xs, ys = out["a.tif"]
    assert xs.dtype == np.float64 and ys.dtype == np.float64
