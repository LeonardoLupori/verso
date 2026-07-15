"""Tests for ``annotation_images`` — which sections an annotation covers."""

from __future__ import annotations

import numpy as np

from verso.engine.annotations import annotation_images
from verso.engine.model.annotation import (
    AnnotationPoint,
    AreaAnnotation,
    PointSeries,
)


def test_point_series_returns_lowercased_basenames():
    series = PointSeries(
        title="pts",
        points=[
            AnnotationPoint(1.0, 2.0, "Slide_01.tif"),
            AnnotationPoint(3.0, 4.0, "slide_01.tif"),  # same image, other case
            AnnotationPoint(5.0, 6.0, "sub/dir/Slide_02.TIF"),  # path -> basename
        ],
    )
    assert annotation_images(series) == {"slide_01.tif", "slide_02.tif"}


def test_empty_point_series_covers_nothing():
    assert annotation_images(PointSeries(title="empty")) == set()


def test_area_counts_only_non_empty_masks():
    area = AreaAnnotation(
        title="area",
        masks={
            "Slide_01.tif": np.ones((4, 4), dtype=bool),
            "slide_02.tif": np.zeros((4, 4), dtype=bool),  # painted then erased
        },
    )
    assert annotation_images(area) == {"slide_01.tif"}
