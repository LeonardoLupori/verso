"""Tests for the annotation model and folder-based I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verso.engine.io.annotation_io import (
    annotations_dir,
    guess_point_columns,
    load_annotations,
    load_points_csv,
    read_points_csv,
    save_annotations,
    slugify,
    write_points_csv,
)
from verso.engine.model.annotation import (
    AREA,
    POINT_SERIES,
    AnnotationPoint,
    AreaAnnotation,
    PointSeries,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_point_series_metadata_round_trip():
    series = PointSeries(
        title="cells ch1",
        color=(10, 20, 30),
        visible=False,
        points=[AnnotationPoint(1.0, 2.0, "a.tif")],
    )
    restored = PointSeries.from_metadata(series.metadata_to_dict(), series.points)
    assert restored == series
    assert restored.type == POINT_SERIES


def test_point_series_metadata_defaults():
    series = PointSeries.from_metadata({"title": "x"}, [])
    assert series.color == (255, 64, 64)
    assert series.visible is True
    assert series.points == []


def test_point_series_ignores_legacy_opacity():
    # Point series used to persist an opacity; loading old data must drop it.
    series = PointSeries.from_metadata({"title": "x", "opacity": 0.3}, [])
    assert not hasattr(series, "opacity")
    assert "opacity" not in series.metadata_to_dict()


# ---------------------------------------------------------------------------
# CSV round-trip and import
# ---------------------------------------------------------------------------


def test_points_csv_round_trip(tmp_path: Path):
    points = [
        AnnotationPoint(12043.0, 8801.5, "slide07_s03.tif"),
        AnnotationPoint(1.0, 2.0, "b.tif"),
    ]
    path = tmp_path / "points.csv"
    write_points_csv(path, points)
    assert read_points_csv(path) == points


def test_load_points_csv_without_image_uses_default(tmp_path: Path):
    path = tmp_path / "cells.csv"
    path.write_text("x,y\n10,20\n30,40\n", encoding="utf-8")
    points = load_points_csv(path, "x", "y", image_col=None, default_image="cur.tif")
    assert points == [
        AnnotationPoint(10.0, 20.0, "cur.tif"),
        AnnotationPoint(30.0, 40.0, "cur.tif"),
    ]


def test_load_points_csv_skips_unparsable_rows(tmp_path: Path):
    path = tmp_path / "cells.csv"
    path.write_text("x,y,image\n10,20,a.tif\nnan_header,,a.tif\n30,40,a.tif\n", encoding="utf-8")
    points = load_points_csv(path, "x", "y", "image")
    assert points == [AnnotationPoint(10.0, 20.0, "a.tif"), AnnotationPoint(30.0, 40.0, "a.tif")]


# ---------------------------------------------------------------------------
# Smart column guessing
# ---------------------------------------------------------------------------


def test_guess_point_columns_exact():
    guess = guess_point_columns(["x", "y", "image"])
    assert guess == {"x": "x", "y": "y", "image": "image"}


def test_guess_point_columns_aliases_case_insensitive():
    guess = guess_point_columns(["Centroid_X", "CENTROID_Y", "FileName", "area"])
    assert guess == {"x": "Centroid_X", "y": "CENTROID_Y", "image": "FileName"}


def test_guess_point_columns_missing_image_is_none():
    guess = guess_point_columns(["pos_x", "pos_y"])
    assert guess["x"] == "pos_x"
    assert guess["y"] == "pos_y"
    assert guess["image"] is None


def test_guess_point_columns_unresolvable_is_none():
    guess = guess_point_columns(["foo", "bar"])
    assert guess == {"x": None, "y": None, "image": None}


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------


def test_slugify_sanitises_and_falls_back():
    assert slugify("cells ch1") == "cells_ch1"
    assert slugify("  a/b:c  ") == "a_b_c"
    assert slugify("///") == "annotation"


# ---------------------------------------------------------------------------
# Folder persistence
# ---------------------------------------------------------------------------


def test_save_load_annotations_round_trip(tmp_path: Path):
    anns = [
        PointSeries(title="cells ch1", color=(255, 0, 0), points=[AnnotationPoint(1, 2, "a.tif")]),
        PointSeries(title="cells ch2", color=(0, 255, 0), visible=False),
    ]
    save_annotations(tmp_path, anns)

    loaded = load_annotations(tmp_path)
    assert loaded == anns
    # Folders are named by slug of the title.
    assert (annotations_dir(tmp_path) / "cells_ch1" / "annotation.json").exists()
    assert (annotations_dir(tmp_path) / "cells_ch1" / "points.csv").exists()


def test_save_annotations_deduplicates_folder_names(tmp_path: Path):
    anns = [PointSeries(title="dupe"), PointSeries(title="dupe")]
    save_annotations(tmp_path, anns)
    names = sorted(p.name for p in annotations_dir(tmp_path).iterdir())
    assert names == ["dupe", "dupe_2"]
    assert len(load_annotations(tmp_path)) == 2


def test_save_annotations_removes_stale_folders(tmp_path: Path):
    save_annotations(tmp_path, [PointSeries(title="keep"), PointSeries(title="drop")])
    assert (annotations_dir(tmp_path) / "drop").exists()

    # A later save with a smaller set prunes the removed annotation's folder.
    save_annotations(tmp_path, [PointSeries(title="keep")])
    remaining = sorted(p.name for p in annotations_dir(tmp_path).iterdir())
    assert remaining == ["keep"]


def test_load_annotations_missing_folder_is_empty(tmp_path: Path):
    assert load_annotations(tmp_path) == []


def test_load_annotations_skips_folder_without_metadata(tmp_path: Path):
    root = annotations_dir(tmp_path)
    (root / "not_an_annotation").mkdir(parents=True)
    (root / "not_an_annotation" / "points.csv").write_text("x,y,image\n", encoding="utf-8")
    assert load_annotations(tmp_path) == []


# ---------------------------------------------------------------------------
# Area annotations
# ---------------------------------------------------------------------------


def _mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """A bool mask of shape (h, w) with ``box`` = (r0, r1, c0, c1) set True."""
    m = np.zeros((h, w), dtype=bool)
    r0, r1, c0, c1 = box
    m[r0:r1, c0:c1] = True
    return m


def test_area_metadata_round_trip():
    area = AreaAnnotation(title="injection", color=(10, 20, 30), opacity=0.5, visible=False)
    restored = AreaAnnotation.from_metadata(area.metadata_to_dict(), {})
    assert restored == area
    assert restored.type == AREA


def test_area_metadata_defaults():
    area = AreaAnnotation.from_metadata({"title": "x", "type": AREA}, {})
    assert area.opacity == 0.5  # areas default to semi-transparent
    assert area.visible is True
    assert area.masks == {}


def test_area_save_load_round_trip(tmp_path: Path):
    masks = {
        "s001.tif": _mask(20, 30, (2, 10, 3, 12)),
        "s005.tif": _mask(20, 30, (5, 15, 8, 20)),
    }
    area = AreaAnnotation(title="injection", color=(255, 0, 0), opacity=0.4, masks=masks)
    save_annotations(tmp_path, [area])

    loaded = load_annotations(tmp_path)
    assert len(loaded) == 1
    got = loaded[0]
    assert isinstance(got, AreaAnnotation)
    assert got.title == "injection"
    assert got.color == (255, 0, 0)
    assert got.opacity == 0.4
    assert set(got.masks) == {"s001.tif", "s005.tif"}
    for name, mask in masks.items():
        assert np.array_equal(got.masks[name], mask)
    # Mask files are named <image>.png so the basename round-trips.
    assert (annotations_dir(tmp_path) / "injection" / "masks" / "s001.tif.png").exists()


def test_area_skips_and_prunes_empty_masks(tmp_path: Path):
    area = AreaAnnotation(
        title="a",
        masks={"s001.tif": _mask(10, 10, (1, 5, 1, 5)), "s002.tif": np.zeros((10, 10), bool)},
    )
    save_annotations(tmp_path, [area])
    masks_dir = annotations_dir(tmp_path) / "a" / "masks"
    # The all-False mask leaves no PNG behind.
    assert (masks_dir / "s001.tif.png").exists()
    assert not (masks_dir / "s002.tif.png").exists()
    assert set(load_annotations(tmp_path)[0].masks) == {"s001.tif"}


def test_area_save_prunes_removed_section_mask(tmp_path: Path):
    area = AreaAnnotation(
        title="a",
        masks={"s001.tif": _mask(10, 10, (1, 5, 1, 5)), "s002.tif": _mask(10, 10, (2, 6, 2, 6))},
    )
    save_annotations(tmp_path, [area])
    masks_dir = annotations_dir(tmp_path) / "a" / "masks"
    assert (masks_dir / "s002.tif.png").exists()

    # Drop one section's mask and re-save: its PNG must be pruned.
    area.masks.pop("s002.tif")
    save_annotations(tmp_path, [area])
    assert not (masks_dir / "s002.tif.png").exists()
    assert (masks_dir / "s001.tif.png").exists()


def test_mixed_point_and_area_round_trip(tmp_path: Path):
    points = PointSeries(
        title="cells", color=(0, 255, 0), points=[AnnotationPoint(1, 2, "s001.tif")]
    )
    area = AreaAnnotation(title="injection", masks={"s001.tif": _mask(8, 8, (1, 4, 1, 4))})
    save_annotations(tmp_path, [points, area])

    loaded = load_annotations(tmp_path)
    kinds = {type(a).__name__ for a in loaded}
    assert kinds == {"PointSeries", "AreaAnnotation"}
    by_title = {a.title: a for a in loaded}
    assert by_title["cells"].points == points.points
    assert np.array_equal(by_title["injection"].masks["s001.tif"], area.masks["s001.tif"])
