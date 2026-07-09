"""Tests for the annotation model and folder-based I/O."""

from __future__ import annotations

from pathlib import Path

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
from verso.engine.model.annotation import POINT_SERIES, AnnotationPoint, PointSeries

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_point_series_metadata_round_trip():
    series = PointSeries(
        title="cells ch1",
        color=(10, 20, 30),
        opacity=0.5,
        visible=False,
        points=[AnnotationPoint(1.0, 2.0, "a.tif")],
    )
    restored = PointSeries.from_metadata(series.metadata_to_dict(), series.points)
    assert restored == series
    assert restored.type == POINT_SERIES


def test_point_series_metadata_defaults():
    series = PointSeries.from_metadata({"title": "x"}, [])
    assert series.color == (255, 64, 64)
    assert series.opacity == 1.0
    assert series.visible is True
    assert series.points == []


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
        PointSeries(title="cells ch2", color=(0, 255, 0), opacity=0.3, visible=False),
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
