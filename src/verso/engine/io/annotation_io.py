"""Folder-based persistence for annotations.

Annotations live in an ``annotations/`` subfolder of the project directory, one
subfolder per annotation named by a filesystem-safe slug of its title::

    my_experiment/
        annotations/
            cells_ch1/
                annotation.json   # {type, title, color, opacity, visible}
                points.csv        # header: x,y,image  (original-res px, filename)

Metadata lives in ``annotation.json``; point data lives in ``points.csv`` so it
stays inspectable and importable/exportable on its own. This module has no Qt
dependency, so it is usable from scripts and tests.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

from verso.engine.model.annotation import POINT_SERIES, AnnotationPoint, PointSeries

_ANNOTATIONS_DIRNAME = "annotations"
_METADATA_FILENAME = "annotation.json"
_POINTS_FILENAME = "points.csv"

# Column-name aliases for smart CSV import, matched case-insensitively.
_X_ALIASES = ("x", "pos_x", "position_x", "centroid_x", "x_px", "px", "col", "column")
_Y_ALIASES = ("y", "pos_y", "position_y", "centroid_y", "y_px", "py", "row")
_IMAGE_ALIASES = ("image", "img", "file", "filename", "file_name", "section", "slice")


# ---------------------------------------------------------------------------
# Paths / slugs
# ---------------------------------------------------------------------------


def annotations_dir(project_dir: str | Path) -> Path:
    """Return the ``annotations/`` folder for a project directory."""
    return Path(project_dir) / _ANNOTATIONS_DIRNAME


def slugify(title: str) -> str:
    """Turn an annotation title into a filesystem-safe folder name."""
    slug = re.sub(r"[^\w\-]+", "_", title.strip()).strip("_")
    return slug or "annotation"


def _unique_slug(base: str, used: set[str]) -> str:
    """Return ``base``, suffixed with ``_2``, ``_3``… if already in ``used``."""
    if base not in used:
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_points_csv(path: str | Path, points: Iterable[AnnotationPoint]) -> None:
    """Write points to a canonical ``x,y,image`` CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "image"])
        for p in points:
            writer.writerow([p.x, p.y, p.image])


def guess_point_columns(headers: Sequence[str]) -> dict[str, str | None]:
    """Guess which CSV columns hold x, y, and image.

    Matching is case-insensitive against known aliases (see the module-level
    ``_*_ALIASES``). The returned dict always has keys ``x``, ``y`` and
    ``image``; a value is ``None`` when no column could be matched (the caller
    should prompt the user — x and y are required).
    """
    lookup = {str(h).strip().lower(): h for h in headers}

    def pick(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            if alias in lookup:
                return lookup[alias]
        return None

    return {"x": pick(_X_ALIASES), "y": pick(_Y_ALIASES), "image": pick(_IMAGE_ALIASES)}


def load_points_csv(
    path: str | Path,
    x_col: str,
    y_col: str,
    image_col: str | None = None,
    default_image: str = "",
) -> list[AnnotationPoint]:
    """Load points from an arbitrary CSV using the given column names.

    Rows whose x/y cannot be parsed as floats are skipped. When ``image_col`` is
    ``None`` (or a row's image cell is empty), the point is assigned
    ``default_image`` — typically the current section's filename.
    """
    points: list[AnnotationPoint] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except (KeyError, ValueError, TypeError):
                continue
            image = (row.get(image_col) if image_col else None) or default_image
            points.append(AnnotationPoint(x=x, y=y, image=str(image)))
    return points


def read_points_csv(path: str | Path) -> list[AnnotationPoint]:
    """Read a canonical ``x,y,image`` points CSV written by VERSO."""
    return load_points_csv(path, "x", "y", "image", default_image="")


# ---------------------------------------------------------------------------
# Annotation folders
# ---------------------------------------------------------------------------


def _write_annotation(folder: Path, annotation: PointSeries) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / _METADATA_FILENAME).write_text(
        json.dumps(annotation.metadata_to_dict(), indent=2), encoding="utf-8"
    )
    write_points_csv(folder / _POINTS_FILENAME, annotation.points)


def load_annotation(folder: str | Path) -> PointSeries:
    """Load a single annotation from its folder.

    Raises:
        FileNotFoundError: If the folder has no ``annotation.json``.
    """
    folder = Path(folder)
    meta = json.loads((folder / _METADATA_FILENAME).read_text(encoding="utf-8"))
    points_path = folder / _POINTS_FILENAME
    points = read_points_csv(points_path) if points_path.exists() else []
    # Only one type today; dispatch here when Area annotations are added.
    ann_type = str(meta.get("type", POINT_SERIES))
    if ann_type != POINT_SERIES:
        raise ValueError(f"Unsupported annotation type {ann_type!r} in {folder}")
    return PointSeries.from_metadata(meta, points)


def load_annotations(project_dir: str | Path) -> list[PointSeries]:
    """Load every annotation in a project's ``annotations/`` folder.

    Folders without a readable ``annotation.json`` are skipped. Returns an empty
    list when the project has no ``annotations/`` folder yet.
    """
    root = annotations_dir(project_dir)
    if not root.exists():
        return []
    out: list[PointSeries] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / _METADATA_FILENAME).exists():
            try:
                out.append(load_annotation(child))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return out


def save_annotations(project_dir: str | Path, annotations: Sequence[PointSeries]) -> None:
    """Persist the full annotation set, syncing the ``annotations/`` folder.

    Each annotation is (re)written to ``annotations/<unique-slug>/``. Folders
    that no longer correspond to any annotation (deletions and renames) are
    removed, so the on-disk folder always mirrors ``annotations`` exactly.
    """
    root = annotations_dir(project_dir)
    root.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    for annotation in annotations:
        folder_name = _unique_slug(slugify(annotation.title), used)
        used.add(folder_name)
        _write_annotation(root / folder_name, annotation)

    for child in root.iterdir():
        if child.is_dir() and child.name not in used:
            shutil.rmtree(child, ignore_errors=True)
