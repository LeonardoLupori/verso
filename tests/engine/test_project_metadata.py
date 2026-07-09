"""Tests for populate_metadata (image dims + atlas resolution/shape at creation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from verso.engine.io.project_metadata import AtlasUnavailableError, populate_metadata
from verso.engine.model.project import Project


class _StubAtlas:
    """Stand-in for AtlasVolume so the test needs no atlas download."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.resolution_um = 25.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (528, 320, 456)


def _write_tiff(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _write_project_missing_metadata(project_dir: Path, original_path: Path) -> Path:
    """Write a project file with unpopulated dims and atlas resolution/shape."""
    data = {
        "version": "1.0",
        "name": "Test",
        "atlas": {"name": "allen_mouse_25um", "source": "brainglobe"},
        "interpolation_axis": "AP",
        "channels": [],
        "working_scale": 0.2,
        "sections": [
            {
                "id": "s001",
                "slice_index": 1,
                "original_path": str(original_path),
                "thumbnail_path": "thumbnails/s001.ome.tif",
            }
        ],
    }
    path = project_dir / "project-verso.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_populate_metadata_fills_dims_and_atlas(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("verso.engine.atlas.AtlasVolume", _StubAtlas)

    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=2000, height=1500)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=400, height=300)
    project_path = _write_project_missing_metadata(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    populate_metadata(project, tmp_path)

    s = project.sections[0]
    assert s.resolution_original_wh == (2000, 1500)
    assert s.resolution_thumbnail_wh == (400, 300)
    assert project.atlas.resolution_um == 25.0
    assert project.atlas.shape == (528, 320, 456)


def test_populate_metadata_uses_provided_volume_without_constructing(tmp_path: Path) -> None:
    # No monkeypatch: passing an atlas must avoid constructing AtlasVolume at all.
    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=100, height=80)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=20, height=16)
    project_path = _write_project_missing_metadata(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    populate_metadata(project, tmp_path, atlas=_StubAtlas("allen_mouse_25um"))

    assert project.atlas.resolution_um == 25.0
    assert project.atlas.shape == (528, 320, 456)


def test_populate_metadata_raises_atlas_unavailable_when_atlas_fails(
    tmp_path: Path, monkeypatch
) -> None:
    # Atlas construction fails (e.g. offline) but the images are present: the
    # caller must be able to tell this apart from an image-I/O failure.
    def _boom(name: str):
        raise ConnectionError("no internet")

    monkeypatch.setattr("verso.engine.atlas.AtlasVolume", _boom)

    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=100, height=80)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=20, height=16)
    project_path = _write_project_missing_metadata(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    with pytest.raises(AtlasUnavailableError):
        populate_metadata(project, tmp_path)


def test_populate_metadata_missing_image_raises_file_not_found(tmp_path: Path) -> None:
    # A missing image is a FileNotFoundError, NOT an AtlasUnavailableError, so the
    # GUI does not misreport an image problem as an atlas download failure.
    original = tmp_path / "raw" / "MISSING.tif"
    project_path = _write_project_missing_metadata(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    with pytest.raises(FileNotFoundError):
        populate_metadata(project, tmp_path, atlas=_StubAtlas("allen_mouse_25um"))
