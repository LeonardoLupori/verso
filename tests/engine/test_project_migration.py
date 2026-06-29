"""Tests for the v1.1 -> v1.2 project metadata backfill (image dims + atlas meta)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from verso.engine.io.project_io import backfill_metadata
from verso.engine.model.project import SCHEMA_VERSION, Project


class _StubAtlas:
    """Stand-in for AtlasVolume so the backfill test needs no atlas download."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.resolution_um = 25.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (528, 320, 456)


def _write_tiff(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _write_v11_project(project_dir: Path, original_path: Path) -> Path:
    """Write a legacy v1.1 project file (no dims, atlas without resolution/shape)."""
    data = {
        "version": "1.1",
        "name": "Legacy",
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


def test_backfill_metadata_fills_dims_and_atlas(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("verso.engine.atlas.AtlasVolume", _StubAtlas)

    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=2000, height=1500)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=400, height=300)
    project_path = _write_v11_project(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    backfill_metadata(project, tmp_path)

    s = project.sections[0]
    assert s.resolution_original_wh == (2000, 1500)
    assert s.resolution_thumbnail_wh == (400, 300)
    assert project.atlas.resolution_um == 25.0
    assert project.atlas.shape == (528, 320, 456)


def test_load_migrates_v11_and_bumps_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("verso.engine.atlas.AtlasVolume", _StubAtlas)

    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=2000, height=1500)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=400, height=300)
    project_path = _write_v11_project(tmp_path, original)

    project = Project.load(project_path)

    assert project.version == SCHEMA_VERSION
    s = project.sections[0]
    assert s.resolution_original_wh == (2000, 1500)
    assert s.resolution_thumbnail_wh == (400, 300)
    assert project.atlas.resolution_um == 25.0
    assert project.atlas.shape == (528, 320, 456)

    # The migrated metadata persists on the next save.
    project.save(project_path)
    reloaded = json.loads(project_path.read_text())
    assert reloaded["version"] == SCHEMA_VERSION
    assert reloaded["sections"][0]["resolution_original_wh"] == [2000, 1500]
    assert reloaded["atlas"]["resolution_um"] == 25.0


def test_backfill_atlas_uses_provided_volume_without_constructing(tmp_path: Path) -> None:
    # No monkeypatch: passing an atlas must avoid constructing AtlasVolume at all.
    original = tmp_path / "raw" / "IMG.tif"
    _write_tiff(original, width=100, height=80)
    _write_tiff(tmp_path / "thumbnails" / "s001.ome.tif", width=20, height=16)
    project_path = _write_v11_project(tmp_path, original)

    project = Project.from_dict(json.loads(project_path.read_text()))
    backfill_metadata(project, tmp_path, atlas=_StubAtlas("allen_mouse_25um"))

    assert project.atlas.resolution_um == 25.0
    assert project.atlas.shape == (528, 320, 456)
