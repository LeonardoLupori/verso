"""Tests for engine/io/quint_io.py — QuickNII / VisuAlign I/O."""

import json
from pathlib import Path

from verso.engine.io.quint_io import (
    _control_points_to_markers,
    _markers_to_control_points,
    _to_quicknii_convention,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_visualign,
)
from verso.engine.model.alignment import AlignmentStatus, ControlPoint
from verso.engine.model.project import AtlasRef, Project, Section

QUICKNII_JSON = {
    "name": "test_dataset",
    "target": "allen_mouse_25um",
    "sections": [
        {
            "filename": "IMG_0001.png",
            "nr": 1,
            "anchoring": [0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
        },
        {
            "filename": "IMG_0002.png",
            "nr": 2,
            "anchoring": [0.0, 160.0, 250.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
        },
    ],
}

VISUALIGN_JSON = {
    **QUICKNII_JSON,
    "sections": [
        {
            **QUICKNII_JSON["sections"][0],
            "markers": [
                {"x": 0.5, "y": 0.3, "dx": 0.02, "dy": -0.01},
                {"x": 0.1, "y": 0.8, "dx": -0.05, "dy": 0.03},
            ],
        },
        QUICKNII_JSON["sections"][1],
    ],
}


# ---------------------------------------------------------------------------
# Marker ↔ ControlPoint conversion
# ---------------------------------------------------------------------------

def test_markers_to_control_points():
    markers = [{"x": 0.5, "y": 0.25, "dx": 0.1, "dy": -0.05}]
    cps = _markers_to_control_points(markers)

    assert len(cps) == 1
    cp = cps[0]
    assert abs(cp.src_x - 0.5) < 1e-6
    assert abs(cp.src_y - 0.25) < 1e-6
    assert abs(cp.dst_x - 0.6) < 1e-6
    assert abs(cp.dst_y - 0.20) < 1e-6


def test_control_points_to_markers():
    cps = [ControlPoint(src_x=0.5, src_y=0.25, dst_x=0.6, dst_y=0.20)]
    markers = _control_points_to_markers(cps)

    assert len(markers) == 1
    m = markers[0]
    assert abs(m["x"] - 0.5) < 1e-5
    assert abs(m["y"] - 0.25) < 1e-5
    assert abs(m["dx"] - 0.1) < 1e-5
    assert abs(m["dy"] - (-0.05)) < 1e-5


def test_marker_control_point_round_trip():
    original = [
        {"x": 0.3, "y": 0.6, "dx": 0.05, "dy": -0.02},
        {"x": 0.8, "y": 0.1, "dx": -0.03, "dy": 0.07},
    ]
    cps = _markers_to_control_points(original)
    restored = _control_points_to_markers(cps)

    for orig, res in zip(original, restored):
        for key in ("x", "y", "dx", "dy"):
            assert abs(orig[key] - res[key]) < 1e-5, f"{key} mismatch"


# ---------------------------------------------------------------------------
# load_quicknii
# ---------------------------------------------------------------------------

def test_load_quicknii_section_count(tmp_path: Path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert len(project.sections) == 2


def test_load_quicknii_anchoring(tmp_path: Path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert project.sections[0].alignment.anchoring == QUICKNII_JSON["sections"][0]["anchoring"]


def test_load_quicknii_atlas_name(tmp_path: Path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert project.atlas.name == "allen_mouse_25um"


# ---------------------------------------------------------------------------
# load_visualign
# ---------------------------------------------------------------------------

def test_load_visualign_control_points(tmp_path: Path):
    p = tmp_path / "va.json"
    p.write_text(json.dumps(VISUALIGN_JSON))

    project = load_visualign(p)
    s0 = project.sections[0]

    assert len(s0.warp.control_points) == 2
    assert s0.warp.status == AlignmentStatus.COMPLETE


def test_load_visualign_section_without_markers(tmp_path: Path):
    p = tmp_path / "va.json"
    p.write_text(json.dumps(VISUALIGN_JSON))

    project = load_visualign(p)
    s1 = project.sections[1]

    assert s1.warp.control_points == []


# ---------------------------------------------------------------------------
# save_quicknii / load_quicknii round-trip
# ---------------------------------------------------------------------------

def test_quicknii_save_load_round_trip(tmp_path: Path):
    src = tmp_path / "qn.json"
    src.write_text(json.dumps(QUICKNII_JSON))
    project = load_quicknii(src)

    dst = tmp_path / "qn_out.json"
    save_quicknii(project, dst)

    reloaded = load_quicknii(dst)
    assert len(reloaded.sections) == len(project.sections)
    for orig, rel in zip(project.sections, reloaded.sections):
        assert rel.alignment.anchoring == orig.alignment.anchoring
        assert rel.serial_number == orig.serial_number


def test_save_quicknii_uses_registration_thumbnail_dimensions(tmp_path: Path):
    from PIL import Image

    original = tmp_path / "original.png"
    thumbnail = tmp_path / "thumbnail.png"
    Image.new("RGB", (1000, 800)).save(original)
    Image.new("RGB", (200, 160)).save(thumbnail)

    project = Project(
        name="scale_test",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                id="s001",
                serial_number=1,
                original_path=str(original),
                thumbnail_path=str(thumbnail),
                scale=0.2,
            )
        ],
    )

    dst = tmp_path / "qn_out.json"
    save_quicknii(project, dst)

    data = json.loads(dst.read_text())
    assert data["slices"][0]["width"] == 200
    assert data["slices"][0]["height"] == 160
    assert data["slices"][0]["filename"] == "thumbnail.png"


def test_save_quicknii_writes_relative_thumbnail_path(tmp_path: Path):
    from PIL import Image

    project_dir = tmp_path / "project"
    thumbnails = project_dir / "thumbnails"
    exports = project_dir / "exports"
    raw_dir = tmp_path / "raw"
    thumbnails.mkdir(parents=True)
    exports.mkdir()
    raw_dir.mkdir()

    original = raw_dir / "section.tif"
    thumbnail = thumbnails / "s001.png"
    Image.new("RGB", (1000, 800)).save(original)
    Image.new("RGB", (200, 160)).save(thumbnail)

    project = Project(
        name="path_test",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                id="s001",
                serial_number=1,
                original_path=str(original),
                thumbnail_path=str(thumbnail),
                scale=0.2,
            )
        ],
    )

    dst = exports / "quicknii.json"
    save_quicknii(project, dst)

    data = json.loads(dst.read_text())
    assert data["slices"][0]["filename"] == "../thumbnails/s001.png"


def test_quicknii_export_convention_flips_ap_and_dv_axes():
    anchoring = [
        10.0, 100.0, 40.0,
        20.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
    ]

    converted = _to_quicknii_convention(anchoring, atlas_shape=(528, 320, 456))

    assert converted == [
        10.0, 428.0, 280.0,
        20.0, -3.0, -4.0,
        5.0, -6.0, -7.0,
    ]
    assert _to_quicknii_convention(converted, atlas_shape=(528, 320, 456)) == anchoring


# ---------------------------------------------------------------------------
# save_visualign / load_visualign round-trip
# ---------------------------------------------------------------------------

def test_visualign_save_load_round_trip(tmp_path: Path):
    src = tmp_path / "va.json"
    src.write_text(json.dumps(VISUALIGN_JSON))
    project = load_visualign(src)

    dst = tmp_path / "va_out.json"
    save_visualign(project, dst)

    reloaded = load_visualign(dst)

    for orig, rel in zip(project.sections, reloaded.sections):
        assert len(rel.warp.control_points) == len(orig.warp.control_points)
        for cp_orig, cp_rel in zip(orig.warp.control_points, rel.warp.control_points):
            assert abs(cp_orig.src_x - cp_rel.src_x) < 0.1
            assert abs(cp_orig.src_y - cp_rel.src_y) < 0.1
            assert abs(cp_orig.dst_x - cp_rel.dst_x) < 0.1
            assert abs(cp_orig.dst_y - cp_rel.dst_y) < 0.1
