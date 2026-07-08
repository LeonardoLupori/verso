"""Tests for engine/io/quint_io.py — QuickNII / VisuAlign I/O."""

import json
from pathlib import Path

from verso.engine.io.quint_io import (
    _control_points_to_markers,
    _markers_to_control_points,
    _to_quicknii_convention,
    load_deepslice,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_quicknii_xml,
    save_visualign,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint
from verso.engine.model.project import AtlasRef, Project, Section

# Native QuickNII/VisuAlign format uses "slices" key.
# Anchoring values here are in QuickNII convention (component 1 = AP, 0 = posterior).
QUICKNII_JSON = {
    "name": "test_dataset",
    "target": "allen_mouse_25um",
    "slices": [
        {
            "filename": "IMG_0001.png",
            "nr": 1,
            "width": 1000,
            "height": 800,
            "anchoring": [0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
        },
        {
            "filename": "IMG_0002.png",
            "nr": 2,
            "width": 1000,
            "height": 800,
            "anchoring": [0.0, 160.0, 250.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
        },
    ],
}

# Anchoring after load (converted from QuickNII → BrainGlobe by _to_quicknii_convention):
#   _to_quicknii_convention([0,160,228,456,0,0,0,320,0], (528,320,456))
#   origin flips about N-1 (array reversal): 527-160, 319-228
#   = [0, 367, 91, 456, -0, -0, 0, -320, -0]
_LOADED_ANCHORING_0 = [0.0, 367.0, 91.0, 456.0, 0.0, 0.0, 0.0, -320.0, 0.0]
_LOADED_ANCHORING_1 = [0.0, 367.0, 69.0, 456.0, 0.0, 0.0, 0.0, -320.0, 0.0]

# VisuAlign native marker format: [src_x_px, src_y_px, dst_x_px, dst_y_px]
VISUALIGN_JSON = {
    **QUICKNII_JSON,
    "slices": [
        {
            **QUICKNII_JSON["slices"][0],
            "markers": [
                [500.0, 240.0, 520.0, 232.0],
                [100.0, 640.0, 50.0, 664.0],
            ],
        },
        QUICKNII_JSON["slices"][1],
    ],
}


# ---------------------------------------------------------------------------
# Marker ↔ ControlPoint conversion
# ---------------------------------------------------------------------------


def test_markers_to_control_points_pixel_array():
    """Native VisuAlign format: pixel-coordinate arrays stored as-is."""
    markers = [[500.0, 200.0, 600.0, 180.0]]
    cps = _markers_to_control_points(markers, width=1000, height=800)

    assert len(cps) == 1
    cp = cps[0]
    assert abs(cp.src_x - 500.0) < 1e-6
    assert abs(cp.src_y - 200.0) < 1e-6
    assert abs(cp.dst_x - 600.0) < 1e-6
    assert abs(cp.dst_y - 180.0) < 1e-6


def test_markers_to_control_points_legacy_dict():
    """Backward-compat: legacy VERSO dict format {x, y, dx, dy}."""
    markers = [{"x": 0.5, "y": 0.25, "dx": 0.1, "dy": -0.05}]
    cps = _markers_to_control_points(markers, width=1, height=1)

    assert len(cps) == 1
    cp = cps[0]
    assert abs(cp.src_x - 0.5) < 1e-6
    assert abs(cp.src_y - 0.25) < 1e-6
    assert abs(cp.dst_x - 0.6) < 1e-6
    assert abs(cp.dst_y - 0.20) < 1e-6


def test_control_points_to_markers():
    """Output should be pixel-coordinate arrays matching VisuAlign native format."""
    cps = [ControlPoint(src_x=500.0, src_y=200.0, dst_x=600.0, dst_y=160.0)]
    markers = _control_points_to_markers(cps)

    assert len(markers) == 1
    m = markers[0]
    assert isinstance(m, list)
    assert abs(m[0] - 500.0) < 1e-3
    assert abs(m[1] - 200.0) < 1e-3
    assert abs(m[2] - 600.0) < 1e-3
    assert abs(m[3] - 160.0) < 1e-3


def test_marker_control_point_round_trip():
    """Pixel-array markers survive a load → save round-trip with no precision loss."""
    W, H = 1000, 800
    original = [
        [300.0, 480.0, 350.0, 464.0],
        [800.0, 80.0, 770.0, 136.0],
    ]
    cps = _markers_to_control_points(original, width=W, height=H)
    restored = _control_points_to_markers(cps)

    for orig, res in zip(original, restored, strict=True):
        for i in range(4):
            assert abs(orig[i] - res[i]) < 1e-3, f"component {i} mismatch"


# ---------------------------------------------------------------------------
# load_quicknii
# ---------------------------------------------------------------------------


def test_load_quicknii_section_count(tmp_path: Path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert len(project.sections) == 2


def test_load_quicknii_anchoring(tmp_path: Path):
    """Loaded anchoring must be converted from QuickNII to BrainGlobe convention."""
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert project.sections[0].alignment.current_anchoring == _LOADED_ANCHORING_0


def test_load_quicknii_atlas_name(tmp_path: Path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(QUICKNII_JSON))

    project = load_quicknii(p)
    assert project.atlas.name == "allen_mouse_25um"


def test_load_deepslice_marks_suggestions_in_progress(tmp_path: Path):
    data = {
        **QUICKNII_JSON,
        "slices": [
            {**QUICKNII_JSON["slices"][0], "confidence": 0.91},
            QUICKNII_JSON["slices"][1],
        ],
    }
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(data))

    project = load_deepslice(p)
    s0 = project.sections[0]

    assert s0.alignment.status == AlignmentStatus.IN_PROGRESS
    assert s0.alignment.source == "deepslice"


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


def test_load_visualign_marker_values(tmp_path: Path):
    """Pixel-array markers are stored as-is (working-resolution pixels) on load."""
    p = tmp_path / "va.json"
    p.write_text(json.dumps(VISUALIGN_JSON))

    project = load_visualign(p)
    s0 = project.sections[0]
    cp0 = s0.warp.control_points[0]

    # First marker: [500, 240, 520, 232] — stored directly as pixels
    assert abs(cp0.src_x - 500.0) < 1e-5
    assert abs(cp0.src_y - 240.0) < 1e-5
    assert abs(cp0.dst_x - 520.0) < 1e-5
    assert abs(cp0.dst_y - 232.0) < 1e-5


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
    for orig, rel in zip(project.sections, reloaded.sections, strict=True):
        # Both loaded in BrainGlobe convention — must be identical
        for a, b in zip(
            orig.alignment.current_anchoring, rel.alignment.current_anchoring, strict=True
        ):
            assert abs(a - b) < 1e-3
        assert rel.slice_index == orig.slice_index


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
                slice_index=1,
                original_path=str(original),
                thumbnail_path=str(thumbnail),
                resolution_thumbnail_wh=(200, 160),
            )
        ],
        working_scale=0.2,
    )

    dst = tmp_path / "qn_out.json"
    save_quicknii(project, dst)

    data = json.loads(dst.read_text())
    assert data["slices"][0]["width"] == 200
    assert data["slices"][0]["height"] == 160
    # Filename is derived from original_path stem + -thumb.png
    assert data["slices"][0]["filename"] == "thumbnails/original-thumb.png"


def test_save_quicknii_writes_relative_thumbnail_path(tmp_path: Path):
    from PIL import Image

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    original = raw_dir / "section.tif"
    thumbnail = tmp_path / "section-thumb.png"
    Image.new("RGB", (1000, 800)).save(original)
    Image.new("RGB", (200, 160)).save(thumbnail)

    project = Project(
        name="path_test",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                id="s001",
                slice_index=1,
                original_path=str(original),
                thumbnail_path=str(thumbnail),
            )
        ],
        working_scale=0.2,
    )

    dst = tmp_path / "quicknii.json"
    save_quicknii(project, dst)

    data = json.loads(dst.read_text())
    # Filename is always thumbnails/{stem}-thumb.png relative to export dir
    assert data["slices"][0]["filename"] == "thumbnails/section-thumb.png"


def test_save_quicknii_xml_infers_atlas_shape(tmp_path: Path):
    """save_quicknii_xml must self-infer atlas_shape (like save_quicknii / save_visualign).

    Without inference, the XML anchoring is left in BrainGlobe convention (AP/DV
    not flipped) and ``target-resolution`` is omitted — inconsistent with the
    VisuAlign JSON, which does infer. The inferred output must be byte-identical
    to passing the atlas_shape explicitly.
    """
    project = Project(
        name="xmltest",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                id="s001",
                slice_index=1,
                original_path="a.tif",
                thumbnail_path="",
                alignment=Alignment(
                    current_anchoring=[10.0, 100.0, 40.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    status=AlignmentStatus.COMPLETE,
                ),
            )
        ],
    )

    inferred_path = tmp_path / "inferred.xml"
    explicit_path = tmp_path / "explicit.xml"
    save_quicknii_xml(project, inferred_path)  # infers
    save_quicknii_xml(project, explicit_path, atlas_shape=(528, 320, 456))  # explicit

    inferred = inferred_path.read_text(encoding="utf-8")
    assert inferred == explicit_path.read_text(encoding="utf-8")
    assert "target-resolution='528 320 456'" in inferred
    # QuickNII convention applied: AP flips (527 - 100 = 427), DV flips (319 - 40 = 279).
    assert "oy=427" in inferred
    assert "oz=279" in inferred


def test_quicknii_export_convention_flips_ap_and_dv_axes():
    anchoring = [
        10.0,
        100.0,
        40.0,
        20.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]

    converted = _to_quicknii_convention(anchoring, atlas_shape=(528, 320, 456))

    # Origin flips about N-1 (527 - 100 = 427, 319 - 40 = 279); vectors negate.
    assert converted == [
        10.0,
        427.0,
        279.0,
        20.0,
        -3.0,
        -4.0,
        5.0,
        -6.0,
        -7.0,
    ]
    assert _to_quicknii_convention(converted, atlas_shape=(528, 320, 456)) == anchoring


def test_sagittal_export_independent_of_slicing_direction(tmp_path: Path):
    """Issue #16 regression: a stored section's exported QuickNII/VisuAlign
    coordinates depend only on its stored anchoring, never on the project's
    slicing axis or proposal direction. The slicing axis only affects *proposals*
    for un-aligned sections, which are never written. The LR component (0) is
    exported unflipped (the asymmetric ML axis is preserved); only AP/DV flip.
    """
    stored = [200.0, 100.0, 40.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def build(axis_name: str) -> Project:
        return Project(
            name="sag",
            atlas=AtlasRef(name="allen_mouse_25um"),
            interpolation_axis=axis_name,
            sections=[
                Section(
                    id="s001",
                    slice_index=1,
                    original_path="a.tif",
                    thumbnail_path="",
                    alignment=Alignment(
                        current_anchoring=list(stored),
                        status=AlignmentStatus.COMPLETE,
                    ),
                )
            ],
        )

    for save_fn, suffix in ((save_quicknii, "json"), (save_visualign, "va.json")):
        sag = tmp_path / f"sagittal.{suffix}"
        cor = tmp_path / f"coronal.{suffix}"
        save_fn(build("ML"), sag)
        save_fn(build("AP"), cor)
        # Same stored alignment + same atlas → byte-identical export regardless
        # of slicing direction.
        assert sag.read_text(encoding="utf-8") == cor.read_text(encoding="utf-8")

        exported = json.loads(sag.read_text(encoding="utf-8"))["slices"][0]["anchoring"]
        assert exported[0] == 200.0  # LR preserved (no ML flip)
        assert exported[1] == 527.0 - 100.0  # AP flips about N-1
        assert exported[2] == 319.0 - 40.0  # DV flips about N-1


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

    for orig, rel in zip(project.sections, reloaded.sections, strict=True):
        assert len(rel.warp.control_points) == len(orig.warp.control_points)
        for cp_orig, cp_rel in zip(orig.warp.control_points, rel.warp.control_points, strict=True):
            assert abs(cp_orig.src_x - cp_rel.src_x) < 1e-4
            assert abs(cp_orig.src_y - cp_rel.src_y) < 1e-4
            assert abs(cp_orig.dst_x - cp_rel.dst_x) < 1e-4
            assert abs(cp_orig.dst_y - cp_rel.dst_y) < 1e-4
