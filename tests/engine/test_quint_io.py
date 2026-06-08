"""Tests for engine/io/quint_io.py — QuickNII / VisuAlign I/O."""

import gzip
import json
import struct
from pathlib import Path

import numpy as np

from verso.engine.io.quint_io import (
    _control_points_to_markers,
    _markers_to_control_points,
    _to_quicknii_convention,
    export_brainglobe_atlas_for_visualign,
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
#   = [0, 528-160, 320-228, 456, -0, -0, 0, -320, -0]
#   = [0, 368, 92, 456, 0, 0, 0, -320, 0]
_LOADED_ANCHORING_0 = [0.0, 368.0, 92.0, 456.0, 0.0, 0.0, 0.0, -320.0, 0.0]
_LOADED_ANCHORING_1 = [0.0, 368.0, 70.0, 456.0, 0.0, 0.0, 0.0, -320.0, 0.0]

# VisuAlign native marker format: [src_x_px, src_y_px, dst_x_px, dst_y_px]
VISUALIGN_JSON = {
    **QUICKNII_JSON,
    "slices": [
        {
            **QUICKNII_JSON["slices"][0],
            "markers": [
                [500.0, 240.0, 520.0, 232.0],
                [100.0, 640.0,  50.0, 664.0],
            ],
        },
        QUICKNII_JSON["slices"][1],
    ],
}


# ---------------------------------------------------------------------------
# Marker ↔ ControlPoint conversion
# ---------------------------------------------------------------------------

def test_markers_to_control_points_pixel_array():
    """Native VisuAlign format: pixel-coordinate arrays."""
    markers = [[500.0, 200.0, 600.0, 180.0]]
    cps = _markers_to_control_points(markers, width=1000, height=800)

    assert len(cps) == 1
    cp = cps[0]
    assert abs(cp.src_x - 0.5) < 1e-6     # 500 / 1000
    assert abs(cp.src_y - 0.25) < 1e-6    # 200 / 800
    assert abs(cp.dst_x - 0.6) < 1e-6     # 600 / 1000
    assert abs(cp.dst_y - 0.225) < 1e-6   # 180 / 800


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
    cps = [ControlPoint(src_x=0.5, src_y=0.25, dst_x=0.6, dst_y=0.20)]
    markers = _control_points_to_markers(cps, width=1000, height=800)

    assert len(markers) == 1
    m = markers[0]
    assert isinstance(m, list)
    assert abs(m[0] - 500.0) < 1e-3   # src_x * 1000
    assert abs(m[1] - 200.0) < 1e-3   # src_y * 800
    assert abs(m[2] - 600.0) < 1e-3   # dst_x * 1000
    assert abs(m[3] - 160.0) < 1e-3   # dst_y * 800


def test_marker_control_point_round_trip():
    """Pixel-array markers survive a load → save round-trip with no precision loss."""
    W, H = 1000, 800
    original = [
        [300.0, 480.0, 350.0, 464.0],
        [800.0,  80.0, 770.0, 136.0],
    ]
    cps = _markers_to_control_points(original, width=W, height=H)
    restored = _control_points_to_markers(cps, width=W, height=H)

    for orig, res in zip(original, restored):
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
    assert project.sections[0].alignment.anchoring == _LOADED_ANCHORING_0


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
    # proposal_anchoring is set from the already-converted BrainGlobe anchoring
    assert s0.alignment.proposal_anchoring == _LOADED_ANCHORING_0
    assert s0.alignment.proposal_confidence == 0.91


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
    """Pixel-array markers are correctly normalised on load."""
    p = tmp_path / "va.json"
    p.write_text(json.dumps(VISUALIGN_JSON))

    project = load_visualign(p)
    s0 = project.sections[0]
    cp0 = s0.warp.control_points[0]

    # First marker: [500, 240, 520, 232] at 1000×800
    assert abs(cp0.src_x - 0.5) < 1e-5     # 500 / 1000
    assert abs(cp0.src_y - 0.3) < 1e-5     # 240 / 800
    assert abs(cp0.dst_x - 0.52) < 1e-5    # 520 / 1000
    assert abs(cp0.dst_y - 0.29) < 1e-5    # 232 / 800


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
        # Both loaded in BrainGlobe convention — must be identical
        for a, b in zip(orig.alignment.anchoring, rel.alignment.anchoring):
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
                    anchoring=[10.0, 100.0, 40.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    status=AlignmentStatus.COMPLETE,
                ),
            )
        ],
    )

    inferred_path = tmp_path / "inferred.xml"
    explicit_path = tmp_path / "explicit.xml"
    save_quicknii_xml(project, inferred_path)                                # infers
    save_quicknii_xml(project, explicit_path, atlas_shape=(528, 320, 456))   # explicit

    inferred = inferred_path.read_text(encoding="utf-8")
    assert inferred == explicit_path.read_text(encoding="utf-8")
    assert "target-resolution='528 320 456'" in inferred
    # QuickNII convention applied: AP flips (528 - 100 = 428), DV flips (320 - 40 = 280).
    assert "oy=428" in inferred
    assert "oz=280" in inferred


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
    from PIL import Image

    # Create actual thumbnail images so _registration_dims returns non-zero values
    thumb = tmp_path / "s001-thumb.png"
    Image.new("RGB", (1000, 800)).save(thumb)

    src = tmp_path / "va.json"
    src.write_text(json.dumps(VISUALIGN_JSON))

    # Patch thumbnail paths so save_visualign can read dimensions
    project = load_visualign(src)
    for section in project.sections:
        section.thumbnail_path = str(thumb)

    dst = tmp_path / "va_out.json"
    save_visualign(project, dst)

    reloaded = load_visualign(dst)

    for orig, rel in zip(project.sections, reloaded.sections):
        assert len(rel.warp.control_points) == len(orig.warp.control_points)
        for cp_orig, cp_rel in zip(orig.warp.control_points, rel.warp.control_points):
            assert abs(cp_orig.src_x - cp_rel.src_x) < 1e-4
            assert abs(cp_orig.src_y - cp_rel.src_y) < 1e-4
            assert abs(cp_orig.dst_x - cp_rel.dst_x) < 1e-4
            assert abs(cp_orig.dst_y - cp_rel.dst_y) < 1e-4


# ---------------------------------------------------------------------------
# export_brainglobe_atlas_for_visualign
# ---------------------------------------------------------------------------

class _MockBGAtlas:
    """Minimal BrainGlobeAtlas stand-in for testing the NIfTI writer."""

    def __init__(self, annotation: np.ndarray) -> None:
        self.annotation = annotation
        self.structures = {
            8: {"name": "Basic cell groups", "rgb_triplet": [191, 218, 227]},
            997: {"name": "root", "rgb_triplet": [255, 255, 255]},
        }


def _run_export(annotation: np.ndarray, tmp_path: Path) -> Path:
    """Call export_brainglobe_atlas_for_visualign with a mocked BrainGlobeAtlas."""
    import unittest.mock as mock

    mock_bg = _MockBGAtlas(annotation)
    with mock.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas",
        return_value=mock_bg,
    ):
        return export_brainglobe_atlas_for_visualign("test_atlas", tmp_path)


def test_export_creates_cutlas_directory(tmp_path: Path):
    ann = np.zeros((4, 3, 5), dtype=np.uint32)  # (AP, DV, LR)
    cutlas_dir = _run_export(ann, tmp_path)

    assert cutlas_dir.is_dir()
    assert (cutlas_dir / "labels.nii.gz").exists()
    assert (cutlas_dir / "labels.txt").exists()
    assert cutlas_dir.name == "test_atlas.cutlas"


def test_export_nifti_header_shape(tmp_path: Path):
    """NIfTI dim fields must reflect transposed volume shape (LR, AP, DV)."""
    # BG shape: (AP=4, DV=3, LR=5)  → expected NIfTI shape: (LR=5, AP=4, DV=3)
    ann = np.zeros((4, 3, 5), dtype=np.uint32)
    cutlas_dir = _run_export(ann, tmp_path)

    with gzip.open(cutlas_dir / "labels.nii.gz", "rb") as f:
        hdr = f.read(348)

    ndims, x, y, z = struct.unpack_from("<4h", hdr, 40)
    assert ndims == 3
    assert x == 5   # LR (was axis 2 of BG annotation)
    assert y == 4   # AP (was axis 0)
    assert z == 3   # DV (was axis 1)

    datatype = struct.unpack_from("<h", hdr, 70)[0]
    assert datatype == 768   # uint32


def test_export_nifti_volume_transposition(tmp_path: Path):
    """Voxel at BG [ap, dv, lr] must appear at NIfTI [lr, ap_flipped, dv_flipped]."""
    # (AP=4, DV=3, LR=5) — place a sentinel value at a known position
    ann = np.zeros((4, 3, 5), dtype=np.uint32)
    ann[1, 2, 4] = 999   # BG ap=1, dv=2, lr=4

    cutlas_dir = _run_export(ann, tmp_path)

    # Read back raw voxel data. NIfTI stores the first axis (x = LR) fastest, so
    # the on-disk bytes are Fortran-ordered for our (LR, AP, DV) volume.
    with gzip.open(cutlas_dir / "labels.nii.gz", "rb") as f:
        raw = f.read()
    vox_data = np.frombuffer(raw[352:], dtype=np.uint32).reshape(5, 4, 3, order="F")

    # After flip+transpose:
    #   NIfTI[lr, ap_qn, dv_qn] = BG[AP_max-1-ap_qn, DV_max-1-dv_qn, lr]
    #   sentinel at BG ap=1, dv=2, lr=4:
    #     ap_qn = AP_max - 1 - ap_bg = 4 - 1 - 1 = 2
    #     dv_qn = DV_max - 1 - dv_bg = 3 - 1 - 2 = 0
    #     lr_qn = lr_bg = 4  (LR not flipped)
    assert vox_data[4, 2, 0] == 999


def test_export_labels_txt_format(tmp_path: Path):
    """labels.txt must be ITK-SNAP format with correct region entries."""
    ann = np.zeros((4, 3, 5), dtype=np.uint32)
    cutlas_dir = _run_export(ann, tmp_path)

    text = (cutlas_dir / "labels.txt").read_text(encoding="utf-8")
    assert "ITK-SnAP" in text
    assert '"Clear Label"' in text
    assert '"root"' in text
    assert "255  255  255" in text   # root RGB
