"""Tests for engine/model — JSON serialisation round-trips."""

import json
from pathlib import Path

from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.mask import Mask, MaskType
from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section

# ---------------------------------------------------------------------------
# ControlPoint
# ---------------------------------------------------------------------------

def test_control_point_round_trip():
    cp = ControlPoint(src_x=10.0, src_y=20.0, dst_x=15.0, dst_y=18.0)
    assert ControlPoint.from_dict(cp.to_dict()) == cp


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_alignment_defaults():
    a = Alignment()
    assert a.anchoring == [0.0] * 9
    assert a.status == AlignmentStatus.NOT_STARTED


def test_alignment_round_trip():
    a = Alignment(
        anchoring=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ap_position_mm=-1.5,
        status=AlignmentStatus.COMPLETE,
        source="deepslice",
        proposal_anchoring=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        proposal_confidence=0.87,
        proposal_run_id="run-1",
    )
    assert Alignment.from_dict(a.to_dict()) == a


def test_alignment_loads_legacy_dict_without_metadata():
    a = Alignment.from_dict({
        "anchoring": [1.0] * 9,
        "ap_position_mm": 2.0,
        "status": "in_progress",
    })
    assert a.source is None
    assert a.proposal_anchoring is None
    assert a.proposal_confidence is None
    assert a.proposal_run_id is None


# ---------------------------------------------------------------------------
# WarpState
# ---------------------------------------------------------------------------

def test_warp_state_round_trip():
    cps = [
        ControlPoint(10.0, 20.0, 12.0, 19.0),
        ControlPoint(50.0, 60.0, 52.0, 58.0),
    ]
    ws = WarpState(control_points=cps, status=AlignmentStatus.IN_PROGRESS)
    assert WarpState.from_dict(ws.to_dict()) == ws


def test_warp_state_empty():
    ws = WarpState()
    assert WarpState.from_dict(ws.to_dict()) == ws


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def test_mask_round_trip():
    m = Mask(path="masks/s001_slice.png", mask_type=MaskType.SLICE)
    assert Mask.from_dict(m.to_dict()) == m


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

def _make_section() -> Section:
    return Section(
        id="s001",
        serial_number=1,
        original_path="/data/raw/IMG_0234.tif",
        thumbnail_path="thumbnails/s001.png",
        channels=["DAPI", "GFP"],
        registration_channel="GFP",
        preprocessing=Preprocessing(flip_horizontal=True),
        alignment=Alignment(
            anchoring=[0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
            status=AlignmentStatus.COMPLETE,
        ),
        warp=WarpState(
            control_points=[ControlPoint(10.0, 20.0, 12.0, 19.0)],
            status=AlignmentStatus.IN_PROGRESS,
        ),
        scale=0.06,
    )


def test_section_round_trip():
    s = _make_section()
    assert Section.from_dict(s.to_dict()) == s


# ---------------------------------------------------------------------------
# Project — save / load from disk
# ---------------------------------------------------------------------------

def _make_project() -> Project:
    return Project(
        name="My Experiment",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[_make_section()],
    )


def test_project_round_trip_in_memory():
    p = _make_project()
    assert Project.from_dict(p.to_dict()) == p


def test_project_save_load_roundtrip(tmp_path: Path):
    p = _make_project()
    json_path = tmp_path / "project.json"
    p.save(json_path)

    loaded = Project.load(json_path)
    assert loaded == p


def test_project_json_is_valid(tmp_path: Path):
    p = _make_project()
    json_path = tmp_path / "project.json"
    p.save(json_path)

    data = json.loads(json_path.read_text())
    assert data["version"] == "1.0"
    assert len(data["sections"]) == 1
    assert data["sections"][0]["id"] == "s001"
