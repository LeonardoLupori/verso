"""Tests for engine/model — JSON serialisation round-trips."""

import json
from pathlib import Path

from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import AtlasRef, ChannelSpec, Preprocessing, Project, Section

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
    assert a.current_anchoring == [0.0] * 9
    assert a.status == AlignmentStatus.NOT_STARTED


def test_alignment_round_trip():
    # Only the saved plane persists (as "anchoring"); the live copy is seeded
    # from it on load and position_mm is derived, so a round-trip is faithful
    # only when current_anchoring == stored_anchoring (i.e. a committed plane).
    a = Alignment(
        current_anchoring=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        status=AlignmentStatus.COMPLETE,
        source="deepslice",
        stored_anchoring=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    assert Alignment.from_dict(a.to_dict()) == a


def test_from_dict_without_anchoring_key_has_no_plane():
    a = Alignment.from_dict({"status": "in_progress"})
    assert a.current_anchoring == [0.0] * 9
    assert a.stored_anchoring is None
    assert a.position_mm is None
    assert a.source is None


def test_position_mm_is_not_persisted():
    a = Alignment(position_mm=-1.5, status=AlignmentStatus.IN_PROGRESS)
    assert "position_mm" not in a.to_dict()
    assert Alignment.from_dict(a.to_dict()).position_mm is None


def test_from_dict_anchoring_key_seeds_stored_and_live():
    a = Alignment.from_dict(
        {
            "anchoring": [1.0] * 9,
            "status": "complete",
        }
    )
    assert a.stored_anchoring == [1.0] * 9
    assert a.current_anchoring == [1.0] * 9


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
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocessing_round_trip():
    pp = Preprocessing(
        flip_horizontal=True,
        flip_vertical=False,
        slice_mask_path="masks/s001-slice-mask.png",
    )
    out = Preprocessing.from_dict(pp.to_dict())
    assert out == pp


def test_preprocessing_ignores_legacy_lr_keys():
    """Legacy projects may carry lr_mask_path/lr_line; loading drops them."""
    pp = Preprocessing.from_dict(
        {
            "flip_horizontal": True,
            "flip_vertical": False,
            "slice_mask_path": "masks/s001-slice-mask.png",
            "lr_mask_path": "lr_masks/s001_lr.png",
            "lr_line": [[1.0, 2.0], [3.0, 4.0]],
        }
    )
    assert pp.flip_horizontal is True
    assert pp.slice_mask_path == "masks/s001-slice-mask.png"
    assert "lr_mask_path" not in pp.to_dict()
    assert "lr_line" not in pp.to_dict()


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


def _make_section() -> Section:
    return Section(
        id="s001",
        slice_index=1,
        original_path="/data/raw/IMG_0234.tif",
        thumbnail_path="thumbnails/s001.ome.tif",
        preprocessing=Preprocessing(flip_horizontal=True),
        alignment=Alignment(
            current_anchoring=[0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
            status=AlignmentStatus.COMPLETE,
        ),
        warp=WarpState(
            control_points=[ControlPoint(10.0, 20.0, 12.0, 19.0)],
            status=AlignmentStatus.IN_PROGRESS,
        ),
    )


def test_section_round_trip():
    s = _make_section()
    assert Section.from_dict(s.to_dict()) == s


def test_section_dims_default_to_zero():
    s = Section(id="s001", slice_index=1, original_path="a.tif", thumbnail_path="t.ome.tif")
    assert s.resolution_original_wh == (0, 0)
    assert s.resolution_thumbnail_wh == (0, 0)


def test_section_round_trip_preserves_dims():
    s = _make_section()
    s.resolution_original_wh = (2000, 1500)
    s.resolution_thumbnail_wh = (400, 300)
    loaded = Section.from_dict(s.to_dict())
    assert loaded.resolution_original_wh == (2000, 1500)
    assert loaded.resolution_thumbnail_wh == (400, 300)
    assert loaded == s


def test_atlas_ref_defaults_and_round_trip():
    ref = AtlasRef(name="allen_mouse_25um")
    assert ref.resolution_um == 0.0
    assert ref.shape == (0, 0, 0)

    ref = AtlasRef(name="allen_mouse_25um", resolution_um=25.0, shape=(528, 320, 456))
    loaded = AtlasRef.from_dict(ref.to_dict())
    assert loaded.resolution_um == 25.0
    assert loaded.shape == (528, 320, 456)
    assert loaded == ref


# ---------------------------------------------------------------------------
# Project — save / load from disk
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    return Project(
        name="My Experiment",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[_make_section()],
        channels=[
            ChannelSpec(name="DAPI", color=(0, 100, 255), scale=0.8, visible=True),
            ChannelSpec(name="GFP", color=(0, 255, 0), scale=1.0, visible=True),
        ],
        working_scale=0.06,
    )


# ---------------------------------------------------------------------------
# ChannelSpec
# ---------------------------------------------------------------------------


def test_channel_spec_round_trip():
    c = ChannelSpec(name="DAPI", color=(0, 100, 255), scale=0.6, visible=False)
    assert ChannelSpec.from_dict(c.to_dict()) == c


def test_channel_spec_defaults():
    c = ChannelSpec(name="GFP")
    assert c.color == (255, 255, 255)
    assert c.scale == 1.0
    assert c.visible is True


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
    assert data["interpolation_axis"] == "AP"
    assert len(data["sections"]) == 1
    assert data["sections"][0]["id"] == "s001"


def test_project_default_interpolation_axis_is_ap():
    p = _make_project()
    assert p.interpolation_axis == "AP"
    assert p.interpolation_axis_index == 1


def test_project_round_trips_non_coronal_axis():
    p = _make_project()
    p.interpolation_axis = "ML"
    loaded = Project.from_dict(p.to_dict())
    assert loaded.interpolation_axis == "ML"
    assert loaded.interpolation_axis_index == 0


def test_project_legacy_dict_loads_with_default_axis_and_preserves_version():
    p = _make_project()
    legacy = p.to_dict()
    legacy.pop("interpolation_axis")
    legacy["version"] = "0.9"

    # from_dict tolerates a missing interpolation_axis (defaults to AP) and
    # preserves whatever version string the file carried.
    loaded = Project.from_dict(legacy)
    assert loaded.interpolation_axis == "AP"
    assert loaded.version == "0.9"


def test_project_invalid_axis_falls_back_to_ap():
    p = _make_project()
    legacy = p.to_dict()
    legacy["interpolation_axis"] = "nonsense"

    loaded = Project.from_dict(legacy)
    assert loaded.interpolation_axis == "AP"
