"""Tests for engine/registration.py — the VersoRegistration façade.

These build small projects in memory (via ``VersoRegistration.from_project``) so
they run headless with no brainglobe download: the class only reads the project
data model, and the fake atlas is used solely to produce canonical-plane
anchorings (its helpers read ``annotation.shape`` only).
"""

import numpy as np
import pytest

from verso.engine.anchoring import anchoring_center, make_atlas_sample_grid, normalized_to_atlas
from verso.engine.atlas import AtlasVolume
from verso.engine.model.alignment import Alignment, ControlPoint, WarpState
from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section
from verso.engine.registration import VersoRegistration

# Fake atlas dimensions (AP, DV, LR) and resolution.
_AP, _DV, _LR = 20, 16, 24
_RES_UM = 25.0


def _fake_atlas() -> AtlasVolume:
    atlas = object.__new__(AtlasVolume)
    atlas._annotation = np.zeros((_AP, _DV, _LR), dtype=np.int32)
    return atlas


def _anchoring(position: float, axis: int = 1) -> list[float]:
    return _fake_atlas().canonical_plane_anchoring(position, axis)


def _section(
    sid: str,
    position: float,
    *,
    axis: int = 1,
    flip_h: bool = False,
    flip_v: bool = False,
    cps: list[ControlPoint] | None = None,
    work: tuple[int, int] = (48, 32),
    full: tuple[int, int] = (96, 64),
    original_path: str | None = None,
) -> Section:
    return Section(
        id=sid,
        slice_index=int(position),
        original_path=original_path or f"{sid}.tif",
        thumbnail_path=f"{sid}.ome.tif",
        resolution_original_wh=full,
        resolution_thumbnail_wh=work,
        preprocessing=Preprocessing(flip_horizontal=flip_h, flip_vertical=flip_v),
        alignment=Alignment(anchoring=_anchoring(position, axis)),
        warp=WarpState(control_points=cps or []),
    )


def _project(sections: list[Section]) -> Project:
    return Project(
        name="t",
        atlas=AtlasRef(name="fake", resolution_um=_RES_UM, shape=(_AP, _DV, _LR)),
        sections=sections,
        working_scale=0.5,
    )


def _reg(sections: list[Section]) -> VersoRegistration:
    return VersoRegistration.from_project(_project(sections))


# --- round trip --------------------------------------------------------------


def test_roundtrip_no_cps():
    reg = _reg([_section("s1", 10.0)])
    p = np.array([[30.0, 20.0], [70.0, 50.0]])
    xyz = reg.coord_image_to_atlas("s1", p)
    assert xyz.shape == (2, 3)
    res = reg.coord_atlas_to_image(xyz)
    assert list(res.section_id) == ["s1", "s1"]
    np.testing.assert_allclose(res.distance, 0.0, atol=1e-6)
    np.testing.assert_allclose(res.xy, p, atol=1e-6)
    assert res.valid.all()


def test_roundtrip_with_flips():
    reg = _reg([_section("s1", 10.0, flip_h=True, flip_v=True)])
    p = np.array([[30.0, 20.0]])
    xyz = reg.coord_image_to_atlas("s1", p)
    res = reg.coord_atlas_to_image(xyz)
    assert res.section_id[0] == "s1"
    np.testing.assert_allclose(res.xy, p, atol=1e-6)


def test_roundtrip_working_space():
    reg = _reg([_section("s1", 10.0)])
    p = np.array([[12.0, 8.0]])
    xyz = reg.coord_image_to_atlas("s1", p, space="working")
    res = reg.coord_atlas_to_image(xyz, space="working")
    np.testing.assert_allclose(res.xy, p, atol=1e-6)


def test_roundtrip_with_control_points():
    # At control-point dst locations the warp inverts exactly; full == work here
    # so full-res pixels equal working pixels.
    cps = [
        ControlPoint(src_x=10.0, src_y=8.0, dst_x=14.0, dst_y=6.0),
        ControlPoint(src_x=30.0, src_y=20.0, dst_x=26.0, dst_y=24.0),
        ControlPoint(src_x=40.0, src_y=12.0, dst_x=38.0, dst_y=16.0),
    ]
    reg = _reg([_section("s1", 10.0, cps=cps, work=(48, 32), full=(48, 32))])
    p = np.array([[cp.dst_x, cp.dst_y] for cp in cps])

    xyz = reg.coord_image_to_atlas("s1", p)
    # Forward maps each dst control point to its src atlas voxel.
    anch = _anchoring(10.0)
    expected = np.array([normalized_to_atlas(cp.src_x / 48, cp.src_y / 32, anch) for cp in cps])
    np.testing.assert_allclose(xyz, expected, atol=1e-6)

    res = reg.coord_atlas_to_image(xyz)
    assert list(res.section_id) == ["s1", "s1", "s1"]
    np.testing.assert_allclose(res.xy, p, atol=1e-6)


# --- nearest-section search --------------------------------------------------


def test_nearest_section_picks_closer_plane():
    reg = _reg([_section("s1", 8.0), _section("s2", 14.0)])
    # Voxel order is QuickNII (LR, AP, DV); AP=9 is 1 from s1, 5 from s2.
    res = reg.coord_atlas_to_image(np.array([[12.0, 9.0, 8.0]]))
    assert res.section_id[0] == "s1"
    np.testing.assert_allclose(res.distance[0], 1.0, atol=1e-6)
    assert res.valid[0]

    # AP=12 is 4 from s1, 2 from s2 → s2.
    res2 = reg.coord_atlas_to_image(np.array([[12.0, 12.0, 8.0]]))
    assert res2.section_id[0] == "s2"
    np.testing.assert_allclose(res2.distance[0], 2.0, atol=1e-6)


def test_voxel_outside_all_footprints_is_invalid():
    reg = _reg([_section("s1", 8.0)])
    # LR=100 → s = 100/24 > 1 → outside the section frame.
    res = reg.coord_atlas_to_image(np.array([[100.0, 8.0, 8.0]]))
    assert res.section_id[0] == ""
    assert not res.valid[0]
    assert not np.isfinite(res.distance[0])
    assert np.isnan(res.xy[0]).all()


def test_max_distance_and_distance_units():
    reg = _reg([_section("s1", 8.0)])
    v = np.array([[12.0, 11.0, 8.0]])  # 3 voxels off the plane

    res = reg.coord_atlas_to_image(v, max_distance=2.0)  # voxels
    assert res.section_id[0] == "s1"  # still matched…
    assert not res.valid[0]  # …but beyond the cutoff
    assert reg.coord_atlas_to_image(v, max_distance=5.0).valid[0]

    res_um = reg.coord_atlas_to_image(v, units="um")
    np.testing.assert_allclose(res_um.distance[0], 3.0 * _RES_UM, atol=1e-6)
    assert reg.coord_atlas_to_image(v, units="um", max_distance=3.0 * _RES_UM + 1).valid[0]
    assert not reg.coord_atlas_to_image(v, units="um", max_distance=3.0 * _RES_UM - 1).valid[0]


# --- export-path parity ------------------------------------------------------


def test_export_parity_with_build_canonical_remap():
    from verso.engine.io.export_stack import build_canonical_remap

    atlas = _fake_atlas()
    cps = [
        ControlPoint(src_x=10.0, src_y=8.0, dst_x=14.0, dst_y=6.0),
        ControlPoint(src_x=30.0, src_y=20.0, dst_x=26.0, dst_y=24.0),
    ]
    sec = _section("s1", 10.0, cps=cps, work=(48, 32), full=(48, 32))
    reg = _reg([sec])

    map_x, map_y, out_w, out_h = build_canonical_remap(
        sec, atlas, axis=1, scale=1.0, work_w=48, work_h=32
    )
    position = float(anchoring_center(sec.alignment.anchoring)[1])
    canonical = atlas.canonical_plane_anchoring(position, 1)
    grid = make_atlas_sample_grid(canonical, out_w, out_h)  # (H, W, 3)

    res = reg.coord_atlas_to_image(grid.reshape(-1, 3), space="working")
    mx = res.xy[:, 0].reshape(out_h, out_w)
    my = res.xy[:, 1].reshape(out_h, out_w)

    covered = map_x >= 0
    np.testing.assert_allclose(mx[covered], map_x[covered], atol=1e-4)
    np.testing.assert_allclose(my[covered], map_y[covered], atol=1e-4)


# --- units, validity, resolver, errors ---------------------------------------


def test_units_forward():
    reg = _reg([_section("s1", 10.0)])
    p = np.array([[30.0, 20.0]])
    vox = reg.coord_image_to_atlas("s1", p)
    np.testing.assert_allclose(reg.coord_image_to_atlas("s1", p, units="um"), vox * _RES_UM)
    np.testing.assert_allclose(
        reg.coord_image_to_atlas("s1", p, units="mm"), vox * _RES_UM / 1000.0
    )


def test_return_valid_flags_out_of_frame():
    reg = _reg([_section("s1", 10.0)])  # full == (96, 64)
    p = np.array([[30.0, 20.0], [999.0, 20.0], [-5.0, 10.0]])
    coords, inside = reg.coord_image_to_atlas("s1", p, return_valid=True)
    assert coords.shape == (3, 3)
    assert list(inside) == [True, False, False]


def test_slice_resolver_by_id_stem_and_basename():
    reg = _reg(
        [
            _section("s1", 8.0, original_path="/data/IMG_1.tif"),
            _section("s2", 14.0, original_path="/data/IMG_2.tif"),
        ]
    )
    assert reg._resolve_slice("s1") == "s1"
    assert reg._resolve_slice("IMG_1") == "s1"  # file stem
    assert reg._resolve_slice("IMG_2.tif") == "s2"  # basename
    assert "IMG_1" in reg
    assert "nope" not in reg
    with pytest.raises(KeyError):
        reg._resolve_slice("missing")


def test_slice_resolver_ambiguous_raises():
    reg = _reg(
        [
            _section("s1", 8.0, original_path="/a/IMG.tif"),
            _section("s2", 14.0, original_path="/b/IMG.tif"),
        ]
    )
    with pytest.raises(KeyError):
        reg._resolve_slice("IMG")


def test_ids_and_len():
    reg = _reg([_section("s1", 8.0), _section("s2", 14.0)])
    assert reg.ids() == ["s1", "s2"]
    assert len(reg) == 2


def test_unaligned_section_raises():
    sec = _section("s1", 10.0)
    sec.alignment = Alignment()  # zero anchoring → degenerate plane
    reg = VersoRegistration.from_project(_project([sec]))
    with pytest.raises(ValueError):
        reg.coord_image_to_atlas("s1", [[10.0, 10.0]])


def test_incomplete_dimensions_raise():
    sec = _section("s1", 10.0)
    sec.resolution_thumbnail_wh = (0, 0)
    with pytest.raises(ValueError):
        VersoRegistration.from_project(_project([sec]))


def test_bad_space_and_units_raise():
    reg = _reg([_section("s1", 10.0)])
    with pytest.raises(ValueError):
        reg.coord_image_to_atlas("s1", [[1.0, 1.0]], space="nope")
    with pytest.raises(ValueError):
        reg.coord_image_to_atlas("s1", [[1.0, 1.0]], units="parsecs")


# --- whole-image atlas resampling (image_to_atlas) ----------------------------


def _fake_atlas_full() -> AtlasVolume:
    """Fake atlas with annotation, reference, and color dict populated.

    Two regions split along the LR (x) axis at the midpoint, so warping and
    flipping visibly change which region a given pixel samples.
    """
    atlas = object.__new__(AtlasVolume)
    ann = np.ones((_AP, _DV, _LR), dtype=np.int32)
    ann[:, :, _LR // 2 :] = 2
    atlas._annotation = ann
    atlas._reference = np.arange(_AP * _DV * _LR, dtype=np.float64).reshape(_AP, _DV, _LR)
    ref_max = float(atlas._reference.max())
    atlas._reference_scale = 255.0 / ref_max if ref_max > 0 else 1.0
    atlas._color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}
    return atlas


def _reg_with_fake_volume(sections: list[Section]) -> VersoRegistration:
    reg = _reg(sections)
    reg._atlas_volume = _fake_atlas_full()
    return reg


def test_image_to_atlas_annotation_matches_pointwise_lookup():
    sec = _section("s1", 10.0, full=(96, 64), work=(48, 32))
    reg = _reg_with_fake_volume([sec])

    labels = reg.image_to_atlas("s1", kind="annotation")
    assert labels.shape == (64, 96)
    assert labels.dtype == np.int32

    # Spot-check a handful of full-res pixels (queried at their pixel *centers*,
    # matching image_to_atlas's own sampling convention) against the pointwise
    # coordinate mapping + the same nearest-voxel (floor/ceil) convention.
    from verso.engine.atlas import _quicknii_floor_indices

    cols = np.array([5, 50, 90])
    rows = np.array([5, 10, 60])
    pts = np.column_stack([cols + 0.5, rows + 0.5])
    xyz = reg.coord_image_to_atlas("s1", pts)
    lr_f, ap_f, dv_f = _quicknii_floor_indices(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ap_max, dv_max, lr_max = (_AP, _DV, _LR)
    inside = (
        (ap_f >= 0)
        & (ap_f < ap_max)
        & (dv_f >= 0)
        & (dv_f < dv_max)
        & (lr_f >= 0)
        & (lr_f < lr_max)
    )
    expected = np.where(inside, np.where(np.clip(lr_f, 0, lr_max - 1) >= _LR // 2, 2, 1), 0)
    np.testing.assert_array_equal(labels[rows, cols], expected)


def test_image_to_atlas_annotation_with_warp():
    cps = [
        ControlPoint(src_x=10.0, src_y=8.0, dst_x=20.0, dst_y=8.0),
        ControlPoint(src_x=30.0, src_y=20.0, dst_x=20.0, dst_y=20.0),
        ControlPoint(src_x=10.0, src_y=25.0, dst_x=20.0, dst_y=25.0),
    ]
    sec = _section("s1", 10.0, cps=cps, work=(48, 32), full=(48, 32))
    reg = _reg_with_fake_volume([sec])

    labels_warped = reg.image_to_atlas("s1", kind="annotation")
    reg_flat = _reg_with_fake_volume([_section("s1", 10.0, work=(48, 32), full=(48, 32))])
    labels_unwarped = reg_flat.image_to_atlas("s1", kind="annotation")

    assert labels_warped.shape == labels_unwarped.shape
    assert not np.array_equal(labels_warped, labels_unwarped)


def test_image_to_atlas_template():
    sec = _section("s1", 10.0, full=(96, 64), work=(48, 32))
    reg = _reg_with_fake_volume([sec])

    gray, in_bounds = reg.image_to_atlas("s1", kind="template", return_valid=True)
    assert gray.shape == (64, 96)
    assert gray.dtype == np.uint8
    assert in_bounds.shape == (64, 96)
    assert np.all(gray[~in_bounds] == 0)


def test_image_to_atlas_boundary():
    sec = _section("s1", 10.0, full=(96, 64), work=(48, 32))
    reg = _reg_with_fake_volume([sec])

    boundary = reg.image_to_atlas("s1", kind="boundary")
    labels = reg.image_to_atlas("s1", kind="annotation")

    assert boundary.dtype == np.bool_
    assert boundary.shape == labels.shape
    assert boundary.any()  # the two-region fake atlas has a real boundary

    from verso.engine.atlas import boundary_mask

    expected = boundary_mask(labels, np.ones_like(labels, dtype=bool))
    np.testing.assert_array_equal(boundary, expected)


def test_image_to_atlas_working_space():
    sec = _section("s1", 10.0, full=(96, 64), work=(48, 32))
    reg = _reg_with_fake_volume([sec])

    labels = reg.image_to_atlas("s1", kind="annotation", space="working")
    assert labels.shape == (32, 48)


def test_image_to_atlas_bad_kind_raises():
    reg = _reg_with_fake_volume([_section("s1", 10.0)])
    with pytest.raises(ValueError):
        reg.image_to_atlas("s1", kind="nope")
