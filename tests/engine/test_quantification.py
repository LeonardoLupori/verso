"""Tests for the quantification package (headless, no brainglobe download).

A fake :class:`AtlasVolume` (small in-memory annotation volume + a tiny structure
tree) is injected, mirroring ``test_registration.py``, so the whole pipeline runs
without downloading an atlas.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from verso.engine.atlas import AtlasVolume
from verso.engine.io.annotation_io import save_annotations
from verso.engine.model.alignment import Alignment, ControlPoint, WarpState
from verso.engine.model.annotation import AnnotationPoint, PointSeries
from verso.engine.model.project import AtlasRef, ChannelSpec, Preprocessing, Project, Section
from verso.engine.preprocessing import save_mask
from verso.engine.quantification import (
    QuantificationError,
    QuantifyOptions,
    quantify_area,
    quantify_dots,
    quantify_intensity,
)
from verso.engine.quantification.aggregate import RegionAggregator, regroup_intensity
from verso.engine.quantification.intensity import IntensityAccumulator, match_to_raw
from verso.engine.quantification.region_map import region_map
from verso.engine.quantification.tables import channel_column

_AP, _DV, _LR = 8, 8, 8
_RES_UM = 25.0

# ---------------------------------------------------------------------------
# Fake atlas
# ---------------------------------------------------------------------------

# A tiny structure tree: root -> grey -> Isocortex -> VISp -> VISp1, plus a
# fiber-tract branch. structure_id_path is root-first, self last (brainglobe order).
_STRUCTURES = {
    997: {"acronym": "root", "name": "root", "structure_id_path": [997]},
    8: {"acronym": "grey", "name": "Basic cell groups", "structure_id_path": [997, 8]},
    315: {"acronym": "Isocortex", "name": "Isocortex", "structure_id_path": [997, 8, 315]},
    385: {
        "acronym": "VISp",
        "name": "Primary visual area",
        "structure_id_path": [997, 8, 315, 385],
    },
    33385: {
        "acronym": "VISp1",
        "name": "VISp layer 1",
        "structure_id_path": [997, 8, 315, 385, 33385],
    },
    1009: {"acronym": "fiber tracts", "name": "fiber tracts", "structure_id_path": [997, 1009]},
    1000: {"acronym": "cc", "name": "corpus callosum", "structure_id_path": [997, 1009, 1000]},
    # Two simple leaf regions used to paint the fake annotation volume.
    10: {"acronym": "L", "name": "left region", "structure_id_path": [997, 8, 315, 385, 10]},
    20: {"acronym": "R", "name": "right region", "structure_id_path": [997, 1009, 20]},
}


def _fake_atlas() -> AtlasVolume:
    atlas = object.__new__(AtlasVolume)
    ann = np.zeros((_AP, _DV, _LR), dtype=np.int32)
    # Paint a coronal-ish partition: left LR half = 10, right half = 20, a strip = 0.
    ann[:, :, :4] = 10
    ann[:, :, 4:] = 20
    ann[:, 0, :] = 0  # a background strip (spans all LR → straddles both hemispheres)
    atlas._annotation = ann
    # Hemisphere volume, matching the real brainglobe Allen "asr" orientation: the
    # LOW-LR half (index < 4) is brainglobe's RIGHT value (2), the HIGH-LR half is
    # its LEFT value (1) — i.e. brainglobe's raw values run opposite to VERSO's
    # display/QuickNII axis (see issue #40 and AtlasVolume.hemisphere_label). So the
    # low-LR region 10 sits on brainglobe-right (reported user-facing as 'l', the
    # screen-left side) and the high-LR region 20 on brainglobe-left (user 'r').
    hemi_row = np.where(np.arange(_LR) < _LR // 2, 2, 1).astype(np.uint8)
    atlas._hemispheres = np.broadcast_to(hemi_row, (_AP, _DV, _LR)).copy()
    atlas._left_val = 1
    atlas._right_val = 2
    atlas._bg = SimpleNamespace(structures=_STRUCTURES)
    return atlas


def _anchoring(position: float) -> list[float]:
    return _fake_atlas().canonical_plane_anchoring(position, axis=1)


# ---------------------------------------------------------------------------
# Project builder (writes real images + masks to tmp)
# ---------------------------------------------------------------------------


def _make_project(
    tmp_path: Path,
    *,
    n_sections: int = 2,
    with_mask: bool = True,
    with_cps: bool = True,
    n_channels: int = 2,
    full_wh: tuple[int, int] = (16, 12),
    work_wh: tuple[int, int] = (8, 6),
    seed: int = 0,
) -> tuple[Project, Path, AtlasVolume]:
    rng = np.random.default_rng(seed)
    full_w, full_h = full_wh
    work_w, work_h = work_wh
    hi = tmp_path / "hiRes"
    masks = tmp_path / "masks"
    hi.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)

    sections: list[Section] = []
    for i in range(n_sections):
        stem = f"s{i + 1}"
        orig = hi / f"{stem}.tif"
        # (C, H, W) uint16 — channels-first, load_full_res_raw normalises to (H, W, C).
        arr = rng.integers(0, 500, size=(n_channels, full_h, full_w), dtype=np.uint16)
        tifffile.imwrite(str(orig), arr)

        mask_path = None
        if with_mask:
            mask = np.zeros((work_h, work_w), dtype=bool)
            mask[:, : work_w // 2 + 1] = True  # left-ish half foreground
            mask_path = masks / f"{stem}-slice-mask.png"
            save_mask(mask, mask_path)

        cps = [ControlPoint(src_x=2.0, src_y=2.0, dst_x=2.2, dst_y=1.8)] if with_cps else []
        sections.append(
            Section(
                id=stem,
                slice_index=i + 1,
                original_path=str(orig),
                thumbnail_path=str(tmp_path / "thumbnails" / f"{stem}-thumb.ome.tif"),
                resolution_original_wh=(full_w, full_h),
                resolution_thumbnail_wh=(work_w, work_h),
                preprocessing=Preprocessing(slice_mask_path=str(mask_path) if mask_path else None),
                alignment=Alignment(current_anchoring=_anchoring(3.0 + i)),
                warp=WarpState(control_points=cps),
            )
        )

    project = Project(
        name="t",
        atlas=AtlasRef(name="fake", resolution_um=_RES_UM, shape=(_AP, _DV, _LR)),
        sections=sections,
        channels=[ChannelSpec(name=f"C{c}") for c in range(n_channels)],
        working_scale=work_w / full_w,
    )
    return project, tmp_path, _fake_atlas()


# ---------------------------------------------------------------------------
# Unit: intensity accumulator
# ---------------------------------------------------------------------------


def test_accumulator_exact_mean_and_tot():
    labels = np.array([[1, 1, 2], [2, 0, 2]], dtype=np.int32)
    scope = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)  # excludes the (1,1)=0 pixel
    raw = np.zeros((2, 3, 1), dtype=np.uint16)
    raw[..., 0] = [[10, 20, 5], [7, 999, 3]]  # 999 is out of scope

    acc = IntensityAccumulator(1)
    acc.add(labels, scope, raw)

    assert acc.counts == {(1, None): 2, (2, None): 3}
    tot = acc.totals_as_lists()
    assert tot[(1, None)] == [30.0]  # 10 + 20
    assert tot[(2, None)] == [5.0 + 7.0 + 3.0]  # 15
    # out-of-scope 999 never contributes
    assert (0, None) not in acc.counts


def test_accumulator_pools_across_sections():
    labels = np.array([[1, 2]], dtype=np.int32)
    scope = np.ones((1, 2), dtype=bool)
    raw = np.zeros((1, 2, 2), dtype=np.uint16)
    raw[0, 0] = [4, 40]
    raw[0, 1] = [6, 60]
    acc = IntensityAccumulator(2)
    acc.add(labels, scope, raw)
    acc.add(labels, scope, raw)
    assert acc.counts == {(1, None): 2, (2, None): 2}
    assert acc.totals_as_lists()[(1, None)] == [8.0, 80.0]


def test_accumulator_hemisphere_split():
    labels = np.array([[1, 1, 2, 2]], dtype=np.int32)
    scope = np.ones((1, 4), dtype=bool)
    hemi = np.array([[1, 2, 1, 2]], dtype=np.uint8)  # region 1 and 2 each straddle
    raw = np.zeros((1, 4, 1), dtype=np.uint16)
    raw[..., 0] = [[10, 20, 30, 40]]

    acc = IntensityAccumulator(1)
    acc.add(labels, scope, raw, hemi)
    # Each region splits into a left (hemi 1) and right (hemi 2) bucket.
    assert acc.counts == {(1, 1): 1, (1, 2): 1, (2, 1): 1, (2, 2): 1}
    tot = acc.totals_as_lists()
    assert tot[(1, 1)] == [10.0]
    assert tot[(1, 2)] == [20.0]
    # Splitting conserves totals: the two hemispheres of region 1 sum to the whole.
    unsplit = IntensityAccumulator(1)
    unsplit.add(labels, scope, raw)
    assert tot[(1, 1)][0] + tot[(1, 2)][0] == unsplit.totals_as_lists()[(1, None)][0]


def test_match_to_raw_resizes_labels_nearest():
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    scope = np.ones((2, 2), dtype=bool)
    hemi = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    lab2, sc2, hemi2 = match_to_raw(labels, scope, (4, 4), hemi)
    assert lab2.shape == (4, 4)
    assert sc2.shape == (4, 4)
    assert hemi2.shape == (4, 4)
    assert set(np.unique(lab2)).issubset({1, 2, 3, 4})  # nearest, no interpolation
    assert set(np.unique(hemi2)).issubset({1, 2})
    # hemi stays None when not splitting
    _, _, none_hemi = match_to_raw(labels, scope, (4, 4), None)
    assert none_hemi is None


def test_channel_column_naming():
    assert channel_column("mean", "DAPI") == "mean_ch_DAPI"
    assert channel_column("tot", "Ch 0") == "tot_ch_Ch_0"


# ---------------------------------------------------------------------------
# Unit: aggregation
# ---------------------------------------------------------------------------


def _agg(members_mid, members_coarse) -> RegionAggregator:
    atlas = _fake_atlas()
    return RegionAggregator(
        atlas,
        {"mid": {"members": list(members_mid)}, "coarse": {"members": list(members_coarse)}},
    )


def test_representative_is_nearest_ancestor():
    agg = _agg([385], [315])
    assert agg.representative("mid", 33385) == 385  # VISp1 -> VISp
    assert agg.representative("coarse", 33385) == 315  # VISp1 -> Isocortex
    assert agg.representative("mid", 385) == 385  # self is a member


def test_representative_order_independent():
    a1 = _agg([385, 315, 8], [315])
    a2 = _agg([8, 315, 385], [315])  # shuffled
    # nearest member ancestor of VISp1 among {8,315,385} is 385 regardless of order
    assert a1.representative("mid", 33385) == 385
    assert a2.representative("mid", 33385) == 385


def test_representative_unassigned_and_fiber_tracts():
    agg = _agg([385, 1009], [315, 1009])
    assert agg.representative("mid", 8) is None  # no member ancestor -> unassigned
    assert agg.representative("mid", 1000) == 1009  # fiber-tract subregion -> fiber tracts
    assert agg.representative("coarse", 1000) == 1009
    assert agg.representative("mid", 0) is None  # background


def test_regroup_intensity_sums_and_recomputes_mean():
    agg = _agg([385], [315])
    counts = {(33385, None): 4, (385, None): 6}
    totals = {(33385, None): [40.0], (385, None): [60.0]}
    rows = regroup_intensity(counts, totals, agg, "mid", _fake_atlas(), ["C0"])
    assert len(rows) == 1
    row = rows[0]
    assert row["region_id"] == 385
    assert row["n_pixels"] == 10
    assert row["tot_ch_C0"] == 100.0
    assert row["mean_ch_C0"] == 10.0  # 100 / 10
    assert "hemisphere" not in row  # not splitting


def test_regroup_intensity_preserves_hemisphere():
    agg = _agg([385], [315])
    # VISp1 pixels split across both hemispheres regroup to VISp per hemisphere.
    # brainglobe raw value 1 (its left) -> user-facing 'r'; value 2 -> 'l' (see
    # hemisphere_label / issue #40).
    counts = {(33385, 1): 4, (33385, 2): 6}
    totals = {(33385, 1): [40.0], (33385, 2): [90.0]}
    rows = regroup_intensity(counts, totals, agg, "mid", _fake_atlas(), ["C0"])
    assert len(rows) == 2
    by_hemi = {r["hemisphere"]: r for r in rows}
    assert by_hemi["r"]["region_id"] == 385 and by_hemi["r"]["n_pixels"] == 4
    assert by_hemi["l"]["region_id"] == 385 and by_hemi["l"]["n_pixels"] == 6
    assert by_hemi["l"]["mean_ch_C0"] == pytest.approx(15.0)  # 90 / 6


def test_regroup_unassigned_bucket():
    agg = _agg([385], [315])
    rows = regroup_intensity({(8, None): 5}, {(8, None): [50.0]}, agg, "mid", _fake_atlas(), ["C0"])
    assert rows[0]["region_id"] == ""
    assert rows[0]["acronym"] == "unassigned"


# ---------------------------------------------------------------------------
# region_map
# ---------------------------------------------------------------------------


def test_region_map_scope_from_mask(tmp_path):
    from verso.engine.registration import VersoRegistration

    project, _pdir, atlas = _make_project(tmp_path, n_sections=1, with_mask=True)
    reg = VersoRegistration.from_project(project)
    labels, scope, hemi = region_map(reg, atlas, project.sections[0])
    full_w, full_h = project.sections[0].resolution_original_wh
    assert labels.shape == (full_h, full_w)
    assert scope.shape == (full_h, full_w)
    assert hemi is None  # not requested
    assert 0 < scope.sum() < scope.size  # mask restricts but doesn't erase


def test_region_map_wholeframe_when_no_mask(tmp_path):
    from verso.engine.registration import VersoRegistration

    project, _pdir, atlas = _make_project(tmp_path, n_sections=1, with_mask=False)
    reg = VersoRegistration.from_project(project)
    _labels, scope, _hemi = region_map(reg, atlas, project.sections[0])
    assert scope.all()  # whole frame


def test_region_map_hemisphere_aligned_with_labels(tmp_path):
    from verso.engine.registration import VersoRegistration

    project, _pdir, atlas = _make_project(tmp_path, n_sections=1, with_mask=False)
    reg = VersoRegistration.from_project(project)
    labels, _scope, hemi = region_map(reg, atlas, project.sections[0], split_hemispheres=True)
    assert hemi is not None
    assert hemi.shape == labels.shape
    # In the fake atlas region 10 is the low-LR half (brainglobe raw hemi value 2)
    # and region 20 the high-LR half (raw value 1); assert the per-pixel maps agree
    # where the region is annotated. (These are brainglobe's raw asr values, before
    # the user-facing l/r conversion in hemisphere_label.)
    assert set(np.unique(hemi[labels == 10])).issubset({2})
    assert set(np.unique(hemi[labels == 20])).issubset({1})


# ---------------------------------------------------------------------------
# Integration: intensity
# ---------------------------------------------------------------------------


def test_quantify_intensity_invariants(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    result = quantify_intensity(project, project_dir=pdir, atlas=atlas)
    rows = result["regions"]
    assert rows

    # Independently recompute the pooled scope totals from the same building blocks.
    from verso.engine.io.image_io import load_full_res_raw
    from verso.engine.registration import VersoRegistration

    reg = VersoRegistration.from_project(project)
    total_px = 0
    total_c0 = 0.0
    for s in project.sections:
        _labels, scope, _hemi = region_map(reg, atlas, s)
        raw = load_full_res_raw(s.original_path)
        total_px += int(scope.sum())
        total_c0 += float(raw[..., 0][scope].sum())

    assert sum(r["n_pixels"] for r in rows) == total_px
    assert sum(r["tot_ch_C0"] for r in rows) == pytest.approx(total_c0)
    # mean == tot / n per row
    for r in rows:
        assert r["mean_ch_C0"] == pytest.approx(r["tot_ch_C0"] / r["n_pixels"])
    # columns present for both channels
    assert {"mean_ch_C0", "tot_ch_C0", "mean_ch_C1", "tot_ch_C1"} <= set(rows[0])


def test_quantify_intensity_writes_csv(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    out = tmp_path / "exports"
    quantify_intensity(project, project_dir=pdir, atlas=atlas, options=QuantifyOptions(out_dir=out))
    folders = list(out.glob("quantification_*"))
    assert len(folders) == 1
    assert (folders[0] / "intensity.csv").exists()


def test_per_slice_sums_match_pooled(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    pooled = quantify_intensity(project, project_dir=pdir, atlas=atlas)["regions"]
    per = quantify_intensity(
        project, project_dir=pdir, atlas=atlas, options=QuantifyOptions(per_slice=True)
    )
    assert set(per) == {"s1", "s2"}  # slugified image stems

    pooled_px = {r["region_id"]: r["n_pixels"] for r in pooled}
    summed: dict = {}
    for sub in per.values():
        for r in sub["regions"]:
            summed[r["region_id"]] = summed.get(r["region_id"], 0) + r["n_pixels"]
    assert summed == pooled_px


def test_intensity_channel_selection(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    rows = quantify_intensity(
        project, project_dir=pdir, atlas=atlas, options=QuantifyOptions(channels=["C1"])
    )["regions"]
    assert "mean_ch_C1" in rows[0]
    assert "mean_ch_C0" not in rows[0]


def test_intensity_aggregate_levels(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    res = quantify_intensity(
        project, project_dir=pdir, atlas=atlas, options=QuantifyOptions(aggregate=("mid", "coarse"))
    )
    assert "regions_mid" in res
    assert "regions_coarse" in res


def test_intensity_hemisphere_split_conserves_and_labels(tmp_path):
    project, pdir, atlas = _make_project(tmp_path)
    pooled = quantify_intensity(project, project_dir=pdir, atlas=atlas)["regions"]
    split = quantify_intensity(
        project, project_dir=pdir, atlas=atlas, options=QuantifyOptions(split_hemispheres=True)
    )["regions"]

    # Every split row carries an l/r/none hemisphere label.
    assert all(r["hemisphere"] in {"l", "r", "none"} for r in split)
    assert all("hemisphere" not in r for r in pooled)

    # Splitting conserves pixels/totals per region (sum of hemispheres == pooled).
    def by_region(rows):
        out: dict = {}
        for r in rows:
            out.setdefault(r["region_id"], {"n": 0, "tot": 0.0})
            out[r["region_id"]]["n"] += r["n_pixels"]
            out[r["region_id"]]["tot"] += r["tot_ch_C0"]
        return out

    ps, ss = by_region(pooled), by_region(split)
    assert set(ps) == set(ss)
    for rid in ps:
        assert ss[rid]["n"] == ps[rid]["n"]
        assert ss[rid]["tot"] == pytest.approx(ps[rid]["tot"])

    # Region 10 is annotated only in the left hemisphere, region 20 only in the
    # right — each yields a single hemisphere row (the one-sided-region case).
    hemis = {}
    for r in split:
        hemis.setdefault(r["region_id"], set()).add(r["hemisphere"])
    assert hemis[10] == {"l"}
    assert hemis[20] == {"r"}
    # Out-of-atlas pixels (region 0, edge pixels whose voxel is out of bounds) have
    # no defined hemisphere and pool into the "none" bucket — nothing is dropped.
    if 0 in hemis:
        assert hemis[0] == {"none"}


def _bright_left_project(
    tmp_path: Path, *, flip_horizontal: bool = False, flip_vertical: bool = False
) -> tuple[Project, Path, AtlasVolume]:
    """One aligned section whose on-disk image is bright on the LEFT half.

    The section is anchored to a full-LR coronal plane (``u = +LR``), so an on-disk
    column maps monotonically across the atlas LR axis. Used to check which
    hemisphere the bright side is quantified into under each flip combination.
    """
    atlas = _fake_atlas()
    full_w, full_h = 16, 12
    work_w, work_h = 8, 6
    hi = tmp_path / "hiRes"
    hi.mkdir(exist_ok=True)
    orig = hi / "s1.tif"
    arr = np.zeros((1, full_h, full_w), dtype=np.uint16)  # (C, H, W)
    arr[0, :, : full_w // 2] = 1000  # bright on the on-disk LEFT half
    tifffile.imwrite(str(orig), arr)
    section = Section(
        id="s1",
        slice_index=1,
        original_path=str(orig),
        thumbnail_path=str(tmp_path / "thumbnails" / "s1-thumb.ome.tif"),
        resolution_original_wh=(full_w, full_h),
        resolution_thumbnail_wh=(work_w, work_h),
        preprocessing=Preprocessing(flip_horizontal=flip_horizontal, flip_vertical=flip_vertical),
        alignment=Alignment(current_anchoring=_anchoring(3.0)),
        warp=WarpState(control_points=[]),
    )
    project = Project(
        name="t",
        atlas=AtlasRef(name="fake", resolution_um=_RES_UM, shape=(_AP, _DV, _LR)),
        sections=[section],
        channels=[ChannelSpec(name="C0")],
        working_scale=work_w / full_w,
    )
    return project, tmp_path, atlas


@pytest.mark.parametrize(
    ("flip_horizontal", "flip_vertical", "expected"),
    [
        # Unflipped: bright on-disk-left -> display-left -> atlas low-LR. That side is
        # reported 'l' (VERSO display / QuickNII convention), even though brainglobe's
        # raw hemispheres value there is its "right" (asr). This is issue #40.
        (False, False, "l"),
        # Horizontal flip mirrors LR -> bright moves to the high-LR side -> 'r'.
        (True, False, "r"),
        # Vertical flip is DV only and must NOT change the hemisphere -> still 'l'.
        (False, True, "l"),
        # Both flips: only the horizontal one affects L/R -> 'r'.
        (True, True, "r"),
    ],
)
def test_hemisphere_follows_display_convention_under_flips(
    tmp_path, flip_horizontal, flip_vertical, expected
):
    project, pdir, atlas = _bright_left_project(
        tmp_path, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    rows = quantify_intensity(
        project,
        project_dir=pdir,
        atlas=atlas,
        options=QuantifyOptions(
            split_hemispheres=True,
            include_unmasked_wholeframe=True,
            include_unwarped_affine=True,
        ),
    )["regions"]
    tot_by_hemi: dict[str, float] = {}
    for r in rows:
        tot_by_hemi[r["hemisphere"]] = tot_by_hemi.get(r["hemisphere"], 0.0) + r["tot_ch_C0"]
    assert tot_by_hemi, "no hemisphere rows produced"
    brightest = max(tot_by_hemi, key=lambda h: tot_by_hemi[h])
    assert brightest == expected


# ---------------------------------------------------------------------------
# Integration: area
# ---------------------------------------------------------------------------


def test_quantify_area_scope_subset(tmp_path):
    from verso.engine.model.annotation import AreaAnnotation

    project, pdir, atlas = _make_project(tmp_path)
    work_w, work_h = project.sections[0].resolution_thumbnail_wh
    # An area mask covering a small corner on the first section only.
    amask = np.zeros((work_h, work_w), dtype=bool)
    amask[:2, :2] = True
    area = AreaAnnotation(title="inj", masks={Path(project.sections[0].original_path).name: amask})
    save_annotations(pdir, [area])

    intensity = quantify_intensity(project, project_dir=pdir, atlas=atlas)["regions"]
    area_rows = quantify_area(project, "inj", project_dir=pdir, atlas=atlas)["regions"]
    assert area_rows
    # Area scope is a subset of the slice scope, so total area pixels <= intensity pixels.
    assert sum(r["n_pixels"] for r in area_rows) < sum(r["n_pixels"] for r in intensity)


# ---------------------------------------------------------------------------
# Integration: dots
# ---------------------------------------------------------------------------


def _add_points(pdir: Path, section: Section, xy: list[tuple[float, float]]) -> None:
    image = Path(section.original_path).name
    series = PointSeries(
        title="cells", points=[AnnotationPoint(x=x, y=y, image=image) for x, y in xy]
    )
    save_annotations(pdir, [series])


def test_quantify_dots_table_and_counts(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    s = project.sections[0]
    full_w, full_h = s.resolution_original_wh
    # Points inside the (left-half) slice mask and one clearly outside it (right edge).
    inside = [(1.0, 1.0), (2.0, 3.0), (3.0, 4.0)]
    outside = [(full_w - 1.0, full_h - 1.0)]
    _add_points(pdir, s, inside + outside)

    res = quantify_dots(project, "cells", project_dir=pdir, atlas=atlas)
    per_dot = res["dots"]
    # The outside-mask dot is dropped (RULE); the inside ones are kept.
    assert len(per_dot) == len(inside)
    for row in per_dot:
        assert {"x", "y", "image", "x_ccf", "y_ccf", "z_ccf", "region_id", "acronym"} <= set(row)

    regions = res["regions"]
    assert sum(r["n_dots"] for r in regions) == len(inside)
    for r in regions:
        if r["n_pixels"]:
            assert r["dots_density"] == pytest.approx(r["n_dots"] / r["n_pixels"])


def test_quantify_dots_intensity_circle(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    s = project.sections[0]
    _add_points(pdir, s, [(2.0, 2.0)])
    res = quantify_dots(
        project,
        "cells",
        intensity_channels=["C0"],
        dot_diameter_px=1.0,
        project_dir=pdir,
        atlas=atlas,
    )
    row = res["dots"][0]
    col = channel_column("mean_intensity", "C0")
    assert col in row
    # diameter 1 -> the single pixel under the dot
    from verso.engine.io.image_io import load_full_res_raw

    raw = load_full_res_raw(s.original_path)
    assert row[col] == pytest.approx(float(raw[2, 2, 0]))


def test_dots_ccf_axis_order(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    s = project.sections[0]
    _add_points(pdir, s, [(4.0, 4.0)])
    from verso.engine.registration import VersoRegistration

    reg = VersoRegistration.from_project(project)
    reg._atlas_volume = atlas
    ccf = reg.coord_image_to_atlas(s.id, np.array([[4.0, 4.0]]), space="full", units="um")[0]
    row = quantify_dots(project, "cells", project_dir=pdir, atlas=atlas)["dots"][0]
    assert row["x_ccf"] == pytest.approx(ccf[1])  # AP
    assert row["y_ccf"] == pytest.approx(ccf[2])  # DV
    assert row["z_ccf"] == pytest.approx(ccf[0])  # LR


def test_quantify_dots_hemisphere_split(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1, with_mask=False)
    s = project.sections[0]
    full_w, full_h = s.resolution_original_wh
    # One dot in the left half (region 10 / hemi l), one in the right (region 20 / r).
    left = (2.0, full_h / 2)
    right = (full_w - 2.0, full_h / 2)
    _add_points(pdir, s, [left, right])

    res = quantify_dots(
        project,
        "cells",
        project_dir=pdir,
        atlas=atlas,
        options=QuantifyOptions(split_hemispheres=True, include_unmasked_wholeframe=True),
    )
    per_dot = res["dots"]
    assert len(per_dot) == 2
    assert {r["hemisphere"] for r in per_dot} == {"l", "r"}

    # Per-region table splits by hemisphere; density denominator is per-hemisphere,
    # so every bucket that has a dot also has a pixel footprint (never zero-div).
    regions = res["regions"]
    assert all(r["hemisphere"] in {"l", "r", "none"} for r in regions)
    for r in regions:
        if r["n_dots"]:
            assert r["n_pixels"] > 0
            assert r["dots_density"] == pytest.approx(r["n_dots"] / r["n_pixels"])
    assert sum(r["n_dots"] for r in regions) == 2


def test_dots_aggregation_columns(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    _add_points(pdir, project.sections[0], [(2.0, 2.0)])
    row = quantify_dots(
        project,
        "cells",
        project_dir=pdir,
        atlas=atlas,
        options=QuantifyOptions(aggregate=("mid", "coarse")),
    )["dots"][0]
    assert {"mid_region_id", "mid_acronym", "coarse_region_id", "coarse_acronym"} <= set(row)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_precondition_missing_image_aborts(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    Path(project.sections[0].original_path).unlink()
    with pytest.raises(QuantificationError, match="not reachable"):
        quantify_intensity(project, project_dir=pdir, atlas=atlas)


def test_precondition_missing_alignment_aborts(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1)
    project.sections[0].alignment = Alignment()  # unanchored
    with pytest.raises(QuantificationError, match="no alignment"):
        quantify_intensity(project, project_dir=pdir, atlas=atlas)


def test_precondition_missing_cps_aborts_unless_allowed(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1, with_cps=False)
    with pytest.raises(QuantificationError, match="control points"):
        quantify_intensity(project, project_dir=pdir, atlas=atlas)
    # Allowed -> runs (affine-only)
    res = quantify_intensity(
        project,
        project_dir=pdir,
        atlas=atlas,
        options=QuantifyOptions(include_unwarped_affine=True),
    )
    assert res["regions"]


def test_precondition_missing_mask_aborts_unless_allowed(tmp_path):
    project, pdir, atlas = _make_project(tmp_path, n_sections=1, with_mask=False)
    with pytest.raises(QuantificationError, match="slice mask"):
        quantify_intensity(project, project_dir=pdir, atlas=atlas)
    res = quantify_intensity(
        project,
        project_dir=pdir,
        atlas=atlas,
        options=QuantifyOptions(include_unmasked_wholeframe=True),
    )
    assert res["regions"]


# ---------------------------------------------------------------------------
# quant_export
# ---------------------------------------------------------------------------


def test_write_csv_roundtrip(tmp_path):
    import csv

    from verso.engine.io.quant_export import write_csv

    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "t.csv"
    write_csv(path, records)
    with open(path, newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert [dict(r) for r in back] == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_slug_for_section_dedup(tmp_path):
    from verso.engine.io.quant_export import slug_for_section

    s1 = Section(id="a", slice_index=1, original_path="/x/slice.tif", thumbnail_path="")
    s2 = Section(id="b", slice_index=2, original_path="/y/slice.tif", thumbnail_path="")
    used: set[str] = set()
    n1 = slug_for_section(s1, used)
    n2 = slug_for_section(s2, used)
    assert n1 == "slice"
    assert n2 != n1  # de-duplicated
