"""Tests for engine/io/export_stack.py — the aligned (un-warped) stack export.

The export inverts the registration to resample each section onto a clean,
axis-aligned atlas plane. These tests cover the geometric inverse
(``build_canonical_remap``), the section resampler (``export_section_aligned``,
including the flip and slice-mask handling), and the merge/background
post-processing (``finalize_aligned_pages``), plus the OME-TIFF writer.

The real :class:`AtlasVolume` needs a brainglobe download, so we build a bare
instance with ``object.__new__`` and a tiny fake annotation — the canonical-plane
helpers only read ``annotation.shape``. Disk I/O in ``ensure_working_copy`` is
monkeypatched away; the slice mask is exercised through a real PNG round-trip.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from verso.engine.atlas import AtlasVolume
from verso.engine.io import export_stack
from verso.engine.io.export_stack import (
    ExportStackOptions,
    _has_usable_anchoring,
    build_canonical_remap,
    export_aligned_stack,
    export_section_aligned,
    finalize_aligned_pages,
    write_aligned_stack,
)
from verso.engine.model.alignment import Alignment, ControlPoint, WarpState
from verso.engine.model.project import Preprocessing, Section
from verso.engine.preprocessing import save_mask

# Fake atlas dimensions (AP, DV, LR) — small so grids are cheap to reason about.
_AP, _DV, _LR = 8, 6, 10


def _fake_atlas(ap: int = _AP, dv: int = _DV, lr: int = _LR) -> AtlasVolume:
    """An AtlasVolume whose canonical-plane helpers work without brainglobe."""
    atlas = object.__new__(AtlasVolume)
    atlas._annotation = np.zeros((ap, dv, lr), dtype=np.int32)
    return atlas


def _canonical_section(
    atlas: AtlasVolume,
    axis: int = 1,
    position: float = 4.0,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
    cps: list[ControlPoint] | None = None,
    mask_path: str | None = None,
    slice_index: int = 1,
) -> Section:
    """Section whose anchoring *is* the canonical plane → identity inverse-affine."""
    anchoring = atlas.canonical_plane_anchoring(position, axis)
    return Section(
        id="s1",
        slice_index=slice_index,
        original_path="img.tif",
        thumbnail_path="thumb.png",
        preprocessing=Preprocessing(
            flip_horizontal=flip_h, flip_vertical=flip_v, slice_mask_path=mask_path
        ),
        alignment=Alignment(current_anchoring=anchoring),
        warp=WarpState(control_points=cps or []),
    )


def _project(axis_index: int = 1) -> SimpleNamespace:
    return SimpleNamespace(working_scale=0.2, interpolation_axis_index=axis_index)


# --- Canonical plane helpers -------------------------------------------------


@pytest.mark.parametrize(
    "axis,expected",
    [(0, (_AP, _DV)), (1, (_LR, _DV)), (2, (_LR, _AP))],
)
def test_axis_plane_dims(axis, expected):
    assert _fake_atlas().axis_plane_dims(axis) == expected


def test_canonical_plane_anchoring_is_axis_aligned():
    """The canonical plane holds the slicing axis constant and spans the rest."""
    atlas = _fake_atlas()
    anchoring = atlas.canonical_plane_anchoring(position=4.0, axis=1)  # AP
    from verso.engine.anchoring import make_atlas_sample_grid

    out_w, out_h = 10, 6
    grid = make_atlas_sample_grid(anchoring, out_width=out_w, out_height=out_h)
    # AP (component 1) is constant at the requested position across the plane.
    np.testing.assert_allclose(grid[:, :, 1], 4.0)
    # LR (0) and DV (2) sweep the atlas extent. With the i/N sampling convention
    # the last pixel stops one step short of the full extent (s never reaches 1).
    assert grid[:, :, 0].min() == 0.0
    assert grid[:, :, 0].max() == pytest.approx(_LR * (out_w - 1) / out_w)
    assert grid[:, :, 2].min() == 0.0
    assert grid[:, :, 2].max() == pytest.approx(_DV * (out_h - 1) / out_h)


def test_axis_plane_dims_rejects_bad_axis():
    with pytest.raises(ValueError):
        _fake_atlas().axis_plane_dims(3)


# --- build_canonical_remap ---------------------------------------------------


def test_build_canonical_remap_identity_ramps():
    """With a canonical anchoring and no warp, the maps are plain linear ramps.

    map_x depends only on the column (recovering s·work_w); map_y only on the
    row (t·work_h). Everything is covered.
    """
    atlas = _fake_atlas()
    section = _canonical_section(atlas, axis=1, position=4.0)
    work_w, work_h = 20, 12

    map_x, map_y, out_w, out_h = build_canonical_remap(
        section, atlas, axis=1, scale=1.0, work_w=work_w, work_h=work_h
    )
    assert (out_w, out_h) == (_LR, _DV)  # (10, 6)

    cols = np.arange(out_w) / out_w * work_w
    rows = np.arange(out_h) / out_h * work_h
    np.testing.assert_allclose(map_x, np.broadcast_to(cols, (out_h, out_w)), atol=1e-4)
    np.testing.assert_allclose(map_y, np.broadcast_to(rows[:, None], (out_h, out_w)), atol=1e-4)
    assert (map_x >= 0).all()  # fully covered


def test_build_canonical_remap_scale_changes_output_size():
    atlas = _fake_atlas()
    section = _canonical_section(atlas, axis=1)
    _, _, out_w, out_h = build_canonical_remap(section, atlas, 1, 2.0, 20, 12)
    assert (out_w, out_h) == (_LR * 2, _DV * 2)


def test_build_canonical_remap_marks_uncovered_pixels():
    """A section that only spans half the plane leaves the rest uncovered (-1)."""
    atlas = _fake_atlas()
    # Anchoring spans only the left half in LR (u = LR/2) → right half has s > 1.
    anchoring = [0.0, 4.0, 0.0, _LR / 2, 0.0, 0.0, 0.0, 0.0, float(_DV)]
    section = _canonical_section(atlas)
    section.alignment.current_anchoring = anchoring
    map_x, _, out_w, _ = build_canonical_remap(section, atlas, 1, 1.0, 20, 12)
    covered = map_x >= 0
    assert covered[:, : out_w // 2].all()  # left half covered
    assert not covered[:, -1].any()  # far-right column uncovered


# --- export_section_aligned --------------------------------------------------


def test_export_section_aligned_none_without_anchoring(monkeypatch):
    atlas = _fake_atlas()
    section = _canonical_section(atlas)
    section.alignment.current_anchoring = [0.0] * 9  # degenerate
    # ensure_working_copy must not even be called for a degenerate anchoring.
    monkeypatch.setattr(
        export_stack, "ensure_working_copy", lambda *a, **k: pytest.fail("should not load")
    )
    assert export_section_aligned(section, _project(), atlas, scale=1.0) is None


def test_export_section_aligned_none_without_working_copy(monkeypatch):
    atlas = _fake_atlas()
    section = _canonical_section(atlas)
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: None)
    assert export_section_aligned(section, _project(), atlas, scale=1.0) is None


def test_export_section_aligned_preserves_orientation(monkeypatch):
    """A left→right intensity gradient stays left→right after resampling."""
    atlas = _fake_atlas()
    work_w, work_h = 20, 12
    gradient = np.tile(np.arange(work_w, dtype=np.uint8), (work_h, 1))[:, :, None]
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: gradient.copy())

    page, valid = export_section_aligned(_canonical_section(atlas), _project(), atlas, scale=1.0)
    assert valid.all()
    # Compare interior columns; the very last column maps one pixel past the
    # source edge (s == 1 → pixel == work_w) and reads the border.
    assert page[:, 5:9, 0].mean() > page[:, 1:5, 0].mean()


def test_export_section_aligned_applies_horizontal_flip(monkeypatch):
    """flip_horizontal must mirror the sampled section (regression: flips were ignored).

    Control points and masks live in the displayed (flipped) space, so the
    resampler must flip the working image before sampling.
    """
    atlas = _fake_atlas()
    work_w, work_h = 20, 12
    gradient = np.tile(np.arange(work_w, dtype=np.uint8), (work_h, 1))[:, :, None]
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: gradient.copy())

    base, _ = export_section_aligned(_canonical_section(atlas), _project(), atlas, scale=1.0)
    flipped, _ = export_section_aligned(
        _canonical_section(atlas, flip_h=True), _project(), atlas, scale=1.0
    )
    # The flip is applied: output differs and its left→right orientation reverses.
    assert not np.allclose(base, flipped)
    assert base[:, 5:9, 0].mean() > base[:, 1:5, 0].mean()
    assert flipped[:, 1:5, 0].mean() > flipped[:, 5:9, 0].mean()


def test_export_section_aligned_applies_slice_mask(monkeypatch, tmp_path):
    """With apply_slice_mask, pixels outside the (remapped) mask are invalid/zeroed."""
    atlas = _fake_atlas()
    work_w, work_h = 20, 12
    img = np.full((work_h, work_w, 1), 100, dtype=np.uint8)
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: img.copy())

    mask = np.zeros((work_h, work_w), dtype=bool)
    mask[:, : work_w // 2] = True  # tissue in the left half (working space)
    mask_path = tmp_path / "slice-mask.png"
    save_mask(mask, mask_path)
    section = _canonical_section(atlas, mask_path=str(mask_path))

    page, valid = export_section_aligned(
        section, _project(), atlas, scale=1.0, apply_slice_mask=True
    )
    # Left columns map into the tissue half → valid (excluding the last row,
    # which maps one pixel past the source edge); far-right columns do not.
    assert valid[:-1, 0].all()
    assert not valid[:, -1].any()
    # Page is zeroed wherever it is invalid; constant tissue stays 100 elsewhere.
    assert (page[~valid] == 0).all()
    assert (page[valid] == 100).all()


def test_export_section_aligned_ignores_mask_when_disabled(monkeypatch, tmp_path):
    atlas = _fake_atlas()
    img = np.full((12, 20, 1), 100, dtype=np.uint8)
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: img.copy())
    mask = np.zeros((12, 20), dtype=bool)
    mask[:, :5] = True
    mask_path = tmp_path / "m.png"
    save_mask(mask, mask_path)
    section = _canonical_section(atlas, mask_path=str(mask_path))

    _, valid = export_section_aligned(section, _project(), atlas, scale=1.0, apply_slice_mask=False)
    assert valid.all()  # coverage only; mask not applied


# --- finalize_aligned_pages --------------------------------------------------


def _entry(slice_index, fill, valid_cols, h=4, w=10, c=1):
    page = np.zeros((h, w, c), dtype=np.uint8)
    valid = np.zeros((h, w), dtype=bool)
    valid[:, valid_cols] = True
    page[valid] = fill
    return slice_index, page, valid


def test_finalize_no_merge_no_background_passthrough():
    e = _entry(1, 50, slice(None))
    pages = finalize_aligned_pages([e], ExportStackOptions())
    assert len(pages) == 1
    np.testing.assert_array_equal(pages[0], e[1])


def test_finalize_background_white_fills_invalid():
    e = _entry(1, 50, slice(0, 5))  # valid left half only
    pages = finalize_aligned_pages([e], ExportStackOptions(background="white"))
    page = pages[0]
    assert (page[:, :5, 0] == 50).all()
    assert (page[:, 5:, 0] == 255).all()


def test_finalize_background_black_fills_invalid():
    e = _entry(1, 50, slice(0, 5))
    pages = finalize_aligned_pages([e], ExportStackOptions(background="black"))
    assert (pages[0][:, 5:, 0] == 0).all()


def test_finalize_merge_max_projects_same_slice_index():
    a = _entry(7, 50, slice(None))
    b = _entry(7, 80, slice(None))
    pages = finalize_aligned_pages([a, b], ExportStackOptions(merge_by_slice_index=True))
    assert len(pages) == 1
    assert (pages[0] == 80).all()  # element-wise max


def test_finalize_merge_keeps_distinct_slice_indices_in_order():
    a = _entry(2, 10, slice(None))
    b = _entry(5, 20, slice(None))
    pages = finalize_aligned_pages([a, b], ExportStackOptions(merge_by_slice_index=True))
    assert len(pages) == 2
    assert pages[0][0, 0, 0] == 10 and pages[1][0, 0, 0] == 20


def test_finalize_merge_unions_masks_then_whitens_gaps():
    """White background + merge: the union of pieces defines the kept region.

    Two complementary halves of one physical slice cover the whole plane once
    merged, so no white background remains and intensities are preserved.
    """
    left = _entry(3, 50, slice(0, 5))
    right = _entry(3, 80, slice(5, 10))
    pages = finalize_aligned_pages(
        [left, right],
        ExportStackOptions(background="white", merge_by_slice_index=True),
    )
    page = pages[0]
    assert len(pages) == 1
    assert not (page == 255).any()  # union covers everything → no white
    assert (page[:, :5, 0] == 50).all()
    assert (page[:, 5:, 0] == 80).all()


# --- write_aligned_stack -----------------------------------------------------


def test_write_aligned_stack_shape_and_metadata(tmp_path):
    pages = [np.full((6, 10, 2), k, dtype=np.uint8) for k in range(3)]
    out = tmp_path / "stack.ome.tif"
    write_aligned_stack(pages, ["c0", "c1"], out)
    data = tifffile.imread(str(out))
    assert data.shape == (3, 2, 6, 10)  # (Z, C, H, W)
    assert data[1, 0, 0, 0] == 1


def test_write_aligned_stack_rejects_empty():
    with pytest.raises(ValueError):
        write_aligned_stack([], ["c0"], None)  # type: ignore[arg-type]


# --- _has_usable_anchoring ---------------------------------------------------


def test_has_usable_anchoring():
    atlas = _fake_atlas()
    assert _has_usable_anchoring(_canonical_section(atlas))
    bad = _canonical_section(atlas)
    bad.alignment.current_anchoring = [0.0] * 9
    assert not _has_usable_anchoring(bad)


# --- export_aligned_stack (driver) -------------------------------------------


def test_export_aligned_stack_skips_unaligned_and_writes(monkeypatch, tmp_path):
    atlas = _fake_atlas()
    img = np.full((12, 20, 1), 100, dtype=np.uint8)
    monkeypatch.setattr(export_stack, "ensure_working_copy", lambda *a, **k: img.copy())

    good = _canonical_section(atlas, slice_index=1)
    good.id = "good"
    bad = _canonical_section(atlas, slice_index=2)
    bad.id = "bad"
    bad.alignment.current_anchoring = [0.0] * 9
    project = SimpleNamespace(
        working_scale=0.2,
        interpolation_axis_index=1,
        channels=[SimpleNamespace(name="c0")],
    )
    out = tmp_path / "out.ome.tif"

    written, skipped = export_aligned_stack([good, bad], project, atlas, ExportStackOptions(), out)
    assert written == out and out.exists()
    assert skipped == ["bad"]  # the degenerate section, by id
    # Only the good section made a page (tifffile squeezes the singleton Z/C).
    with tifffile.TiffFile(str(out)) as tf:
        assert len(tf.pages) == 1
