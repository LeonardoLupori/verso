"""Tests for engine/registration.py."""

import math

import numpy as np
import pytest

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Section
from verso.engine.registration import (
    anchoring_to_vectors,
    atlas_to_normalized,
    flip_anchoring_horizontal,
    interpolate_anchorings,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    quicknii_coronal_default_anchoring,
    quicknii_coronal_series_anchorings,
    quicknii_pack_anchoring,
    quicknii_unpack_anchoring,
    rotate_anchoring,
    scale_anchoring,
    set_ap_center_position,
    set_ap_position,
    vectors_to_anchoring,
)

# Coronal anchoring for Allen Mouse 25 µm atlas (illustrative — not exact).
# Represents a mid-brain coronal section.
SAMPLE_ANCHORING = [
    # origin: left-top corner of the section in voxel space
    0.0, 160.0, 228.0,
    # u: 456 px wide section → moves 456 voxels along x
    456.0, 0.0, 0.0,
    # v: 320 px tall section → moves 320 voxels along y
    0.0, 320.0, 0.0,
]


# ---------------------------------------------------------------------------
# anchoring_to_vectors / vectors_to_anchoring
# ---------------------------------------------------------------------------

def test_anchoring_round_trip():
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    rebuilt = vectors_to_anchoring(o, u, v)
    np.testing.assert_allclose(rebuilt, SAMPLE_ANCHORING)


def test_anchoring_to_vectors_wrong_length():
    with pytest.raises(ValueError):
        anchoring_to_vectors([1.0] * 8)


# ---------------------------------------------------------------------------
# normalized_to_atlas / atlas_to_normalized
# ---------------------------------------------------------------------------

def test_origin_maps_to_origin():
    xyz = normalized_to_atlas(0.0, 0.0, SAMPLE_ANCHORING)
    np.testing.assert_allclose(xyz, SAMPLE_ANCHORING[:3])


def test_corner_maps_correctly():
    xyz = normalized_to_atlas(1.0, 1.0, SAMPLE_ANCHORING)
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    np.testing.assert_allclose(xyz, o + u + v)


def test_round_trip_normalized_atlas():
    for s, t in [(0.0, 0.0), (0.5, 0.3), (1.0, 1.0), (0.25, 0.75)]:
        xyz = normalized_to_atlas(s, t, SAMPLE_ANCHORING)
        s2, t2 = atlas_to_normalized(xyz, SAMPLE_ANCHORING)
        assert abs(s2 - s) < 1e-9, f"s mismatch at ({s}, {t})"
        assert abs(t2 - t) < 1e-9, f"t mismatch at ({s}, {t})"


# ---------------------------------------------------------------------------
# pixel_to_normalized / normalized_to_pixel
# ---------------------------------------------------------------------------

def test_pixel_normalized_round_trip():
    w, h = 456, 320
    for px, py in [(0, 0), (228, 160), (455, 319)]:
        s, t = pixel_to_normalized(px, py, w, h)
        px2, py2 = normalized_to_pixel(s, t, w, h)
        assert abs(px2 - px) < 1e-9
        assert abs(py2 - py) < 1e-9


# ---------------------------------------------------------------------------
# set_ap_position
# ---------------------------------------------------------------------------

def test_set_ap_position_changes_only_origin_z():
    new_anch = set_ap_position(SAMPLE_ANCHORING, ap_voxel=300.0, ap_axis=2)
    o, u, v = anchoring_to_vectors(new_anch)

    assert abs(o[2] - 300.0) < 1e-9
    # u and v unchanged
    np.testing.assert_allclose(u, SAMPLE_ANCHORING[3:6])
    np.testing.assert_allclose(v, SAMPLE_ANCHORING[6:9])


def test_set_ap_center_position_moves_midpoint_only():
    tilted = [
        10.0, 20.0, 30.0,
        100.0, 12.0, 0.0,
        0.0, 18.0, 80.0,
    ]

    new_anch = set_ap_center_position(tilted, ap_voxel=75.0, ap_axis=1)
    o, u, v = anchoring_to_vectors(new_anch)
    old_o, old_u, old_v = anchoring_to_vectors(tilted)
    center = o + (u + v) / 2.0

    assert abs(center[1] - 75.0) < 1e-9
    np.testing.assert_allclose(u, old_u)
    np.testing.assert_allclose(v, old_v)
    np.testing.assert_allclose(o[[0, 2]], old_o[[0, 2]])


# ---------------------------------------------------------------------------
# rotate_anchoring
# ---------------------------------------------------------------------------

def test_rotate_180_inverts_uv():
    rotated = rotate_anchoring(SAMPLE_ANCHORING, math.pi)
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    ro, ru, rv = anchoring_to_vectors(rotated)

    np.testing.assert_allclose(ru, -u, atol=1e-9)
    np.testing.assert_allclose(rv, -v, atol=1e-9)


def test_rotate_preserves_pivot_in_atlas_space():
    pivot_s, pivot_t = 0.5, 0.5
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    pivot_before = o + pivot_s * u + pivot_t * v

    rotated = rotate_anchoring(SAMPLE_ANCHORING, math.pi / 4, pivot_s, pivot_t)
    ro, ru, rv = anchoring_to_vectors(rotated)
    pivot_after = ro + pivot_s * ru + pivot_t * rv

    np.testing.assert_allclose(pivot_after, pivot_before, atol=1e-9)


# ---------------------------------------------------------------------------
# scale_anchoring
# ---------------------------------------------------------------------------

def test_scale_uniform_doubles_uv():
    scaled = scale_anchoring(SAMPLE_ANCHORING, 2.0)
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    so, su, sv = anchoring_to_vectors(scaled)

    np.testing.assert_allclose(su, 2.0 * u, atol=1e-9)
    np.testing.assert_allclose(sv, 2.0 * v, atol=1e-9)


def test_scale_preserves_pivot():
    pivot_s, pivot_t = 0.5, 0.5
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    pivot_before = o + pivot_s * u + pivot_t * v

    scaled = scale_anchoring(SAMPLE_ANCHORING, 1.5, pivot_s=pivot_s, pivot_t=pivot_t)
    so, su, sv = anchoring_to_vectors(scaled)
    pivot_after = so + pivot_s * su + pivot_t * sv

    np.testing.assert_allclose(pivot_after, pivot_before, atol=1e-9)


def test_flip_anchoring_horizontal_is_involutive():
    flipped = flip_anchoring_horizontal(SAMPLE_ANCHORING)
    restored = flip_anchoring_horizontal(flipped)

    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    fo, fu, fv = anchoring_to_vectors(flipped)

    np.testing.assert_allclose(fo, o + u)
    np.testing.assert_allclose(fu, -u)
    np.testing.assert_allclose(fv, v)
    np.testing.assert_allclose(restored, SAMPLE_ANCHORING)


# ---------------------------------------------------------------------------
# quicknii_coronal_default_anchoring
# ---------------------------------------------------------------------------

def test_quicknii_default_anchoring_uses_series_stretch():
    anchoring = quicknii_coronal_default_anchoring(
        image_width=500,
        image_height=400,
        max_width=1000,
        max_height=800,
        atlas_shape=(528, 320, 456),
    )

    o, u, v = anchoring_to_vectors(anchoring)
    np.testing.assert_allclose(u, [228.0, 0.0, 0.0])
    np.testing.assert_allclose(v, [0.0, 0.0, 160.0])
    np.testing.assert_allclose(o, [114.0, 264.0, 80.0])


def test_quicknii_pack_unpack_round_trip():
    unpacked = [456, 527, 160, 1, 0, 0, 0, 0, -1, 0.456, 0.4]
    anchoring = quicknii_pack_anchoring(unpacked, image_width=1000, image_height=800)
    restored = quicknii_unpack_anchoring(anchoring, image_width=1000, image_height=800)

    np.testing.assert_allclose(restored, unpacked)


def test_quicknii_coronal_series_initializes_ap_endpoints():
    anchorings = quicknii_coronal_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        serial_numbers=[1, 2, 3],
        atlas_shape=(528, 320, 456),
    )

    centers = []
    vectors = []
    for anchoring in anchorings:
        o, u, v = anchoring_to_vectors(anchoring)
        centers.append(o + u / 2 + v / 2)
        vectors.append((u, v))
    np.testing.assert_allclose([c[1] for c in centers], [527.0, 263.5, 0.0])
    np.testing.assert_allclose([c[0] for c in centers], [228.0, 228.0, 228.0])
    np.testing.assert_allclose([c[2] for c in centers], [160.0, 160.0, 160.0])
    np.testing.assert_allclose(vectors[0][0], [456.0, 0.0, 0.0])
    np.testing.assert_allclose(vectors[0][1], [0.0, 0.0, 320.0])


def test_quicknii_coronal_series_can_reverse_ap_proposal():
    anchorings = quicknii_coronal_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        serial_numbers=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        reverse_ap=True,
    )

    centers = []
    for anchoring in anchorings:
        o, u, v = anchoring_to_vectors(anchoring)
        centers.append(o + u / 2 + v / 2)
    np.testing.assert_allclose([c[1] for c in centers], [0.0, 263.5, 527.0])


def test_quicknii_coronal_series_uses_serial_numbers_not_list_indices():
    anchorings = quicknii_coronal_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        serial_numbers=[30, 10, 20],
        atlas_shape=(528, 320, 456),
    )

    centers_by_serial = {}
    for serial, anchoring in zip([30, 10, 20], anchorings):
        o, u, v = anchoring_to_vectors(anchoring)
        centers_by_serial[serial] = o + u / 2 + v / 2

    np.testing.assert_allclose(
        [centers_by_serial[n][1] for n in [10, 20, 30]],
        [527.0, 263.5, 0.0],
    )


def test_interpolate_anchorings_uses_quicknii_decomposed_space(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    stored = quicknii_coronal_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        serial_numbers=[1, 2, 3],
        atlas_shape=(528, 320, 456),
    )
    sections = [
        Section(
            id="s001",
            serial_number=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            alignment=Alignment(anchoring=stored[0], status=AlignmentStatus.COMPLETE),
        ),
        Section(
            id="s002",
            serial_number=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
        ),
        Section(
            id="s003",
            serial_number=3,
            original_path=str(paths[2]),
            thumbnail_path=str(paths[2]),
            alignment=Alignment(anchoring=stored[2], status=AlignmentStatus.COMPLETE),
        ),
    ]

    interpolate_anchorings(sections)

    expected = quicknii_coronal_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        serial_numbers=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        stored_anchorings=[stored[0], None, stored[2]],
    )
    np.testing.assert_allclose(sections[1].alignment.anchoring, expected[1])
    assert sections[1].alignment.status == AlignmentStatus.IN_PROGRESS


def test_interpolate_anchorings_with_one_keyframe_matches_quicknii_noop(tmp_path):
    from PIL import Image

    paths = []
    for i in range(2):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    sections = [
        Section(
            id="s001",
            serial_number=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            alignment=Alignment(
                anchoring=SAMPLE_ANCHORING,
                status=AlignmentStatus.COMPLETE,
            ),
        ),
        Section(
            id="s002",
            serial_number=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
        ),
    ]

    interpolate_anchorings(sections)

    assert sections[1].alignment.anchoring == [0.0] * 9
    assert sections[1].alignment.status == AlignmentStatus.NOT_STARTED


# ---------------------------------------------------------------------------
# make_atlas_sample_grid
# ---------------------------------------------------------------------------

def test_sample_grid_shape():
    grid = make_atlas_sample_grid(SAMPLE_ANCHORING, out_width=10, out_height=8)
    assert grid.shape == (8, 10, 3)


def test_sample_grid_corners():
    grid = make_atlas_sample_grid(SAMPLE_ANCHORING, out_width=10, out_height=8)
    o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)

    np.testing.assert_allclose(grid[0, 0], o, atol=1e-9)
    np.testing.assert_allclose(grid[0, -1], o + u, atol=1e-9)
    np.testing.assert_allclose(grid[-1, 0], o + v, atol=1e-9)
    np.testing.assert_allclose(grid[-1, -1], o + u + v, atol=1e-9)
