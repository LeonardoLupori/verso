"""Tests for engine/registration.py."""

import math

import numpy as np
import pytest

from verso.engine.registration import (
    anchoring_to_vectors,
    atlas_to_normalized,
    make_atlas_sample_grid,
    normalized_to_atlas,
    pixel_to_normalized,
    normalized_to_pixel,
    rotate_anchoring,
    scale_anchoring,
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
