"""Tests for engine/anchoring.py."""

import math

import numpy as np
import pytest

from verso.engine.anchoring import (
    anchoring_to_vectors,
    atlas_to_normalized,
    clamp_inplane_rotation,
    clamp_rotation_to_max_tilt,
    flip_anchoring_horizontal,
    interpolate_anchorings,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    plane_tilt_deg,
    quicknii_default_anchoring,
    quicknii_pack_anchoring,
    quicknii_series_anchorings,
    quicknii_unpack_anchoring,
    rotate_anchoring,
    scale_anchoring,
    set_center_position_along_axis,
    set_position_along_axis,
    tilt_plane_about_atlas_axis,
    vectors_to_anchoring,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Preprocessing, Section

# Coronal anchoring for Allen Mouse 25 µm atlas (illustrative — not exact).
# Represents a mid-brain coronal section.
SAMPLE_ANCHORING = [
    # origin: left-top corner of the section in voxel space
    0.0,
    160.0,
    228.0,
    # u: 456 px wide section → moves 456 voxels along x
    456.0,
    0.0,
    0.0,
    # v: 320 px tall section → moves 320 voxels along y
    0.0,
    320.0,
    0.0,
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
# set_position_along_axis
# ---------------------------------------------------------------------------


def test_set_position_along_axis_changes_only_origin_z():
    new_anch = set_position_along_axis(SAMPLE_ANCHORING, voxel=300.0, axis=2)
    o, u, v = anchoring_to_vectors(new_anch)

    assert abs(o[2] - 300.0) < 1e-9
    # u and v unchanged
    np.testing.assert_allclose(u, SAMPLE_ANCHORING[3:6])
    np.testing.assert_allclose(v, SAMPLE_ANCHORING[6:9])


def test_set_center_position_along_axis_moves_midpoint_only():
    tilted = [
        10.0,
        20.0,
        30.0,
        100.0,
        12.0,
        0.0,
        0.0,
        18.0,
        80.0,
    ]

    new_anch = set_center_position_along_axis(tilted, voxel=75.0, axis=1)
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
    _o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    _ro, ru, rv = anchoring_to_vectors(rotated)

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
    _o, u, v = anchoring_to_vectors(SAMPLE_ANCHORING)
    _so, su, sv = anchoring_to_vectors(scaled)

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
# quicknii_default_anchoring
# ---------------------------------------------------------------------------


def test_quicknii_default_anchoring_uses_series_stretch():
    anchoring = quicknii_default_anchoring(
        image_width=500,
        image_height=400,
        max_width=1000,
        max_height=800,
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
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
    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
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
    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        reverse_axis=True,
    )

    centers = []
    for anchoring in anchorings:
        o, u, v = anchoring_to_vectors(anchoring)
        centers.append(o + u / 2 + v / 2)
    np.testing.assert_allclose([c[1] for c in centers], [0.0, 263.5, 527.0])


def test_quicknii_coronal_series_uses_slice_indices_not_list_indices():
    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[30, 10, 20],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
    )

    centers_by_serial = {}
    for serial, anchoring in zip([30, 10, 20], anchorings, strict=True):
        o, u, v = anchoring_to_vectors(anchoring)
        centers_by_serial[serial] = o + u / 2 + v / 2

    np.testing.assert_allclose(
        [centers_by_serial[n][1] for n in [10, 20, 30]],
        [527.0, 263.5, 0.0],
    )


def test_quicknii_coronal_series_duplicate_serial_gets_stored_ap_but_default_orientation():
    stored = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[9, 10, 11],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
    )

    anchorings = quicknii_series_anchorings(
        image_sizes=[(800, 600), (1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[10, 10, 11, 12],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        stored_anchorings=[None, stored[1], None, None],
    )

    stored_u = quicknii_unpack_anchoring(anchorings[1], 1000, 800)
    dup_u = quicknii_unpack_anchoring(anchorings[0], 800, 600)

    # AP position matches the stored section.
    np.testing.assert_allclose(dup_u[1], stored_u[1])
    # Orientation is reset to the default upright coronal, not copied from stored.
    np.testing.assert_allclose(dup_u[3:9], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-9)
    # LR and DV are at the atlas centre.
    np.testing.assert_allclose(dup_u[0], 228.0)  # lr_dim/2 = 456/2
    np.testing.assert_allclose(dup_u[2], 160.0)  # dv_dim/2 = 320/2


def test_quicknii_coronal_series_same_serial_same_ap_with_different_sizes():
    """Sections sharing a serial get the same AP position regardless of image size."""
    anchorings = quicknii_series_anchorings(
        image_sizes=[(800, 600), (1000, 800), (600, 400)],
        slice_indices=[10, 10, 10],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
    )
    # All three must land on the same AP voxel (same midpoint in atlas space).
    centers = [
        anchoring_to_vectors(a)[0] + anchoring_to_vectors(a)[1] / 2 + anchoring_to_vectors(a)[2] / 2
        for a in anchorings
    ]
    np.testing.assert_allclose(centers[0][1], centers[1][1])  # AP axis
    np.testing.assert_allclose(centers[0][1], centers[2][1])


def test_quicknii_coronal_series_centers_generated_proposals_from_off_center_keyframes():
    off_center_left = quicknii_pack_anchoring(
        [120.0, 500.0, 90.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.456, 0.4],
        image_width=1000,
        image_height=800,
    )
    off_center_right = quicknii_pack_anchoring(
        [340.0, 100.0, 250.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.456, 0.4],
        image_width=1000,
        image_height=800,
    )

    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        stored_anchorings=[off_center_left, None, off_center_right],
    )

    proposal = quicknii_unpack_anchoring(anchorings[1], 1000, 800)
    np.testing.assert_allclose(proposal[0], 228.0)
    np.testing.assert_allclose(proposal[2], 160.0)
    np.testing.assert_allclose(anchorings[0], off_center_left)
    np.testing.assert_allclose(anchorings[2], off_center_right)


def test_quicknii_coronal_series_proposals_are_upright_even_when_keyframe_is_rotated():
    """Interpolated proposals must have default (upright) rotation regardless of keyframes."""
    left_anchoring = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        voxel=400.0,
    )
    right_base = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        voxel=100.0,
    )
    right_anchoring = rotate_anchoring(right_base, math.radians(15))

    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800)] * 3,
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        stored_anchorings=[left_anchoring, None, right_anchoring],
    )

    mid_u = quicknii_unpack_anchoring(anchorings[1], 1000, 800)
    default_u = quicknii_unpack_anchoring(
        quicknii_default_anchoring(
            image_width=1000,
            image_height=800,
            max_width=1000,
            max_height=800,
            atlas_shape=(528, 320, 456),
            interpolation_axis=1,
        ),
        1000,
        800,
    )
    # Rotation components of the proposal must equal the default upright orientation.
    np.testing.assert_allclose(mid_u[3:9], default_u[3:9], atol=1e-9)


def test_interpolate_anchorings_uses_quicknii_decomposed_space(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    stored = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
    )
    sections = [
        Section(
            id="s001",
            slice_index=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(anchoring=stored[0], status=AlignmentStatus.COMPLETE),
        ),
        Section(
            id="s002",
            slice_index=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
        ),
        Section(
            id="s003",
            slice_index=3,
            original_path=str(paths[2]),
            thumbnail_path=str(paths[2]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(anchoring=stored[2], status=AlignmentStatus.COMPLETE),
        ),
    ]

    interpolate_anchorings(sections)

    expected = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        stored_anchorings=[stored[0], None, stored[2]],
    )
    np.testing.assert_allclose(sections[1].alignment.anchoring, expected[1])
    assert sections[1].alignment.status == AlignmentStatus.IN_PROGRESS


def test_interpolate_anchorings_with_one_keyframe_matches_quicknii(tmp_path):
    from PIL import Image

    paths = []
    for i in range(2):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    sections = [
        Section(
            id="s001",
            slice_index=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(
                anchoring=SAMPLE_ANCHORING,
                status=AlignmentStatus.COMPLETE,
            ),
        ),
        Section(
            id="s002",
            slice_index=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
        ),
    ]

    interpolate_anchorings(sections, atlas_shape=(528, 320, 456))

    expected = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800)],
        slice_indices=[1, 2],
        atlas_shape=(528, 320, 456),
        interpolation_axis=1,
        stored_anchorings=[SAMPLE_ANCHORING, None],
    )
    np.testing.assert_allclose(sections[1].alignment.anchoring, expected[1])
    assert sections[1].alignment.status == AlignmentStatus.IN_PROGRESS


def test_interpolate_anchorings_handles_horizontally_flipped_stored_keyframe(
    tmp_path,
):
    from PIL import Image

    atlas_shape = (528, 320, 456)
    paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    angle = math.radians(18.0)
    unpacked_left = [
        228.0,
        500.0,
        160.0,
        math.cos(angle),
        math.sin(angle),
        0.0,
        0.0,
        0.0,
        1.0,
        0.456,
        0.4,
    ]
    unpacked_right = list(unpacked_left)
    unpacked_right[1] = 100.0
    left = quicknii_pack_anchoring(unpacked_left, 1000, 800)
    right = quicknii_pack_anchoring(unpacked_right, 1000, 800)
    sections = [
        Section(
            id="s001",
            slice_index=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(anchoring=left, status=AlignmentStatus.COMPLETE),
        ),
        Section(
            id="s002",
            slice_index=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
        ),
        Section(
            id="s003",
            slice_index=3,
            original_path=str(paths[2]),
            thumbnail_path=str(paths[2]),
            resolution_thumbnail_wh=(1000, 800),
            preprocessing=Preprocessing(flip_horizontal=True),
            alignment=Alignment(
                anchoring=right,
                stored_anchoring=right,
                status=AlignmentStatus.COMPLETE,
            ),
        ),
    ]

    interpolate_anchorings(sections, atlas_shape=atlas_shape)

    middle = quicknii_unpack_anchoring(sections[1].alignment.anchoring, 1000, 800)
    np.testing.assert_allclose(middle[4], math.sin(angle), atol=1e-9)
    np.testing.assert_allclose(middle[1], 300.0, atol=1e-9)
    assert sections[1].alignment.status == AlignmentStatus.IN_PROGRESS


def test_interpolate_anchorings_duplicate_serial_strips_inplane_rotation_keeps_tilt(
    tmp_path,
):
    from PIL import Image

    atlas_shape = (528, 320, 456)
    # Stored section: upright coronal anchoring with an in-plane rotation applied.
    stored_base = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=atlas_shape,
        interpolation_axis=1,
        voxel=300.0,
    )
    stored_anchoring = rotate_anchoring(stored_base, math.radians(20))

    paths = []
    image_sizes = [(800, 600), (1000, 800), (1000, 800)]
    for i, size in enumerate(image_sizes):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", size).save(path)
        paths.append(path)

    sections = [
        Section(
            id="s001",
            slice_index=10,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(800, 600),
        ),
        Section(
            id="s002",
            slice_index=10,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(
                anchoring=stored_anchoring,
                status=AlignmentStatus.COMPLETE,
            ),
        ),
        Section(
            id="s003",
            slice_index=11,
            original_path=str(paths[2]),
            thumbnail_path=str(paths[2]),
            resolution_thumbnail_wh=(1000, 800),
        ),
    ]

    interpolate_anchorings(sections, atlas_shape=atlas_shape)

    duplicate_unpacked = quicknii_unpack_anchoring(
        sections[0].alignment.anchoring,
        *image_sizes[0],
    )
    stored_unpacked = quicknii_unpack_anchoring(
        sections[1].alignment.anchoring,
        *image_sizes[1],
    )
    # AP position must match the stored section.
    np.testing.assert_allclose(duplicate_unpacked[1], stored_unpacked[1])
    # In-plane rotation removed; rotate_anchoring leaves u_y=v_y=0, so result is upright.
    np.testing.assert_allclose(duplicate_unpacked[3:9], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-9)
    assert sections[0].alignment.status == AlignmentStatus.IN_PROGRESS


def test_interpolate_anchorings_without_atlas_shape_keeps_legacy_one_keyframe_noop(
    tmp_path,
):
    from PIL import Image

    paths = []
    for i in range(2):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    sections = [
        Section(
            id="s001",
            slice_index=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(
                anchoring=SAMPLE_ANCHORING,
                status=AlignmentStatus.COMPLETE,
            ),
        ),
        Section(
            id="s002",
            slice_index=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
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


# ---------------------------------------------------------------------------
# Non-coronal axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis, axis_dim_idx, u_axis, v_axis",
    [
        (0, 2, 1, 2),  # sagittal (ML) — atlas_shape[2]=lr_dim along ML
        (1, 0, 0, 2),  # coronal (AP) — atlas_shape[0]=ap_dim along AP
        (2, 1, 0, 1),  # horizontal (DV) — atlas_shape[1]=dv_dim along DV
    ],
)
def test_quicknii_default_anchoring_for_each_axis(axis, axis_dim_idx, u_axis, v_axis):
    atlas_shape = (528, 320, 456)  # (AP, DV, LR)
    anchoring = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=atlas_shape,
        interpolation_axis=axis,
    )
    _o, u, v = anchoring_to_vectors(anchoring)
    # The slicing-axis component of u and v is zero (plane perpendicular to axis).
    assert abs(u[axis]) < 1e-9
    assert abs(v[axis]) < 1e-9
    # u lies along its natural axis, v along its natural axis.
    assert abs(u[u_axis]) > 0
    assert abs(v[v_axis]) > 0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_quicknii_series_endpoint_voxels_match_axis_dim(axis):
    atlas_shape = (528, 320, 456)
    qn_dims = (456, 528, 320)  # (ML, AP, DV)
    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800)],
        slice_indices=[1, 2],
        atlas_shape=atlas_shape,
        interpolation_axis=axis,
    )
    o0, u0, v0 = anchoring_to_vectors(anchorings[0])
    o1, u1, v1 = anchoring_to_vectors(anchorings[1])
    center0 = o0 + (u0 + v0) / 2.0
    center1 = o1 + (u1 + v1) / 2.0
    # First section endpoint sits at the far end of the slicing axis; second at 0.
    assert abs(center0[axis] - (qn_dims[axis] - 1)) < 1e-9
    assert abs(center1[axis] - 0.0) < 1e-9


def test_quicknii_sagittal_series_interpolates_along_ml():
    """For a sagittal series (axis=0), proposals should vary in ML, not AP."""
    atlas_shape = (528, 320, 456)
    anchorings = quicknii_series_anchorings(
        image_sizes=[(1000, 800), (1000, 800), (1000, 800)],
        slice_indices=[1, 2, 3],
        atlas_shape=atlas_shape,
        interpolation_axis=0,
    )
    centers = [
        anchoring_to_vectors(a)[0] + (anchoring_to_vectors(a)[1] + anchoring_to_vectors(a)[2]) / 2.0
        for a in anchorings
    ]
    # ML axis varies through the series.
    assert centers[0][0] != centers[1][0]
    assert centers[1][0] != centers[2][0]
    # AP/DV stay centered at the atlas midpoint.
    for c in centers:
        np.testing.assert_allclose(c[1], 528 / 2.0)  # AP midpoint
        np.testing.assert_allclose(c[2], 320 / 2.0)  # DV midpoint


def test_interpolate_anchorings_sagittal_axis_strips_in_plane_rotation(tmp_path):
    """Rotation around the slicing axis is stripped for any axis."""
    from PIL import Image

    atlas_shape = (528, 320, 456)
    axis = 0  # ML slicing
    paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1}.png"
        Image.new("RGB", (1000, 800)).save(path)
        paths.append(path)

    left = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=atlas_shape,
        interpolation_axis=axis,
        voxel=400.0,
    )
    right_base = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=atlas_shape,
        interpolation_axis=axis,
        voxel=50.0,
    )
    right = rotate_anchoring(right_base, math.radians(15))

    sections = [
        Section(
            id="s001",
            slice_index=1,
            original_path=str(paths[0]),
            thumbnail_path=str(paths[0]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(anchoring=left, status=AlignmentStatus.COMPLETE),
        ),
        Section(
            id="s002",
            slice_index=2,
            original_path=str(paths[1]),
            thumbnail_path=str(paths[1]),
            resolution_thumbnail_wh=(1000, 800),
        ),
        Section(
            id="s003",
            slice_index=3,
            original_path=str(paths[2]),
            thumbnail_path=str(paths[2]),
            resolution_thumbnail_wh=(1000, 800),
            alignment=Alignment(anchoring=right, status=AlignmentStatus.COMPLETE),
        ),
    ]

    interpolate_anchorings(sections, atlas_shape=atlas_shape, interpolation_axis=axis)

    mid_unpacked = quicknii_unpack_anchoring(sections[1].alignment.anchoring, 1000, 800)
    # The in-plane components in u and v that aren't the slicing axis must be
    # zero (rotation around the slicing axis stripped); the slicing-axis
    # components (the tilt) can be non-zero. For axis=0, in-plane axes are AP=1
    # and DV=2 so we expect u_unit ∝ (0, 1, 0) and v_unit ∝ (0, 0, 1).
    u_unit = mid_unpacked[3:6]
    v_unit = mid_unpacked[6:9]
    # rotate_anchoring around the section normal preserves the slicing-axis
    # component (axis=0) at zero, so after stripping rotation we expect
    # canonical units along axes 1 and 2.
    np.testing.assert_allclose(u_unit, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(v_unit, [0.0, 0.0, 1.0], atol=1e-9)
    assert sections[1].alignment.status == AlignmentStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# plane_tilt_deg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_plane_tilt_deg_zero_for_axis_aligned_plane(axis):
    atlas_shape = (528, 320, 456)
    anchoring = quicknii_default_anchoring(
        image_width=1000,
        image_height=800,
        max_width=1000,
        max_height=800,
        atlas_shape=atlas_shape,
        interpolation_axis=axis,
    )
    assert plane_tilt_deg(anchoring, axis) == pytest.approx(0.0, abs=1e-9)


def test_plane_tilt_deg_matches_rotation_angle():
    # Coronal default plane (axis=1): u along LR, v along DV, normal along AP.
    # Rotate v toward AP by 30° around LR; the plane should tilt by 30°.
    o = np.array([0.0, 264.0, 0.0])
    u = np.array([456.0, 0.0, 0.0])  # LR
    deg = 30.0
    a = math.radians(deg)
    v_tilted = np.array([0.0, math.sin(a) * 320.0, math.cos(a) * 320.0])
    anchoring = vectors_to_anchoring(o, u, v_tilted)
    assert plane_tilt_deg(anchoring, 1) == pytest.approx(deg, abs=1e-6)


def test_rotate_anchoring_is_in_plane_only():
    # In-plane rotation must not change the direction of the plane normal,
    # so plane_tilt_deg is invariant under rotate_anchoring (the basis for
    # clamping tilt independently of in-plane spin).
    anchoring = SAMPLE_ANCHORING
    before = plane_tilt_deg(anchoring, 1)
    rotated = rotate_anchoring(anchoring, math.radians(37.0))
    after = plane_tilt_deg(rotated, 1)
    assert after == pytest.approx(before, abs=1e-9)

    n0 = np.cross(*anchoring_to_vectors(anchoring)[1:])
    n1 = np.cross(*anchoring_to_vectors(rotated)[1:])
    cos = np.dot(n0, n1) / (np.linalg.norm(n0) * np.linalg.norm(n1))
    assert cos == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# tilt_plane_about_atlas_axis
# ---------------------------------------------------------------------------

# Axis-aligned coronal plane: u along LR (axis 0), v along DV (axis 2), so the
# plane normal is along AP (axis 1) → zero tilt relative to slicing axis 1.
_CORONAL_ALIGNED = vectors_to_anchoring(
    np.array([0.0, 264.0, 0.0]),
    np.array([456.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 320.0]),
)


def _spin_deg(anchoring, slicing_axis):
    """In-plane spin magnitude (deg) of ``u`` relative to axis-aligned."""
    u_axis, v_axis = sorted(i for i in (0, 1, 2) if i != slicing_axis)
    _o, u, _v = anchoring_to_vectors(anchoring)
    return abs(float(np.degrees(np.arctan2(u[v_axis], u[u_axis]))))


def test_tilt_plane_zero_is_identity():
    result = tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, axis=0, deg=0.0)
    np.testing.assert_allclose(result, _CORONAL_ALIGNED, atol=1e-9)


def test_tilt_plane_preserves_center():
    o, u, v = anchoring_to_vectors(_CORONAL_ALIGNED)
    center_before = o + u / 2.0 + v / 2.0
    o2, u2, v2 = anchoring_to_vectors(
        tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, axis=0, deg=27.0)
    )
    np.testing.assert_allclose(o2 + u2 / 2.0 + v2 / 2.0, center_before, atol=1e-9)


def test_tilt_plane_round_trip():
    tilted = tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, axis=2, deg=33.0)
    restored = tilt_plane_about_atlas_axis(tilted, axis=2, deg=-33.0)
    np.testing.assert_allclose(restored, _CORONAL_ALIGNED, atol=1e-9)


def test_tilt_plane_produces_expected_tilt():
    # Rotating the whole plane about LR (axis 0) tilts the AP-aligned normal by
    # the same angle, so the tilt relative to slicing axis 1 equals ``deg``.
    rotated = tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, axis=0, deg=30.0)
    assert plane_tilt_deg(rotated, 1) == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------------------
# clamp_rotation_to_max_tilt
# ---------------------------------------------------------------------------


def test_clamp_tilt_under_limit_unchanged():
    deg = clamp_rotation_to_max_tilt(
        _CORONAL_ALIGNED, axis=0, deg=10.0, slicing_axis=1, max_tilt_deg=44.0
    )
    assert deg == pytest.approx(10.0)


def test_clamp_tilt_shortens_overshoot_to_limit():
    deg = clamp_rotation_to_max_tilt(
        _CORONAL_ALIGNED, axis=0, deg=80.0, slicing_axis=1, max_tilt_deg=44.0
    )
    assert 0.0 < deg < 80.0
    tilt = plane_tilt_deg(tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, 0, deg), 1)
    assert tilt <= 44.0 + 1e-3
    assert deg == pytest.approx(44.0, abs=1e-3)


def test_clamp_tilt_no_op_about_slicing_axis():
    # Rotation about the slicing axis leaves tilt unchanged, so nothing is clamped.
    deg = clamp_rotation_to_max_tilt(
        _CORONAL_ALIGNED, axis=1, deg=80.0, slicing_axis=1, max_tilt_deg=44.0
    )
    assert deg == pytest.approx(80.0)


def test_clamp_tilt_already_over_limit_only_reduces():
    over = tilt_plane_about_atlas_axis(_CORONAL_ALIGNED, axis=0, deg=60.0)
    assert plane_tilt_deg(over, 1) == pytest.approx(60.0, abs=1e-6)
    # A step that would increase tilt is rejected (→ 0); one that reduces is kept.
    assert clamp_rotation_to_max_tilt(over, 0, 10.0, 1, 44.0) == pytest.approx(0.0)
    assert clamp_rotation_to_max_tilt(over, 0, -10.0, 1, 44.0) == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# clamp_inplane_rotation
# ---------------------------------------------------------------------------

# SAMPLE_ANCHORING has u along LR, v along AP → normal along DV (slicing axis 2);
# the two in-plane axes are then (0, 1), where u is axis-aligned (spin 0).
_INPLANE_SLICING_AXIS = 2


def test_clamp_inplane_under_limit_unchanged():
    angle = math.radians(20.0)
    result = clamp_inplane_rotation(
        SAMPLE_ANCHORING, angle, _INPLANE_SLICING_AXIS, max_inplane_deg=45.0
    )
    assert result == pytest.approx(angle)


def test_clamp_inplane_shortens_overshoot_to_limit():
    result = clamp_inplane_rotation(
        SAMPLE_ANCHORING, math.radians(80.0), _INPLANE_SLICING_AXIS, max_inplane_deg=45.0
    )
    assert 0.0 < result < math.radians(80.0)
    spun = rotate_anchoring(SAMPLE_ANCHORING, result)
    assert _spin_deg(spun, _INPLANE_SLICING_AXIS) <= 45.0 + 1e-3
    assert result == pytest.approx(math.radians(45.0), abs=1e-3)


def test_clamp_inplane_already_over_limit_only_reduces():
    over = rotate_anchoring(SAMPLE_ANCHORING, math.radians(60.0))
    assert _spin_deg(over, _INPLANE_SLICING_AXIS) == pytest.approx(60.0, abs=1e-6)
    # Increasing the spin is rejected (→ 0); reducing it is allowed.
    assert clamp_inplane_rotation(
        over, math.radians(10.0), _INPLANE_SLICING_AXIS, 45.0
    ) == pytest.approx(0.0)
    assert clamp_inplane_rotation(
        over, math.radians(-10.0), _INPLANE_SLICING_AXIS, 45.0
    ) == pytest.approx(math.radians(-10.0))
