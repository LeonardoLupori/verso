"""11-value packing and series propagation / interpolation.

The heaviest, most stateful block of the anchoring package: the
midpoint/unit-vector/stretch packing used for interpolation, the
series-propagation algorithm that fills in not-yet-registered sections from
their neighbours, and the model-aware wrappers that apply propagated
anchorings to a list of sections. See :mod:`verso.engine.anchoring` for the
vector-format spec.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from verso.engine.anchoring.core import anchoring_to_vectors, is_anchored, vectors_to_anchoring

if TYPE_CHECKING:
    from verso.engine.model.project import Section


def _in_plane_axes(interpolation_axis: int) -> tuple[int, int]:
    """Return ``(u_default, v_default)`` axis indices for the given slicing axis.

    For a slicing axis ``k``, the two in-plane atlas voxel axes are the other
    two indices sorted ascending — the lower becomes ``u``'s natural axis, the
    higher becomes ``v``'s. Examples:

    - ``k = 1`` (coronal, AP) → ``(0, 2)``  i.e. u along ML, v along DV.
    - ``k = 0`` (sagittal, ML) → ``(1, 2)`` i.e. u along AP, v along DV.
    - ``k = 2`` (horizontal, DV) → ``(0, 1)`` i.e. u along ML, v along AP.
    """
    if interpolation_axis not in (0, 1, 2):
        raise ValueError(f"interpolation_axis must be 0, 1, or 2, got {interpolation_axis}")
    others = sorted(i for i in (0, 1, 2) if i != interpolation_axis)
    return others[0], others[1]


def _dims_in_anchoring_order(atlas_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return atlas dims indexed by anchoring voxel axis (ML=0, AP=1, DV=2).

    BrainGlobe stores annotation shape as ``(AP, DV, LR)`` while VERSO's
    anchoring math addresses axes in ``(ML, AP, DV)`` order. This helper
    converts so callers can index by axis number directly.
    """
    ap_dim, dv_dim, lr_dim = atlas_shape
    return (lr_dim, ap_dim, dv_dim)


def series_default_anchoring(
    image_width: int,
    image_height: int,
    max_width: int,
    max_height: int,
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int,
    voxel: float | None = None,
) -> list[float]:
    """Create a centered anchoring sized to fit within the atlas by image stretch.

    The atlas-plane scale is initialized from image dimensions, not from
    display aspect ratio alone. For each series a common horizontal stretch
    ``atlas_u_dim / max_image_width`` and vertical stretch
    ``atlas_v_dim / max_image_height`` is used across the series. Each section
    then gets plane vectors proportional to its own registration image size.
    Here ``u``/``v`` map to the two in-plane axes derived from
    :func:`_in_plane_axes`.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("maximum image dimensions must be positive")

    k = interpolation_axis
    u_axis, v_axis = _in_plane_axes(k)
    dims = _dims_in_anchoring_order(atlas_shape)
    u_dim = float(dims[u_axis])
    v_dim = float(dims[v_axis])

    axis_voxel = float(dims[k]) / 2.0 if voxel is None else float(voxel)

    h_stretch = u_dim / float(max_width)
    v_stretch = v_dim / float(max_height)
    u_span = h_stretch * float(image_width)
    v_span = v_stretch * float(image_height)

    origin = [0.0, 0.0, 0.0]
    origin[k] = axis_voxel
    origin[u_axis] = (u_dim - u_span) / 2.0
    origin[v_axis] = (v_dim - v_span) / 2.0

    u_vec = [0.0, 0.0, 0.0]
    u_vec[u_axis] = u_span
    v_vec = [0.0, 0.0, 0.0]
    v_vec[v_axis] = v_span

    return [*origin, *u_vec, *v_vec]


def unpack_series_anchoring(
    anchoring: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Unpack an anchoring into midpoint, unit vectors, and stretches for interpolation."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    o, u, v = anchoring_to_vectors(anchoring)
    midpoint = o + (u + v) / 2.0

    u_len = float(np.linalg.norm(u))
    v_len = float(np.linalg.norm(v))
    if u_len <= 0 or v_len <= 0:
        raise ValueError("anchoring vectors must be non-zero")

    u_unit = u / u_len
    v_unit = v / v_len

    # Orthonormalize v against u so interpolation blends well-defined unit frames.
    u_unit = u_unit / np.linalg.norm(u_unit)
    v_unit = v_unit - u_unit * float(np.dot(u_unit, v_unit))
    v_unit = v_unit / np.linalg.norm(v_unit)

    return [
        float(midpoint[0]),
        float(midpoint[1]),
        float(midpoint[2]),
        float(u_unit[0]),
        float(u_unit[1]),
        float(u_unit[2]),
        float(v_unit[0]),
        float(v_unit[1]),
        float(v_unit[2]),
        u_len / float(image_width),
        v_len / float(image_height),
    ]


def pack_series_anchoring(
    unpacked: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Pack midpoint/unit-vector/stretch values back into a 9-element anchoring."""
    if len(unpacked) != 11:
        raise ValueError(f"unpacked anchoring must have 11 elements, got {len(unpacked)}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    midpoint = np.asarray(unpacked[0:3], dtype=np.float64)
    u_unit = np.asarray(unpacked[3:6], dtype=np.float64)
    v_unit = np.asarray(unpacked[6:9], dtype=np.float64)
    u = u_unit * float(unpacked[9]) * float(image_width)
    v = v_unit * float(unpacked[10]) * float(image_height)
    o = midpoint - (u + v) / 2.0
    return vectors_to_anchoring(o, u, v)


def propagate_series_anchorings(
    image_sizes: list[tuple[int, int]],
    slice_indices: list[int],
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int,
    stored_anchorings: list[list[float] | None] | None = None,
    reverse_axis: bool = False,
    center_proposals: bool = True,
) -> list[list[float]]:
    """Propagate anchorings along ``interpolation_axis`` for a series of sections.

    Sections with a stored anchoring act as fixed control points; every other
    section's plane is linearly interpolated (or extrapolated, at the series
    ends) between its nearest controls.

    Args:
        image_sizes: Per-section registration image ``(width, height)``.
        slice_indices: Per-section slice indices used for interpolation.
        atlas_shape: BrainGlobe annotation shape ``(AP, DV, LR)``.
        interpolation_axis: Anchoring voxel axis along which the series runs
            (0 = ML / LR, 1 = AP, 2 = DV).
        stored_anchorings: Optional stored/user anchorings. ``None`` entries
            receive propagated anchorings.
        reverse_axis: If true, reverse the initial proposal direction along the
            slicing axis while leaving section order unchanged.
        center_proposals: If true, generated proposals are recentered on the
            atlas midpoint of the two in-plane axes while stored anchorings
            remain unchanged.

    Returns:
        One packed anchoring per section.
    """
    if len(image_sizes) != len(slice_indices):
        raise ValueError("image_sizes and slice_indices must have the same length")
    if stored_anchorings is not None and len(stored_anchorings) != len(image_sizes):
        raise ValueError("stored_anchorings must match image_sizes length")
    if not image_sizes:
        return []

    k = interpolation_axis
    u_axis, v_axis = _in_plane_axes(k)

    order = sorted(range(len(slice_indices)), key=lambda i: (slice_indices[i], i))
    if order != list(range(len(slice_indices))):
        sorted_anchorings = propagate_series_anchorings(
            image_sizes=[image_sizes[i] for i in order],
            slice_indices=[slice_indices[i] for i in order],
            atlas_shape=atlas_shape,
            interpolation_axis=k,
            stored_anchorings=(
                [stored_anchorings[i] for i in order] if stored_anchorings is not None else None
            ),
            reverse_axis=reverse_axis,
            center_proposals=center_proposals,
        )
        restored: list[list[float] | None] = [None] * len(slice_indices)
        for sorted_idx, original_idx in enumerate(order):
            restored[original_idx] = sorted_anchorings[sorted_idx]
        return [anchoring for anchoring in restored if anchoring is not None]

    dims = _dims_in_anchoring_order(atlas_shape)
    axis_dim = dims[k]
    u_dim = dims[u_axis]
    v_dim = dims[v_axis]
    max_w = max(w for w, _ in image_sizes)
    max_h = max(h for _, h in image_sizes)
    if max_w <= 0 or max_h <= 0:
        raise ValueError("image dimensions must be positive")

    stored_anchorings = stored_anchorings or [None] * len(image_sizes)
    stored_indices = [i for i, anchoring in enumerate(stored_anchorings) if is_anchored(anchoring)]

    def default_unpacked(axis_voxel: float) -> list[float]:
        midpoint = [0.0, 0.0, 0.0]
        midpoint[k] = float(axis_voxel)
        midpoint[u_axis] = float(u_dim) / 2.0
        midpoint[v_axis] = float(v_dim) / 2.0
        u_unit = [0.0, 0.0, 0.0]
        u_unit[u_axis] = 1.0
        v_unit = [0.0, 0.0, 0.0]
        v_unit[v_axis] = 1.0
        return [
            *midpoint,
            *u_unit,
            *v_unit,
            float(u_dim) / float(max_w),
            float(v_dim) / float(max_h),
        ]

    unpacked_by_index: dict[int, list[float]] = {}
    for i in stored_indices:
        anchoring = stored_anchorings[i]
        assert anchoring is not None
        w, h = image_sizes[i]
        unpacked_by_index[i] = unpack_series_anchoring(anchoring, w, h)

    stored_by_slice_index: dict[int, int] = {}
    for i in stored_indices:
        stored_by_slice_index.setdefault(slice_indices[i], i)

    duplicate_index_positions: list[int] = []
    for i, slice_idx in enumerate(slice_indices):
        if i in stored_indices or slice_idx not in stored_by_slice_index:
            continue
        stored_idx = stored_by_slice_index[slice_idx]
        unpacked_by_index[i] = list(unpacked_by_index[stored_idx])
        duplicate_index_positions.append(i)

    anchor_indices = sorted(stored_indices + duplicate_index_positions)

    controls: list[int] = []
    if not stored_indices:
        first_voxel = 0.0 if reverse_axis else float(axis_dim - 1)
        last_voxel = float(axis_dim - 1) if reverse_axis else 0.0
        unpacked_by_index[0] = default_unpacked(first_voxel)
        controls.append(0)
        if len(image_sizes) > 1:
            unpacked_by_index[len(image_sizes) - 1] = default_unpacked(last_voxel)
            controls.append(len(image_sizes) - 1)
    elif len(stored_indices) == 1:
        idx = stored_indices[0]
        first_control = anchor_indices[0]
        last_control = anchor_indices[-1]
        stored = unpacked_by_index[idx]
        first_voxel = 0.0 if reverse_axis else float(axis_dim - 1)
        last_voxel = float(axis_dim - 1) if reverse_axis else 0.0
        if first_control != 0:
            first = list(stored)
            first[k] = first_voxel
            unpacked_by_index[0] = first
            controls.append(0)
        controls.extend(anchor_indices)
        if last_control != len(image_sizes) - 1:
            last = list(stored)
            last[k] = last_voxel
            unpacked_by_index[len(image_sizes) - 1] = last
            controls.append(len(image_sizes) - 1)
    else:
        controls.extend(anchor_indices)
        if anchor_indices[0] != 0:
            unpacked_by_index[0] = _regression_extrapolated_unpacked(
                stored_indices, unpacked_by_index, slice_indices, slice_indices[0]
            )
            controls.insert(0, 0)
        if anchor_indices[-1] != len(image_sizes) - 1:
            last_idx = len(image_sizes) - 1
            unpacked_by_index[last_idx] = _regression_extrapolated_unpacked(
                stored_indices, unpacked_by_index, slice_indices, slice_indices[last_idx]
            )
            controls.append(last_idx)

    controls = sorted(set(controls))
    propagated = [None] * len(image_sizes)

    for i in controls:
        propagated[i] = unpacked_by_index[i]

    if len(controls) == 1:
        propagated = [list(unpacked_by_index[controls[0]]) for _ in image_sizes]
    else:
        for left, right in itertools.pairwise(controls):
            left_index = slice_indices[left]
            right_index = slice_indices[right]
            left_u = unpacked_by_index[left]
            right_u = unpacked_by_index[right]
            for i in range(left, right + 1):
                denom = right_index - left_index
                t = 0.0 if denom == 0 else (slice_indices[i] - left_index) / denom
                propagated[i] = [a + t * (b - a) for a, b in zip(left_u, right_u, strict=False)]

    # Strip in-plane rotation (rotation around the slicing axis) from
    # proposals while preserving position along the slicing axis, physical
    # tilt (the slicing-axis component of each unit vector), and stretch.
    _stored_set = set(stored_indices)
    for i, row in enumerate(propagated):
        if row is None or i in _stored_set:
            continue
        u_tilt = row[3 + k]
        v_tilt = row[6 + k]
        row[3 + v_axis] = 0.0
        row[6 + u_axis] = 0.0
        row[3 + u_axis] = float(np.sqrt(max(0.0, 1.0 - u_tilt * u_tilt)))
        row[6 + v_axis] = float(np.sqrt(max(0.0, 1.0 - v_tilt * v_tilt)))

    packed: list[list[float]] = []
    for i, (unpacked, (w, h)) in enumerate(zip(propagated, image_sizes, strict=False)):
        if unpacked is None:
            raise RuntimeError("series propagation left a section without anchoring")
        if center_proposals and i not in stored_indices:
            unpacked = list(unpacked)
            unpacked[u_axis] = float(u_dim) / 2.0
            unpacked[v_axis] = float(v_dim) / 2.0
        packed.append(pack_series_anchoring(unpacked, w, h))
    return packed


def _regression_extrapolated_unpacked(
    stored_indices: list[int],
    unpacked_by_index: dict[int, list[float]],
    slice_indices: list[int],
    target_index: int,
) -> list[float]:
    """Linear-regression endpoint estimate for a series end with no nearby control."""
    xs = np.asarray([slice_indices[i] for i in stored_indices], dtype=np.float64)
    out: list[float] = []
    for component in range(11):
        ys = np.asarray([unpacked_by_index[i][component] for i in stored_indices])
        x_mean = float(xs.mean())
        y_mean = float(ys.mean())
        denom = float(((xs - x_mean) ** 2).sum())
        slope = 0.0 if denom == 0.0 else float(((xs - x_mean) * (ys - y_mean)).sum() / denom)
        out.append(y_mean + slope * (float(target_index) - x_mean))
    return out


# ---------------------------------------------------------------------------
# Anchoring interpolation (model-aware wrappers)
# ---------------------------------------------------------------------------


def interpolate_anchorings(
    sections: list,
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int = 1,
    reverse_axis: bool = False,
    center_proposals: bool = True,
) -> None:
    """Propagate anchorings for non-stored sections from their neighbours.

    Handles the no-stored-anchoring and one-stored-anchoring cases, where
    endpoint controls are synthesized (from ``atlas_shape``) before linear
    interpolation.
    """
    from verso.engine.model.alignment import AlignmentStatus

    ordered = sorted(sections, key=lambda s: s.slice_index)
    slice_indices = [section.slice_index for section in ordered]

    stored_indices: list[int] = []
    for idx, section in enumerate(ordered):
        if section.alignment.status == AlignmentStatus.COMPLETE and is_anchored(
            section.alignment.stored_anchoring
        ):
            stored_indices.append(idx)

    stored_anchorings_for_series = [
        list(section.alignment.stored_anchoring) if idx in stored_indices else None
        for idx, section in enumerate(ordered)
    ]
    propagated_anchorings = propagate_series_anchorings(
        image_sizes=[section.resolution_thumbnail_wh for section in ordered],
        slice_indices=slice_indices,
        atlas_shape=atlas_shape,
        interpolation_axis=interpolation_axis,
        stored_anchorings=stored_anchorings_for_series,
        reverse_axis=reverse_axis,
        center_proposals=center_proposals,
    )

    for section, anchoring in zip(ordered, propagated_anchorings, strict=False):
        if section.alignment.status == AlignmentStatus.COMPLETE:
            continue
        section.alignment.current_anchoring = anchoring
        section.alignment.status = AlignmentStatus.IN_PROGRESS
        # Mark these as auto-generated proposals (same tag
        # initialize_default_anchorings uses).  Without it, a later
        # re-interpolation treats these IN_PROGRESS sections as manual
        # edits and skips them, so a newly-saved keyframe's angle never
        # propagates here.
        section.alignment.source = "series_interpolation"


def initialize_default_anchorings(
    sections: list,
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int = 1,
    reverse_axis: bool = False,
) -> None:
    """Seed not-yet-aligned sections with default series proposals.

    Like :func:`interpolate_anchorings`, but *preserves* existing manual work:
    a section keeps its current plane unless it is still an untouched
    ``"series_interpolation"`` proposal or has no plane at all. A section that
    was saved or hand-edited is never overwritten — including a same-index
    sibling of a stored section, whose own manual plane would otherwise be
    silently reset to mirror the sibling.

    Args:
        sections: The project's sections (in series order).
        atlas_shape: BrainGlobe annotation shape ``(AP, DV, LR)``.
        interpolation_axis: Anchoring voxel axis the series runs along.
        reverse_axis: Reverse the default proposal direction along that axis.
    """
    from verso.engine.model.alignment import AlignmentStatus

    stored_anchorings = []
    for section in sections:
        is_complete = section.alignment.status == AlignmentStatus.COMPLETE
        stored = section.alignment.stored_anchoring
        stored_anchorings.append(list(stored) if is_complete and is_anchored(stored) else None)
    propagated = propagate_series_anchorings(
        image_sizes=[section.resolution_thumbnail_wh for section in sections],
        slice_indices=[section.slice_index for section in sections],
        atlas_shape=atlas_shape,
        interpolation_axis=interpolation_axis,
        stored_anchorings=stored_anchorings,
        reverse_axis=reverse_axis,
        center_proposals=True,
    )

    for section, anchoring, anch in zip(sections, propagated, stored_anchorings, strict=False):
        if anch is not None:
            continue
        # Keep a section's own manual edit; only (re)seed sections that are
        # still an untouched default proposal or have no plane at all. A
        # same-index sibling of a stored section is not special-cased here:
        # forcing it to mirror the stored plane would clobber a plane the user
        # had hand-placed but not yet saved.
        if (
            section.alignment.is_anchored
            and section.alignment.status != AlignmentStatus.NOT_STARTED
            and section.alignment.source != "series_interpolation"
        ):
            continue
        section.alignment.current_anchoring = anchoring
        if section.alignment.status == AlignmentStatus.NOT_STARTED:
            section.alignment.status = AlignmentStatus.IN_PROGRESS
        section.alignment.source = "series_interpolation"


def reset_in_progress_to_default_proposals(
    sections: list[Section],
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int = 1,
    reverse_axis: bool = False,
    include_complete: bool = False,
) -> int:
    """Clear editable suggestions and regenerate default series proposals."""
    from verso.engine.model.alignment import AlignmentStatus

    stored_anchorings = []
    for section in sections:
        is_stored = not include_complete and section.alignment.status == AlignmentStatus.COMPLETE
        if not is_stored:
            stored_anchorings.append(None)
            continue
        stored = section.alignment.stored_anchoring
        stored_anchorings.append(list(stored) if is_anchored(stored) else None)
    propagated = propagate_series_anchorings(
        image_sizes=[section.resolution_thumbnail_wh for section in sections],
        slice_indices=[section.slice_index for section in sections],
        atlas_shape=atlas_shape,
        interpolation_axis=interpolation_axis,
        stored_anchorings=stored_anchorings,
        reverse_axis=reverse_axis,
        center_proposals=True,
    )

    changed = 0
    for section, anchoring, stored in zip(sections, propagated, stored_anchorings, strict=False):
        if stored is not None:
            continue
        section.alignment.set_auto_proposal(anchoring, source="series_interpolation")
        section.warp.reset()
        changed += 1

    return changed
