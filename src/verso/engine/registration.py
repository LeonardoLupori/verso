"""Affine registration utilities — anchoring matrix math.

The anchoring vector is a 9-element list:
    [ox, oy, oz, ux, uy, uz, vx, vy, vz]

Origin ``o`` and direction vectors ``u``, ``v`` are in atlas *voxel* space.
For a point at normalized section coordinates (s, t) ∈ [0, 1]²:

    atlas_voxel = o + s·u + t·v

This matches the QuickNII format exactly.  All functions here operate on the
anchoring vector directly so they stay independent of any atlas library.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Anchoring vector decomposition
# ---------------------------------------------------------------------------

def anchoring_to_vectors(
    anchoring: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 9-element anchoring list into origin and direction vectors.

    Args:
        anchoring: [ox, oy, oz, ux, uy, uz, vx, vy, vz]

    Returns:
        Tuple ``(o, u, v)`` each as shape-(3,) float64 arrays.
    """
    a = np.asarray(anchoring, dtype=np.float64)
    if a.shape != (9,):
        raise ValueError(f"anchoring must have 9 elements, got {len(anchoring)}")
    return a[0:3], a[3:6], a[6:9]


def vectors_to_anchoring(
    o: np.ndarray, u: np.ndarray, v: np.ndarray
) -> list[float]:
    """Pack origin and direction vectors back into a 9-element anchoring list."""
    return np.concatenate([o, u, v]).tolist()


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def normalized_to_atlas(s: float, t: float, anchoring: list[float]) -> np.ndarray:
    """Map normalized section coords (s, t) → atlas voxel coords (x, y, z).

    Args:
        s: Normalized x-coordinate along section width, in [0, 1].
        t: Normalized y-coordinate along section height, in [0, 1].
        anchoring: 9-element anchoring vector.

    Returns:
        Shape-(3,) array of atlas voxel coordinates.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    return o + s * u + t * v


def atlas_to_normalized(
    xyz: np.ndarray | list[float], anchoring: list[float]
) -> tuple[float, float]:
    """Map atlas voxel coords → normalized section coords (s, t).

    Solves the least-squares system  xyz − o = s·u + t·v.

    Args:
        xyz: Atlas voxel position, shape (3,).
        anchoring: 9-element anchoring vector.

    Returns:
        Tuple (s, t).  Values outside [0, 1] indicate the point is outside the
        section boundary.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    A = np.column_stack([u, v])  # (3, 2)
    b = np.asarray(xyz, dtype=np.float64) - o
    result, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1])


# ---------------------------------------------------------------------------
# Anchoring manipulation
# ---------------------------------------------------------------------------

def set_position_along_axis(
    anchoring: list[float],
    voxel: float,
    axis: int,
) -> list[float]:
    """Return a new anchoring with the origin shifted to ``voxel`` along ``axis``.

    This is the primary control used in the alignment view: sliding the section
    along the slicing axis (typically AP for coronal series, ML for sagittal,
    DV for horizontal) moves ``o[axis]`` while keeping ``u`` and ``v`` unchanged.

    Args:
        anchoring: Current 9-element anchoring vector.
        voxel: New position along ``axis`` in atlas voxel units.
        axis: Index of the slicing axis in QuickNII voxel space
            (0 = ML / LR, 1 = AP, 2 = DV).

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    o = o.copy()
    o[axis] = voxel
    return vectors_to_anchoring(o, u, v)


def anchoring_center(anchoring: list[float]) -> np.ndarray:
    """Return the center point of an anchoring plane in atlas voxel space."""
    o, u, v = anchoring_to_vectors(anchoring)
    return o + (u + v) / 2.0


def set_center_position_along_axis(
    anchoring: list[float],
    voxel: float,
    axis: int,
) -> list[float]:
    """Return a new anchoring whose plane center lies at ``voxel`` along ``axis``.

    QuickNII's 11-value registration form stores the section midpoint, so axis
    navigation should move the plane center rather than the anchoring origin.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    center = o + (u + v) / 2.0
    o = o.copy()
    o[axis] += voxel - center[axis]
    return vectors_to_anchoring(o, u, v)


def rotate_anchoring(
    anchoring: list[float],
    angle_rad: float,
    pivot_s: float = 0.5,
    pivot_t: float = 0.5,
) -> list[float]:
    """Rotate the section plane around a pivot point in normalized coords.

    Rotates the u and v vectors in the plane defined by u and v (in-plane
    rotation only — does not change the AP position of the pivot).

    Args:
        anchoring: Current 9-element anchoring vector.
        angle_rad: Counter-clockwise rotation angle in radians.
        pivot_s: Pivot s-coordinate in normalized section space (default centre).
        pivot_t: Pivot t-coordinate in normalized section space (default centre).

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    u_new = cos_a * u - sin_a * v
    v_new = sin_a * u + cos_a * v

    # Shift origin so the pivot point stays fixed in atlas space.
    pivot_atlas = o + pivot_s * u + pivot_t * v
    o_new = pivot_atlas - pivot_s * u_new - pivot_t * v_new

    return vectors_to_anchoring(o_new, u_new, v_new)


def scale_anchoring(
    anchoring: list[float],
    scale_s: float,
    scale_t: float | None = None,
    pivot_s: float = 0.5,
    pivot_t: float = 0.5,
) -> list[float]:
    """Scale the section plane around a pivot point.

    Args:
        anchoring: Current 9-element anchoring vector.
        scale_s: Scale factor along the u direction (section width).
        scale_t: Scale factor along the v direction (section height).
            Defaults to ``scale_s`` for uniform scaling.
        pivot_s: Pivot s-coordinate in normalised section space.
        pivot_t: Pivot t-coordinate in normalised section space.

    Returns:
        New 9-element anchoring vector.
    """
    if scale_t is None:
        scale_t = scale_s

    o, u, v = anchoring_to_vectors(anchoring)

    pivot_atlas = o + pivot_s * u + pivot_t * v
    u_new = u * scale_s
    v_new = v * scale_t
    o_new = pivot_atlas - pivot_s * u_new - pivot_t * v_new

    return vectors_to_anchoring(o_new, u_new, v_new)


def flip_anchoring_horizontal(anchoring: list[float]) -> list[float]:
    """Mirror an anchoring horizontally in section coordinates.

    A horizontal display flip changes section coordinates as
    ``s_flipped = 1 - s_original``. The equivalent anchoring is:
        ``o' = o + u``, ``u' = -u``, ``v' = v``.

    The transform is its own inverse, so the same function can be used when
    toggling a stored alignment into or out of flipped display space.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    return vectors_to_anchoring(o + u, -u, v)


def flip_anchoring_vertical(anchoring: list[float]) -> list[float]:
    """Mirror an anchoring vertically in section coordinates.

    A vertical display flip changes section coordinates as
    ``t_flipped = 1 - t_original``. The equivalent anchoring is:
        ``o' = o + v``, ``u' = u``, ``v' = -v``.

    The transform is its own inverse, so the same function can be used when
    toggling a stored alignment into or out of flipped display space.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    return vectors_to_anchoring(o + v, u, -v)


def _display_space_anchoring(section) -> list[float]:
    """Return anchoring in display space (exactly as the user positioned it).

    ``stored_anchoring`` and ``anchoring`` are both in display space (the new
    invariant). This is the value written to QuickNII/VisuAlign exports so that
    the saved coordinates match what the user saw.
    """
    stored = section.alignment.stored_anchoring
    if stored and any(v != 0.0 for v in stored):
        return list(stored)
    return list(section.alignment.anchoring)



# ---------------------------------------------------------------------------
# Pixel ↔ Normalised (convenience wrappers for the GUI)
# ---------------------------------------------------------------------------

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


def _quicknii_dims(atlas_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return atlas dims indexed by QuickNII voxel axis (ML=0, AP=1, DV=2).

    BrainGlobe stores annotation shape as ``(AP, DV, LR)`` while VERSO's
    anchoring math addresses axes in QuickNII order ``(ML, AP, DV)``. This
    helper converts so callers can index by axis number directly.
    """
    ap_dim, dv_dim, lr_dim = atlas_shape
    return (lr_dim, ap_dim, dv_dim)


def quicknii_default_anchoring(
    image_width: int,
    image_height: int,
    max_width: int,
    max_height: int,
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int,
    voxel: float | None = None,
) -> list[float]:
    """Create a centered anchoring using QuickNII stretch semantics.

    QuickNII initializes the atlas-plane scale from image dimensions, not from
    display aspect ratio alone. For each series it uses a common horizontal
    stretch ``atlas_u_dim / max_image_width`` and vertical stretch
    ``atlas_v_dim / max_image_height`` across the series. Each section then
    gets plane vectors proportional to its own registration image size. Here
    ``u``/``v`` map to the two in-plane axes derived from
    :func:`_in_plane_axes`.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("maximum image dimensions must be positive")

    k = interpolation_axis
    u_axis, v_axis = _in_plane_axes(k)
    qn_dims = _quicknii_dims(atlas_shape)
    u_dim = float(qn_dims[u_axis])
    v_dim = float(qn_dims[v_axis])

    axis_voxel = float(qn_dims[k]) / 2.0 if voxel is None else float(voxel)

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


def quicknii_unpack_anchoring(
    anchoring: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Unpack a QuickNII anchoring into midpoint, unit vectors, and stretches."""
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

    # Match QuickNII's orthonormalization step.
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


def quicknii_pack_anchoring(
    unpacked: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Pack QuickNII midpoint/unit-vector/stretch values into anchoring."""
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


def quicknii_series_anchorings(
    image_sizes: list[tuple[int, int]],
    serial_numbers: list[int],
    atlas_shape: tuple[int, int, int],
    interpolation_axis: int,
    stored_anchorings: list[list[float] | None] | None = None,
    reverse_axis: bool = False,
    center_proposals: bool = True,
) -> list[list[float]]:
    """Propagate anchorings along ``interpolation_axis`` using QuickNII semantics.

    Args:
        image_sizes: Per-section registration image ``(width, height)``.
        serial_numbers: Per-section serial numbers used for interpolation.
        atlas_shape: BrainGlobe annotation shape ``(AP, DV, LR)``.
        interpolation_axis: QuickNII voxel axis along which the series runs
            (0 = ML / LR, 1 = AP, 2 = DV).
        stored_anchorings: Optional stored/user anchorings. ``None`` entries
            receive propagated anchorings.
        reverse_axis: If true, reverse the initial proposal direction along the
            slicing axis while leaving section order unchanged.
        center_proposals: If true, generated proposals are recentered on the
            atlas midpoint of the two in-plane axes while stored anchorings
            remain unchanged.

    Returns:
        One packed QuickNII anchoring per section.
    """
    if len(image_sizes) != len(serial_numbers):
        raise ValueError("image_sizes and serial_numbers must have the same length")
    if stored_anchorings is not None and len(stored_anchorings) != len(image_sizes):
        raise ValueError("stored_anchorings must match image_sizes length")
    if not image_sizes:
        return []

    k = interpolation_axis
    u_axis, v_axis = _in_plane_axes(k)

    order = sorted(range(len(serial_numbers)), key=lambda i: (serial_numbers[i], i))
    if order != list(range(len(serial_numbers))):
        sorted_anchorings = quicknii_series_anchorings(
            image_sizes=[image_sizes[i] for i in order],
            serial_numbers=[serial_numbers[i] for i in order],
            atlas_shape=atlas_shape,
            interpolation_axis=k,
            stored_anchorings=(
                [stored_anchorings[i] for i in order]
                if stored_anchorings is not None
                else None
            ),
            reverse_axis=reverse_axis,
            center_proposals=center_proposals,
        )
        restored: list[list[float] | None] = [None] * len(serial_numbers)
        for sorted_idx, original_idx in enumerate(order):
            restored[original_idx] = sorted_anchorings[sorted_idx]
        return [anchoring for anchoring in restored if anchoring is not None]

    qn_dims = _quicknii_dims(atlas_shape)
    axis_dim = qn_dims[k]
    u_dim = qn_dims[u_axis]
    v_dim = qn_dims[v_axis]
    max_w = max(w for w, _ in image_sizes)
    max_h = max(h for _, h in image_sizes)
    if max_w <= 0 or max_h <= 0:
        raise ValueError("image dimensions must be positive")

    stored_anchorings = stored_anchorings or [None] * len(image_sizes)
    stored_indices = [
        i for i, anchoring in enumerate(stored_anchorings)
        if anchoring is not None and any(val != 0.0 for val in anchoring)
    ]

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
        unpacked_by_index[i] = quicknii_unpack_anchoring(anchoring, w, h)

    stored_by_serial: dict[int, int] = {}
    for i in stored_indices:
        stored_by_serial.setdefault(serial_numbers[i], i)

    duplicate_serial_indices: list[int] = []
    for i, serial in enumerate(serial_numbers):
        if i in stored_indices or serial not in stored_by_serial:
            continue
        stored_idx = stored_by_serial[serial]
        unpacked_by_index[i] = list(unpacked_by_index[stored_idx])
        duplicate_serial_indices.append(i)

    anchor_indices = sorted(stored_indices + duplicate_serial_indices)

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
            unpacked_by_index[0] = _quicknii_regressed_unpacked(
                stored_indices, unpacked_by_index, serial_numbers, serial_numbers[0]
            )
            controls.insert(0, 0)
        if anchor_indices[-1] != len(image_sizes) - 1:
            last_idx = len(image_sizes) - 1
            unpacked_by_index[last_idx] = _quicknii_regressed_unpacked(
                stored_indices, unpacked_by_index, serial_numbers, serial_numbers[last_idx]
            )
            controls.append(last_idx)

    controls = sorted(set(controls))
    propagated = [None] * len(image_sizes)

    for i in controls:
        propagated[i] = unpacked_by_index[i]

    if len(controls) == 1:
        propagated = [list(unpacked_by_index[controls[0]]) for _ in image_sizes]
    else:
        for left, right in zip(controls, controls[1:]):
            left_sno = serial_numbers[left]
            right_sno = serial_numbers[right]
            left_u = unpacked_by_index[left]
            right_u = unpacked_by_index[right]
            for i in range(left, right + 1):
                denom = right_sno - left_sno
                t = 0.0 if denom == 0 else (serial_numbers[i] - left_sno) / denom
                propagated[i] = [
                    a + t * (b - a)
                    for a, b in zip(left_u, right_u)
                ]

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
    for i, (unpacked, (w, h)) in enumerate(zip(propagated, image_sizes)):
        if unpacked is None:
            raise RuntimeError("QuickNII propagation left a section without anchoring")
        if center_proposals and i not in stored_indices:
            unpacked = list(unpacked)
            unpacked[u_axis] = float(u_dim) / 2.0
            unpacked[v_axis] = float(v_dim) / 2.0
        packed.append(quicknii_pack_anchoring(unpacked, w, h))
    return packed


def _quicknii_regressed_unpacked(
    stored_indices: list[int],
    unpacked_by_index: dict[int, list[float]],
    serial_numbers: list[int],
    target_serial: int,
) -> list[float]:
    """Linear-regression endpoint estimate matching QuickNII's fallback path."""
    xs = np.asarray([serial_numbers[i] for i in stored_indices], dtype=np.float64)
    out: list[float] = []
    for component in range(11):
        ys = np.asarray([unpacked_by_index[i][component] for i in stored_indices])
        x_mean = float(xs.mean())
        y_mean = float(ys.mean())
        denom = float(((xs - x_mean) ** 2).sum())
        slope = 0.0 if denom == 0.0 else float(((xs - x_mean) * (ys - y_mean)).sum() / denom)
        out.append(y_mean + slope * (float(target_serial) - x_mean))
    return out


def pixel_to_normalized(
    px: float, py: float, width: int, height: int
) -> tuple[float, float]:
    """Convert working-resolution pixel coords to normalised section coords."""
    return px / width, py / height


def normalized_to_pixel(
    s: float, t: float, width: int, height: int
) -> tuple[float, float]:
    """Convert normalised section coords to working-resolution pixel coords."""
    return s * width, t * height


# ---------------------------------------------------------------------------
# Anchoring interpolation
# ---------------------------------------------------------------------------

def interpolate_anchorings(
    sections: list,
    atlas_shape: tuple[int, int, int] | None = None,
    interpolation_axis: int = 1,
    reverse_axis: bool = False,
    center_proposals: bool = True,
) -> None:
    """Propagate anchorings for non-stored sections using QuickNII semantics.

    When ``atlas_shape`` is available this follows QuickNII's
    ``MgmtPanel.dointerpolate`` path, including the no-stored-anchoring and
    one-stored-anchoring cases where endpoint controls are synthesized before
    linear interpolation. Without an atlas shape, only the multi-anchor path can
    be resolved because QuickNII's synthesized endpoint controls depend on atlas
    dimensions.
    """
    from verso.engine.io.image_io import registration_dimensions
    from verso.engine.model.alignment import AlignmentStatus

    usable = []
    for section in sections:
        try:
            w, h = registration_dimensions(section)
        except Exception:
            continue
        if w > 0 and h > 0:
            usable.append((section, w, h))
    if not usable:
        return

    k = interpolation_axis
    u_axis, v_axis = _in_plane_axes(k)

    order = sorted(range(len(usable)), key=lambda i: (usable[i][0].serial_number, i))
    sorted_usable = [usable[i] for i in order]
    serial_numbers = [section.serial_number for section, _, _ in sorted_usable]

    unpacked_by_index: dict[int, list[float]] = {}
    stored_indices: list[int] = []
    for idx, (section, w, h) in enumerate(sorted_usable):
        if section.alignment.status == AlignmentStatus.COMPLETE:
            display = _display_space_anchoring(section)
            if any(v != 0.0 for v in display):
                unpacked_by_index[idx] = quicknii_unpack_anchoring(display, w, h)
                stored_indices.append(idx)

    if atlas_shape is not None:
        stored_anchorings_for_series = [
            _display_space_anchoring(section) if idx in stored_indices else None
            for idx, (section, _, _) in enumerate(sorted_usable)
        ]
        propagated_anchorings = quicknii_series_anchorings(
            image_sizes=[(w, h) for _, w, h in sorted_usable],
            serial_numbers=serial_numbers,
            atlas_shape=atlas_shape,
            interpolation_axis=k,
            stored_anchorings=stored_anchorings_for_series,
            reverse_axis=reverse_axis,
            center_proposals=center_proposals,
        )

        for (section, _, _), anchoring in zip(sorted_usable, propagated_anchorings):
            if section.alignment.status == AlignmentStatus.COMPLETE:
                continue
            section.alignment.anchoring = anchoring
            section.alignment.status = AlignmentStatus.IN_PROGRESS
        return

    if len(stored_indices) < 2:
        return

    controls = list(stored_indices)
    if stored_indices[0] != 0:
        unpacked_by_index[0] = _quicknii_regressed_unpacked(
            stored_indices,
            unpacked_by_index,
            serial_numbers,
            serial_numbers[0],
        )
        controls.insert(0, 0)
    if stored_indices[-1] != len(sorted_usable) - 1:
        last_idx = len(sorted_usable) - 1
        unpacked_by_index[last_idx] = _quicknii_regressed_unpacked(
            stored_indices,
            unpacked_by_index,
            serial_numbers,
            serial_numbers[last_idx],
        )
        controls.append(last_idx)

    controls = sorted(set(controls))
    propagated: dict[int, list[float]] = {}
    for left, right in zip(controls, controls[1:]):
        left_serial = serial_numbers[left]
        right_serial = serial_numbers[right]
        left_unpacked = unpacked_by_index[left]
        right_unpacked = unpacked_by_index[right]
        for idx in range(left, right + 1):
            denom = right_serial - left_serial
            t = 0.0 if denom == 0 else (serial_numbers[idx] - left_serial) / denom
            propagated[idx] = [
                a + t * (b - a)
                for a, b in zip(left_unpacked, right_unpacked)
            ]

    _stored_set_legacy = set(stored_indices)
    for idx, row in propagated.items():
        if idx in _stored_set_legacy:
            continue
        u_tilt = row[3 + k]
        v_tilt = row[6 + k]
        row[3 + v_axis] = 0.0
        row[6 + u_axis] = 0.0
        row[3 + u_axis] = float(np.sqrt(max(0.0, 1.0 - u_tilt * u_tilt)))
        row[6 + v_axis] = float(np.sqrt(max(0.0, 1.0 - v_tilt * v_tilt)))

    for idx, unpacked in propagated.items():
        section, w, h = sorted_usable[idx]
        if section.alignment.status == AlignmentStatus.COMPLETE:
            continue
        section.alignment.anchoring = quicknii_pack_anchoring(unpacked, w, h)
        section.alignment.status = AlignmentStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# Atlas slice sampling grid
# ---------------------------------------------------------------------------

def make_atlas_sample_grid(
    anchoring: list[float],
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Build a (H, W, 3) array of atlas voxel coordinates for a 2D slice.

    The grid covers the full section plane at the given output resolution.
    Each cell ``grid[row, col]`` contains the atlas voxel (x, y, z) that
    corresponds to the normalized section coordinate
    ``(col / (W-1), row / (H-1))``.

    This grid is passed to the atlas volume sampler (in ``atlas.py``) to
    extract a 2D slice image.

    Args:
        anchoring: 9-element anchoring vector.
        out_width: Width of the desired output slice image in pixels.
        out_height: Height of the desired output slice image in pixels.

    Returns:
        Float64 array of shape (out_height, out_width, 3).
    """
    o, u, v = anchoring_to_vectors(anchoring)
    s = np.linspace(0.0, 1.0, out_width)
    t = np.linspace(0.0, 1.0, out_height)
    ss, tt = np.meshgrid(s, t)  # (H, W) each
    grid = o + ss[..., np.newaxis] * u + tt[..., np.newaxis] * v  # (H, W, 3)
    return grid
