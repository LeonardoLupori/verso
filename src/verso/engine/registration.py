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

def set_ap_position(
    anchoring: list[float],
    ap_voxel: float,
    ap_axis: int = 2,
) -> list[float]:
    """Return a new anchoring with the origin shifted to ``ap_voxel`` along ``ap_axis``.

    This is the primary control used in the alignment view: sliding the coronal
    section along the anterior-posterior axis moves ``o[ap_axis]`` while keeping
    the u and v vectors unchanged.

    Args:
        anchoring: Current 9-element anchoring vector.
        ap_voxel: New position along the AP axis in atlas voxel units.
        ap_axis: Index of the AP axis in atlas voxel space (default 2 for Allen
            Mouse Atlas where the z-axis is AP).

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    o = o.copy()
    o[ap_axis] = ap_voxel
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


# ---------------------------------------------------------------------------
# Pixel ↔ Normalised (convenience wrappers for the GUI)
# ---------------------------------------------------------------------------

def quicknii_coronal_default_anchoring(
    image_width: int,
    image_height: int,
    max_width: int,
    max_height: int,
    atlas_shape: tuple[int, int, int],
    ap_voxel: float | None = None,
) -> list[float]:
    """Create a centered coronal anchoring using QuickNII stretch semantics.

    QuickNII initializes the atlas-plane scale from image dimensions, not from
    display aspect ratio alone. For coronal Allen-style data it uses a common
    horizontal stretch ``atlas_lr / max_image_width`` and vertical stretch
    ``atlas_dv / max_image_height`` across the series. Each section then gets
    plane vectors proportional to its own registration image size.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("maximum image dimensions must be positive")

    ap_dim, dv_dim, lr_dim = atlas_shape
    ap = float(ap_dim) / 2.0 if ap_voxel is None else float(ap_voxel)

    h_stretch = float(lr_dim) / float(max_width)
    v_stretch = float(dv_dim) / float(max_height)
    lr_span = h_stretch * float(image_width)
    dv_span = v_stretch * float(image_height)

    ox = (float(lr_dim) - lr_span) / 2.0
    oz = (float(dv_dim) - dv_span) / 2.0

    return [ox, ap, oz, lr_span, 0.0, 0.0, 0.0, 0.0, dv_span]


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


def quicknii_coronal_series_anchorings(
    image_sizes: list[tuple[int, int]],
    serial_numbers: list[int],
    atlas_shape: tuple[int, int, int],
    stored_anchorings: list[list[float] | None] | None = None,
    reverse_ap: bool = False,
) -> list[list[float]]:
    """Propagate coronal anchorings using QuickNII's series interpolation logic.

    Args:
        image_sizes: Per-section registration image ``(width, height)``.
        serial_numbers: Per-section serial numbers used for interpolation.
        atlas_shape: BrainGlobe annotation shape ``(AP, DV, LR)``.
        stored_anchorings: Optional stored/user anchorings. ``None`` entries
            receive propagated anchorings.
        reverse_ap: If true, reverse the initial AP proposal direction while
            leaving section order unchanged.

    Returns:
        One packed QuickNII anchoring per section.
    """
    if len(image_sizes) != len(serial_numbers):
        raise ValueError("image_sizes and serial_numbers must have the same length")
    if stored_anchorings is not None and len(stored_anchorings) != len(image_sizes):
        raise ValueError("stored_anchorings must match image_sizes length")
    if not image_sizes:
        return []

    ap_dim, dv_dim, lr_dim = atlas_shape
    max_w = max(w for w, _ in image_sizes)
    max_h = max(h for _, h in image_sizes)
    if max_w <= 0 or max_h <= 0:
        raise ValueError("image dimensions must be positive")

    stored_anchorings = stored_anchorings or [None] * len(image_sizes)
    stored_indices = [
        i for i, anchoring in enumerate(stored_anchorings)
        if anchoring is not None and any(v != 0.0 for v in anchoring)
    ]

    def default_unpacked(ap_voxel: float) -> list[float]:
        return [
            float(lr_dim) / 2.0,
            float(ap_voxel),
            float(dv_dim) / 2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            float(lr_dim) / float(max_w),
            float(dv_dim) / float(max_h),
        ]

    unpacked_by_index: dict[int, list[float]] = {}
    for i in stored_indices:
        anchoring = stored_anchorings[i]
        assert anchoring is not None
        w, h = image_sizes[i]
        unpacked_by_index[i] = quicknii_unpack_anchoring(anchoring, w, h)

    controls: list[int] = []
    if not stored_indices:
        first_ap = 0.0 if reverse_ap else float(ap_dim - 1)
        last_ap = float(ap_dim - 1) if reverse_ap else 0.0
        unpacked_by_index[0] = default_unpacked(first_ap)
        controls.append(0)
        if len(image_sizes) > 1:
            unpacked_by_index[len(image_sizes) - 1] = default_unpacked(last_ap)
            controls.append(len(image_sizes) - 1)
    elif len(stored_indices) == 1:
        idx = stored_indices[0]
        stored = unpacked_by_index[idx]
        first_ap = 0.0 if reverse_ap else float(ap_dim - 1)
        last_ap = float(ap_dim - 1) if reverse_ap else 0.0
        if idx != 0:
            first = list(stored)
            first[1] = first_ap
            unpacked_by_index[0] = first
            controls.append(0)
        controls.append(idx)
        if idx != len(image_sizes) - 1:
            last = list(stored)
            last[1] = last_ap
            unpacked_by_index[len(image_sizes) - 1] = last
            controls.append(len(image_sizes) - 1)
    else:
        controls.extend(stored_indices)
        if stored_indices[0] != 0:
            unpacked_by_index[0] = _quicknii_regressed_unpacked(
                stored_indices, unpacked_by_index, serial_numbers, serial_numbers[0]
            )
            controls.insert(0, 0)
        if stored_indices[-1] != len(image_sizes) - 1:
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

    packed: list[list[float]] = []
    for unpacked, (w, h) in zip(propagated, image_sizes):
        if unpacked is None:
            raise RuntimeError("QuickNII propagation left a section without anchoring")
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

def interpolate_anchorings(sections: list) -> None:
    """Linearly interpolate/extrapolate anchoring for non-stored sections.

    Sections with ``AlignmentStatus.COMPLETE`` and a valid (non-zero) anchoring
    act as keyframes.  All other sections receive an interpolated anchoring and
    their status is set to ``IN_PROGRESS``.  Already-COMPLETE sections are never
    modified.

    Interpolation is performed per serial_number, matching QuickNII behaviour.

    Args:
        sections: List of :class:`~verso.engine.model.project.Section` objects
            in any order.  Modified in-place.
    """
    from verso.engine.model.alignment import AlignmentStatus

    def _valid(s) -> bool:
        return (
            s.alignment.status == AlignmentStatus.COMPLETE
            and s.alignment.anchoring
            and any(v != 0.0 for v in s.alignment.anchoring)
        )

    stored = sorted([s for s in sections if _valid(s)], key=lambda s: s.serial_number)
    if not stored:
        return

    for section in sections:
        if section.alignment.status == AlignmentStatus.COMPLETE:
            continue

        nr = section.serial_number
        before = [s for s in stored if s.serial_number <= nr]
        after = [s for s in stored if s.serial_number >= nr]

        if before and after:
            s1, s2 = before[-1], after[0]
            if s1 is s2:
                anchoring = list(s1.alignment.anchoring)
            else:
                t = (nr - s1.serial_number) / (s2.serial_number - s1.serial_number)
                anchoring = [
                    a1 + t * (a2 - a1)
                    for a1, a2 in zip(s1.alignment.anchoring, s2.alignment.anchoring)
                ]
        elif before:
            # After the last keyframe — extrapolate from last two (or copy)
            if len(before) >= 2:
                s1, s2 = before[-2], before[-1]
                dn = s2.serial_number - s1.serial_number
                slope = (nr - s2.serial_number) / dn if dn else 0.0
                anchoring = [
                    a2 + slope * (a2 - a1)
                    for a1, a2 in zip(s1.alignment.anchoring, s2.alignment.anchoring)
                ]
            else:
                anchoring = list(before[-1].alignment.anchoring)
        else:
            # Before the first keyframe — extrapolate from first two (or copy)
            if len(after) >= 2:
                s1, s2 = after[0], after[1]
                dn = s2.serial_number - s1.serial_number
                slope = (nr - s1.serial_number) / dn if dn else 0.0
                anchoring = [
                    a1 + slope * (a2 - a1)
                    for a1, a2 in zip(s1.alignment.anchoring, s2.alignment.anchoring)
                ]
            else:
                anchoring = list(after[0].alignment.anchoring)

        section.alignment.anchoring = anchoring
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
