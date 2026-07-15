"""Rigid manipulation of the section plane — position, rotate, scale, tilt, flip.

These operate on the 9-element anchoring vector directly (see
:mod:`verso.engine.anchoring`) and stay independent of any atlas library.
"""

from __future__ import annotations

import numpy as np

from verso.engine.anchoring.core import anchoring_to_vectors, vectors_to_anchoring


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
        axis: Index of the slicing axis in anchoring voxel space
            (0 = ML / LR, 1 = AP, 2 = DV).

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    o = o.copy()
    o[axis] = voxel
    return vectors_to_anchoring(o, u, v)


def set_center_position_along_axis(
    anchoring: list[float],
    voxel: float,
    axis: int,
) -> list[float]:
    """Return a new anchoring whose plane center lies at ``voxel`` along ``axis``.

    The 11-value interpolation form (see :mod:`~verso.engine.anchoring.series_interpolation`)
    stores the section midpoint, so axis navigation should move the plane
    center rather than the anchoring origin.
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

    This is a *rigid* in-plane spin: ``u`` and ``v`` are rotated about the plane
    normal ``cross(u, v)`` via Rodrigues, so their lengths and the angle between
    them are preserved and the normal direction is left unchanged (tilt is
    untouched). Rotating in the raw ``u``/``v`` basis instead would distort the
    section whenever ``|u| != |v|`` (squashing it in height as it spins); this
    matches the rigid rotation the orthogonal navigator views use.

    Args:
        anchoring: Current 9-element anchoring vector.
        angle_rad: Counter-clockwise rotation angle in radians.
        pivot_s: Pivot s-coordinate in normalized section space (default centre).
        pivot_t: Pivot t-coordinate in normalized section space (default centre).

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)

    normal = np.cross(u, v)
    n_norm = float(np.linalg.norm(normal))
    if n_norm == 0.0:
        # Degenerate plane (collinear u, v): no well-defined rotation axis.
        return list(anchoring)
    n = normal / n_norm

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    def _rot(w: np.ndarray) -> np.ndarray:
        # Rodrigues about the unit normal. The ``-sin`` on the cross term keeps
        # the legacy drag direction (positive angle spins u toward -v).
        return cos_a * w - sin_a * np.cross(n, w) + (1.0 - cos_a) * float(np.dot(n, w)) * n

    u_new = _rot(u)
    v_new = _rot(v)

    # Shift origin so the pivot point stays fixed in atlas space.
    pivot_atlas = o + pivot_s * u + pivot_t * v
    o_new = pivot_atlas - pivot_s * u_new - pivot_t * v_new

    return vectors_to_anchoring(o_new, u_new, v_new)


def plane_tilt_deg(anchoring: list[float], slicing_axis: int) -> float:
    """Acute angle in degrees between the plane normal and the slicing axis.

    The plane normal is ``cross(u, v)`` and the slicing axis is the unit vector
    along ``slicing_axis`` (0=LR, 1=AP, 2=DV in anchoring voxel space).  A plane
    perpendicular to the slicing axis (no tilt) returns 0°.  In-plane rotation
    leaves the direction of ``cross(u, v)`` unchanged, so the result reflects
    tilt only — making this the reference for clamping tilt.

    Args:
        anchoring: 9-element anchoring vector.
        slicing_axis: Atlas voxel axis index the cutting series runs along.

    Returns:
        Acute tilt angle in degrees, in [0, 90].
    """
    _o, u, v = anchoring_to_vectors(anchoring)
    normal = np.cross(u, v)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return 0.0
    cos_t = min(1.0, max(0.0, abs(normal[slicing_axis]) / norm))
    return float(np.degrees(np.arccos(cos_t)))


def _rodrigues(vec: np.ndarray, axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate ``vec`` about the unit vector ``axis_unit`` by ``angle_rad``.

    Standard Rodrigues rotation with the ``+sin`` convention (a right-handed
    rotation about ``axis_unit``). Note this differs from the ``-sin`` used in
    :func:`rotate_anchoring`, which spins about the *plane normal* to match the
    legacy drag direction; here the axis is an arbitrary atlas axis, so the plain
    right-handed form is used. ``axis_unit`` must already be normalized.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return (
        c * vec
        + s * np.cross(axis_unit, vec)
        + (1.0 - c) * float(np.dot(axis_unit, vec)) * axis_unit
    )


def tilt_plane_about_atlas_axis(anchoring: list[float], axis: int, deg: float) -> list[float]:
    """Tilt the section plane by rotating it about a canonical atlas axis.

    Rotates both ``u`` and ``v`` about the atlas basis axis ``axis`` (0=LR, 1=AP,
    2=DV) by ``deg`` degrees while holding the plane *center* (``o + u/2 + v/2``)
    fixed in atlas space, then repacks the anchoring. Unlike
    :func:`rotate_anchoring` (a rigid *in-plane* spin about the plane normal that
    leaves tilt unchanged), this rotates about an external axis and so changes the
    plane's out-of-plane tilt — it is the shared body of the navigator's
    orthogonal-view rotation controls.

    Args:
        anchoring: Current 9-element anchoring vector.
        axis: Atlas voxel axis index to rotate about (0/1/2).
        deg: Rotation angle in degrees.

    Returns:
        New 9-element anchoring vector.
    """
    o, u, v = anchoring_to_vectors(anchoring)
    axis_unit = np.zeros(3)
    axis_unit[axis] = 1.0
    angle_rad = np.radians(deg)
    center = o + u / 2.0 + v / 2.0
    u_new = _rodrigues(u, axis_unit, angle_rad)
    v_new = _rodrigues(v, axis_unit, angle_rad)
    o_new = center - u_new / 2.0 - v_new / 2.0
    return vectors_to_anchoring(o_new, u_new, v_new)


def clamp_rotation_to_max_tilt(
    anchoring: list[float],
    axis: int,
    deg: float,
    slicing_axis: int,
    max_tilt_deg: float,
) -> float:
    """Shorten ``deg`` so the plane never tilts past ``max_tilt_deg``.

    Tilt (relative to ``slicing_axis``) is monotonic in the rotation magnitude over
    a 90° window, so the request is first capped to that window and, when it would
    overshoot, bisected for the boundary. Rotation about the slicing axis leaves
    tilt unchanged, so this is a no-op there. If the plane already exceeds the limit
    (e.g. an imported plane), only steps that *reduce* tilt are allowed.

    Args:
        anchoring: Current 9-element anchoring vector.
        axis: Atlas voxel axis index the rotation is about (0/1/2).
        deg: Requested rotation angle in degrees.
        slicing_axis: Atlas voxel axis the cutting series runs along.
        max_tilt_deg: Maximum allowed plane tilt in degrees.

    Returns:
        The clamped rotation angle in degrees.
    """

    def tilt_after(d: float) -> float:
        rotated = tilt_plane_about_atlas_axis(anchoring, axis, d)
        return plane_tilt_deg(rotated, slicing_axis)

    deg = max(-90.0, min(90.0, deg))
    if deg == 0.0 or tilt_after(deg) <= max_tilt_deg:
        return deg
    if tilt_after(0.0) > max_tilt_deg:
        return deg if tilt_after(deg) < tilt_after(0.0) else 0.0
    lo, hi = 0.0, deg
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if tilt_after(mid) <= max_tilt_deg:
            lo = mid
        else:
            hi = mid
    return lo


def clamp_inplane_rotation(
    anchoring: list[float],
    angle_rad: float,
    slicing_axis: int,
    max_inplane_deg: float,
) -> float:
    """Shorten an in-plane spin so the overlay stays within ``max_inplane_deg``.

    The spin is the signed angle of ``u`` projected onto the two in-plane atlas
    axes (0° = axis-aligned). The request is first capped to ±90° so a fast flick
    can't leap across the ±180° branch cut and land the plane upside-down; the
    boundary is then found by bisection (spin magnitude is monotonic over the
    feasible prefix of ``[0, angle_rad]``). If the plane already exceeds the limit,
    only steps that *reduce* the spin are allowed. Uses :func:`rotate_anchoring`,
    the same rigid in-plane spin the caller applies.

    Args:
        anchoring: Current 9-element anchoring vector.
        angle_rad: Requested in-plane rotation angle in radians.
        slicing_axis: Atlas voxel axis the cutting series runs along.
        max_inplane_deg: Maximum allowed spin from axis-aligned, in degrees.

    Returns:
        The clamped rotation angle in radians.
    """
    u_axis, v_axis = sorted(i for i in (0, 1, 2) if i != slicing_axis)
    angle_rad = max(-np.pi / 2.0, min(np.pi / 2.0, angle_rad))

    def spin_after(a: float) -> float:
        u = np.asarray(rotate_anchoring(anchoring, a)[3:6])
        return abs(float(np.degrees(np.arctan2(u[v_axis], u[u_axis]))))

    if angle_rad == 0.0 or spin_after(angle_rad) <= max_inplane_deg:
        return angle_rad
    if spin_after(0.0) > max_inplane_deg:
        return angle_rad if spin_after(angle_rad) < spin_after(0.0) else 0.0
    lo, hi = 0.0, angle_rad
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if spin_after(mid) <= max_inplane_deg:
            lo = mid
        else:
            hi = mid
    return lo


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
