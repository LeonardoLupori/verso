"""Anchoring matrix math — the affine section-to-atlas plane.

The anchoring vector is a 9-element list:
    [ox, oy, oz, ux, uy, uz, vx, vy, vz]

Origin ``o`` and direction vectors ``u``, ``v`` are in atlas *voxel* space.
For a point at normalized section coordinates (s, t) ∈ [0, 1]²:

    atlas_voxel = o + s·u + t·v

This matches the QuickNII format exactly (see ``engine/io/quint_io.py`` for the
compatibility layer).  All functions here operate on the anchoring vector
directly so they stay independent of any atlas library.

The implementation is split across three submodules — kept internal; import the
names below from ``verso.engine.anchoring`` directly:

- :mod:`~verso.engine.anchoring.core` — vector decomposition, coordinate
  transforms, pixel↔normalized wrappers, and the atlas sample grid.
- :mod:`~verso.engine.anchoring.manipulate` — rigid plane manipulation
  (position/rotate/scale/tilt/flip) and the tilt/in-plane clamps.
- :mod:`~verso.engine.anchoring.series_interpolation` — 11-value packing and
  the series propagation / interpolation algorithm.
"""

from __future__ import annotations

from verso.engine.anchoring.core import (
    anchoring_center,
    anchoring_to_vectors,
    atlas_to_normalized,
    infer_interpolation_axis,
    is_anchored,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    vectors_to_anchoring,
)
from verso.engine.anchoring.manipulate import (
    clamp_inplane_rotation,
    clamp_rotation_to_max_tilt,
    flip_anchoring_horizontal,
    flip_anchoring_vertical,
    plane_tilt_deg,
    rotate_anchoring,
    scale_anchoring,
    set_center_position_along_axis,
    set_position_along_axis,
    tilt_plane_about_atlas_axis,
)
from verso.engine.anchoring.series_interpolation import (
    # ``_in_plane_axes`` is re-exported (noqa) for its cross-module importer,
    # atlas.py; see F12 for folding the axis-convention helpers into one home.
    _in_plane_axes,  # noqa: F401
    initialize_default_anchorings,
    interpolate_anchorings,
    pack_series_anchoring,
    propagate_series_anchorings,
    reset_in_progress_to_default_proposals,
    series_default_anchoring,
    unpack_series_anchoring,
)

__all__ = [
    "anchoring_center",
    "anchoring_to_vectors",
    "atlas_to_normalized",
    "clamp_inplane_rotation",
    "clamp_rotation_to_max_tilt",
    "flip_anchoring_horizontal",
    "flip_anchoring_vertical",
    "infer_interpolation_axis",
    "initialize_default_anchorings",
    "interpolate_anchorings",
    "is_anchored",
    "make_atlas_sample_grid",
    "normalized_to_atlas",
    "normalized_to_pixel",
    "pack_series_anchoring",
    "pixel_to_normalized",
    "plane_tilt_deg",
    "propagate_series_anchorings",
    "reset_in_progress_to_default_proposals",
    "rotate_anchoring",
    "scale_anchoring",
    "series_default_anchoring",
    "set_center_position_along_axis",
    "set_position_along_axis",
    "tilt_plane_about_atlas_axis",
    "unpack_series_anchoring",
    "vectors_to_anchoring",
]
