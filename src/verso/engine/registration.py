"""High-level pixel <-> atlas mapping façade for a VERSO project.

:class:`VersoRegistration` packages the engine's coordinate math into one object
that converts any pixel of an original section image into an Allen CCFv3 atlas
voxel (and back), reading only the native ``project-verso.json`` — which is
self-contained for coordinate work since it stores per-section pixel dimensions
and atlas resolution/shape.

The heavy lifting is delegated to the low-level primitives in
:mod:`verso.engine.anchoring` (affine anchoring math) and
:mod:`verso.engine.warping` (piecewise-affine Delaunay warp); this module only
composes them and applies the per-section preprocessing flips.

Coordinate conventions
----------------------
- Image points default to **full-resolution original pixels** (``space="full"``),
  i.e. pixels of the un-flipped image on disk. ``space="working"`` uses
  working-resolution (thumbnail) pixels instead.
- Preprocessing flips are applied internally: control points, masks and the
  anchoring all live in *displayed* (flipped) section space, while the image on
  disk is un-flipped, so an on-disk pixel is mirrored by the section's flips
  before the transform (and un-mirrored on the way back).
- Atlas output defaults to **voxels** (``units="voxel"``); ``"um"`` scales by the
  atlas ``resolution_um`` and ``"mm"`` by ``resolution_um / 1000``.

This module is pure Python — no PyQt/pyqtgraph — per the engine/GUI split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verso.engine.anchoring import anchoring_to_vectors
from verso.engine.model.project import Project, Section
from verso.engine.warping import (
    warp_points_atlas_to_section,
    warp_points_section_to_atlas,
)

_SPACES = ("full", "working")
_UNITS = ("voxel", "um", "mm")


@dataclass(frozen=True)
class _SectionSnapshot:
    """Per-section numeric state needed for coordinate math, resolved once."""

    id: str
    original_path: str
    work_w: int
    work_h: int
    full_w: int
    full_h: int
    o: np.ndarray  # (3,) anchoring origin, atlas voxels
    u: np.ndarray  # (3,) anchoring right vector
    v: np.ndarray  # (3,) anchoring down vector
    src_px: np.ndarray  # (N, 2) control-point atlas positions, working-res px
    dst_px: np.ndarray  # (N, 2) control-point section positions, working-res px
    flip_h: bool
    flip_v: bool
    aligned: bool  # anchoring spans a non-degenerate plane


@dataclass
class AtlasToImageResult:
    """Result of :meth:`VersoRegistration.atlas_to_image`, arrays aligned by row.

    Attributes:
        section_id: (N,) object array of the matched section id, ``""`` where no
            section footprint covers the voxel.
        xy: (N, 2) float array of image pixels on the matched section (in the
            requested ``space``); ``nan`` where uncovered.
        distance: (N,) float array of off-plane perpendicular distance in the
            requested ``units``; ``inf`` where uncovered.
        valid: (N,) bool array — inside a section footprint and, when
            ``max_distance`` is given, within it.
    """

    section_id: np.ndarray
    xy: np.ndarray
    distance: np.ndarray
    valid: np.ndarray


class VersoRegistration:
    """Convert pixels to Allen CCFv3 atlas coordinates for a VERSO project.

    Construct from the native project file::

        r = VersoRegistration("my_experiment/project-verso.json")
        xyz = r.image_to_atlas("s001", [[1200, 3400], [1500, 3600]])
        res = r.atlas_to_image(xyz)

    A slice is addressed by :attr:`Section.id`, or by the original image's file
    stem or basename.
    """

    def __init__(self, path: str | Path) -> None:
        """Load a project from its native JSON and build the coordinate snapshot.

        Older project files (pre-v1.2) are migrated on load: missing per-section
        pixel dimensions and atlas metadata are backfilled from the image files
        and brainglobe. A modern, self-contained file loads fully offline.

        Args:
            path: Path to a ``project-verso.json`` file.
        """
        from verso.engine.io.project_io import backfill_metadata

        path = Path(path)
        project = Project.load(path)
        backfill_metadata(project, path.parent)
        self._init_from_project(project)

    @classmethod
    def from_project(cls, project: Project) -> VersoRegistration:
        """Build a registration from an in-memory, fully-populated project.

        Args:
            project: A project whose section dimensions and atlas metadata are
                already populated (no backfill is performed).

        Returns:
            A ready-to-use :class:`VersoRegistration`.
        """
        self = object.__new__(cls)
        self._init_from_project(project)
        return self

    # -- construction ------------------------------------------------------

    def _init_from_project(self, project: Project) -> None:
        atlas = project.atlas
        if atlas.resolution_um <= 0 or any(d <= 0 for d in atlas.shape):
            raise ValueError(
                "Project atlas metadata is incomplete (resolution_um / shape); "
                "the project file is not self-contained for coordinate math."
            )
        self._resolution_um = float(atlas.resolution_um)
        self._atlas_shape = tuple(int(d) for d in atlas.shape)

        self._snapshots: dict[str, _SectionSnapshot] = {}
        self._ids: list[str] = []
        for section in project.sections:
            snap = self._build_snapshot(section)
            self._snapshots[snap.id] = snap
            self._ids.append(snap.id)

    @staticmethod
    def _build_snapshot(section: Section) -> _SectionSnapshot:
        work_w, work_h = section.resolution_thumbnail_wh
        full_w, full_h = section.resolution_original_wh
        if min(work_w, work_h, full_w, full_h) <= 0:
            raise ValueError(
                f"Section {section.id!r} has unpopulated pixel dimensions; the "
                f"project file is not self-contained for coordinate math."
            )
        o, u, v = anchoring_to_vectors(section.alignment.anchoring)
        cps = section.warp.control_points
        if cps:
            src_px = np.array([[cp.src_x, cp.src_y] for cp in cps], dtype=np.float64)
            dst_px = np.array([[cp.dst_x, cp.dst_y] for cp in cps], dtype=np.float64)
        else:
            src_px = np.empty((0, 2), dtype=np.float64)
            dst_px = np.empty((0, 2), dtype=np.float64)
        aligned = bool(np.linalg.norm(np.cross(u, v)) > 0.0)
        return _SectionSnapshot(
            id=section.id,
            original_path=section.original_path,
            work_w=int(work_w),
            work_h=int(work_h),
            full_w=int(full_w),
            full_h=int(full_h),
            o=o,
            u=u,
            v=v,
            src_px=src_px,
            dst_px=dst_px,
            flip_h=bool(section.preprocessing.flip_horizontal),
            flip_v=bool(section.preprocessing.flip_vertical),
            aligned=aligned,
        )

    # -- container protocol ------------------------------------------------

    def ids(self) -> list[str]:
        """Return the section ids in project order."""
        return list(self._ids)

    def __contains__(self, key: object) -> bool:
        try:
            self._resolve_slice(str(key))
        except (KeyError, ValueError):
            return False
        return True

    def __len__(self) -> int:
        return len(self._ids)

    # -- slice resolution --------------------------------------------------

    def _resolve_slice(self, key: str) -> str:
        """Resolve a slice key (id, file stem, or basename) to a section id."""
        key = str(key)
        if key in self._snapshots:
            return key
        by_stem = [s.id for s in self._snapshots.values() if Path(s.original_path).stem == key]
        by_name = [s.id for s in self._snapshots.values() if Path(s.original_path).name == key]
        matches = by_stem or by_name
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"No section matches {key!r}. Available ids: {self._ids}")
        raise KeyError(f"Slice {key!r} is ambiguous; candidate ids: {matches}")

    # -- forward: image -> atlas ------------------------------------------

    def image_to_atlas(
        self,
        slice: str,
        xy: np.ndarray | list,
        *,
        space: str = "full",
        units: str = "voxel",
        return_valid: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Map image pixels on ``slice`` to atlas coordinates.

        Args:
            slice: Section id, original-image file stem, or basename.
            xy: A single ``(2,)`` point or an ``(N, 2)`` array of image pixels.
            space: ``"full"`` (original full-resolution pixels, default) or
                ``"working"`` (working/thumbnail-resolution pixels).
            units: ``"voxel"`` (default), ``"um"`` or ``"mm"``.
            return_valid: If True, also return an ``(N,)`` boolean mask that is
                True where the pixel falls within the section frame.

        Returns:
            An ``(N, 3)`` array of atlas coordinates, or ``(coords, inside)``
            when ``return_valid`` is True.
        """
        if space not in _SPACES:
            raise ValueError(f"space must be one of {_SPACES}, got {space!r}")
        if units not in _UNITS:
            raise ValueError(f"units must be one of {_UNITS}, got {units!r}")
        snap = self._snapshots[self._resolve_slice(slice)]
        if not snap.aligned:
            raise ValueError(f"Section {snap.id!r} has no alignment; cannot map pixels.")

        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if space == "full":
            px = pts[:, 0] * snap.work_w / snap.full_w
            py = pts[:, 1] * snap.work_h / snap.full_h
        else:  # "working"
            px, py = pts[:, 0].copy(), pts[:, 1].copy()

        # Mirror on-disk (un-flipped) pixels into displayed section space.
        if snap.flip_h:
            px = snap.work_w - px
        if snap.flip_v:
            py = snap.work_h - py

        s = px / snap.work_w
        t = py / snap.work_h
        st = np.column_stack([s, t])
        if len(snap.src_px):
            uv = warp_points_section_to_atlas(
                st, snap.src_px, snap.dst_px, snap.work_w, snap.work_h
            )
        else:
            uv = st

        voxel = (
            snap.o[None, :] + uv[:, 0, None] * snap.u[None, :] + uv[:, 1, None] * snap.v[None, :]
        )
        coords = self._to_units(voxel, units)
        if return_valid:
            inside = (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
            return coords, inside
        return coords

    # -- reverse: atlas -> image ------------------------------------------

    def atlas_to_image(
        self,
        xyz: np.ndarray | list,
        *,
        space: str = "full",
        units: str = "voxel",
        max_distance: float | None = None,
    ) -> AtlasToImageResult:
        """Back-project atlas voxels to image pixels via nearest-section search.

        Sections sparsely sample the atlas, so a voxel usually lies *between*
        planes. Each voxel is matched to the nearest section whose footprint
        covers it; the matched section id is an output.

        Args:
            xyz: A single ``(3,)`` point or an ``(N, 3)`` array of atlas voxel
                coordinates.
            space: Image space of the returned pixels — ``"full"`` (default) or
                ``"working"``.
            units: Units for the reported ``distance`` (and ``max_distance``):
                ``"voxel"`` (default), ``"um"`` or ``"mm"``.
            max_distance: If given, voxels farther than this (in ``units``) from
                the matched plane are marked invalid.

        Returns:
            An :class:`AtlasToImageResult` with per-voxel arrays.
        """
        if space not in _SPACES:
            raise ValueError(f"space must be one of {_SPACES}, got {space!r}")
        if units not in _UNITS:
            raise ValueError(f"units must be one of {_UNITS}, got {units!r}")

        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = len(pts)
        best_dist = np.full(n, np.inf)
        section_id = np.full(n, "", dtype=object)
        xy = np.full((n, 2), np.nan, dtype=np.float64)

        for snap in self._snapshots.values():
            if not snap.aligned:
                continue
            normal = np.cross(snap.u, snap.v)
            norm = np.linalg.norm(normal)
            if norm == 0.0:
                continue
            normal = normal / norm

            rel = pts - snap.o[None, :]
            dist = np.abs(rel @ normal)  # perpendicular distance, voxels

            # Affine inverse onto the section plane -> atlas-overlay (u, v).
            pinv = np.linalg.pinv(np.column_stack([snap.u, snap.v]))  # (2, 3)
            uv = rel @ pinv.T  # (N, 2)
            inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)

            better = inside & (dist < best_dist)
            if not np.any(better):
                continue

            st = warp_points_atlas_to_section(
                uv[better], snap.src_px, snap.dst_px, snap.work_w, snap.work_h
            )
            px = st[:, 0] * snap.work_w
            py = st[:, 1] * snap.work_h
            # Un-mirror displayed section pixels back to on-disk orientation.
            if snap.flip_h:
                px = snap.work_w - px
            if snap.flip_v:
                py = snap.work_h - py
            if space == "full":
                px = px * snap.full_w / snap.work_w
                py = py * snap.full_h / snap.work_h

            best_dist[better] = dist[better]
            section_id[better] = snap.id
            xy[better, 0] = px
            xy[better, 1] = py

        covered = np.isfinite(best_dist)
        distance = self._scale_distance(best_dist, units)
        valid = covered.copy()
        if max_distance is not None:
            valid &= distance <= max_distance
        return AtlasToImageResult(section_id=section_id, xy=xy, distance=distance, valid=valid)

    # -- units -------------------------------------------------------------

    def _to_units(self, voxel: np.ndarray, units: str) -> np.ndarray:
        if units == "voxel":
            return voxel
        if units == "um":
            return voxel * self._resolution_um
        return voxel * self._resolution_um / 1000.0  # "mm"

    def _scale_distance(self, dist_voxel: np.ndarray, units: str) -> np.ndarray:
        if units == "voxel":
            return dist_voxel
        if units == "um":
            return dist_voxel * self._resolution_um
        return dist_voxel * self._resolution_um / 1000.0  # "mm"
