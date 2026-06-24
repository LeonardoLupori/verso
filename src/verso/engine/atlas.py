"""Atlas volume loading, slicing, and color mapping via brainglobe-atlasapi.

Coordinate mapping (QuickNII voxel → brainglobe Allen Mouse 25µm):
  QuickNII  (x=LR, y=AP, z=DV)  →  annotation[ceil(y), ceil(z), floor(x)]
  Voxel selection matches VisuAlign/QUINT (floor in QuickNII convention); see
  ``_quicknii_floor_indices`` for why AP/DV use ceil and LR uses floor.
  The annotation volume has shape (AP=528, DV=320, LR=456) for allen_mouse_25um.
"""

from __future__ import annotations

import numpy as np

from verso.engine.registration import make_atlas_sample_grid


def _quicknii_floor_indices(
    lr_c: np.ndarray, ap_c: np.ndarray, dv_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel indices matching VisuAlign/QUINT (and PyNutil) sampling exactly.

    VisuAlign's ``getInt32Slice`` truncates (``floor``) the voxel coordinate in
    *QuickNII* convention.  AP and DV are array-reversed there relative to
    BrainGlobe (see ``registration._to_quicknii_convention``), so flooring a
    reversed axis is equivalent to taking the **ceil** of the BrainGlobe
    coordinate::

        ann_ap = (ap_max - 1) - floor((ap_max - 1) - ap_bg) = ceil(ap_bg)

    LR is shared (not reversed), so it **floors** directly.  Sampling this way
    makes the atlas labels the user aligns against in VERSO identical (verified
    100% on the stock ABA cutlas) to what VisuAlign/PyNutil quantify, instead of
    the ~0.5-voxel boundary offset a plain ``round`` introduces.

    Returns the ``(lr, ap, dv)`` index arrays (still float; caller bounds-checks
    and casts).
    """
    return np.floor(lr_c), np.ceil(ap_c), np.ceil(dv_c)


class AtlasVolume:
    """Wraps a brainglobe atlas for 2D slicing and color mapping."""

    def __init__(self, atlas_name: str) -> None:
        from brainglobe_atlasapi import BrainGlobeAtlas

        self._bg = BrainGlobeAtlas(atlas_name, check_latest=False)
        self.atlas_name = atlas_name
        self.resolution_um: float = float(self._bg.resolution[0])
        self._annotation: np.ndarray = self._bg.annotation  # (AP, DV, LR)
        self._reference: np.ndarray = self._bg.reference  # (AP, DV, LR)
        ref_max = float(self._reference.max())
        self._reference_scale: float = 255.0 / ref_max if ref_max > 0 else 1.0
        self._color_dict: dict[int, tuple[int, int, int]] = self._build_color_dict()

    def _build_color_dict(self) -> dict[int, tuple[int, int, int]]:
        d: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
        for label_id, info in self._bg.structures.items():
            rgb = info.get("rgb_triplet", [128, 128, 128])
            d[int(label_id)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return d

    # ------------------------------------------------------------------
    # Internal sampling — returns labels + in-bounds mask

    def _sample(
        self, anchoring: list[float], out_w: int, out_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the annotation along the cut plane.

        Returns:
            labels:    (H, W) int32  — annotation labels (in-bounds pixels only)
            in_bounds: (H, W) bool   — True where the voxel is inside the atlas volume
        """
        grid = make_atlas_sample_grid(anchoring, out_w, out_h)  # (H, W, 3)
        # Keep floats for bounds check — converting first can overflow int32 for extreme anchorings.
        # Use VisuAlign/QUINT-matching voxel selection (see _quicknii_floor_indices).
        lr_f, ap_f, dv_f = _quicknii_floor_indices(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2])

        ap_max, dv_max, lr_max = self._annotation.shape
        in_bounds: np.ndarray = (
            (ap_f >= 0)
            & (ap_f < ap_max)
            & (dv_f >= 0)
            & (dv_f < dv_max)
            & (lr_f >= 0)
            & (lr_f < lr_max)
        )

        # Now safe to clip and cast — all values are within [0, dim-1]
        ap = np.clip(ap_f, 0, ap_max - 1).astype(np.int32)
        dv = np.clip(dv_f, 0, dv_max - 1).astype(np.int32)
        lr = np.clip(lr_f, 0, lr_max - 1).astype(np.int32)
        return self._annotation[ap, dv, lr], in_bounds

    # ------------------------------------------------------------------
    # Public slice methods

    def sample_labels(
        self, anchoring: list[float], out_w: int, out_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the annotation along the cut plane at the requested resolution.

        Public wrapper around :meth:`_sample` for export pipelines that need a
        raw label map (e.g. to extract sub-pixel contours via marching squares).

        Returns:
            labels:    (H, W) int32 — annotation labels (in-bounds pixels only)
            in_bounds: (H, W) bool — True where the voxel is inside the volume
        """
        return self._sample(anchoring, out_w, out_h)

    def slice_annotation(self, anchoring: list[float], out_w: int, out_h: int) -> np.ndarray:
        """Slice the annotation volume → RGBA uint8 (H, W, 4).

        Alpha is:
          255 — within brain (labeled region)
            0 — outside the atlas volume entirely
           25 — within atlas volume but outside annotated brain
               (ensures the cut plane is always visible as a faint shape)
        """
        labels, in_bounds = self._sample(anchoring, out_w, out_h)

        unique_ids, inverse = np.unique(labels, return_inverse=True)
        colors = np.array(
            [self._color_dict.get(int(uid), (128, 128, 128)) for uid in unique_ids],
            dtype=np.uint8,
        )
        rgb = colors[inverse].reshape(out_h, out_w, 3)
        # Give background-within-atlas pixels a neutral gray so they're visible
        rgb[in_bounds & (labels == 0)] = [80, 80, 80]

        alpha = np.where(~in_bounds, 0, np.where(labels == 0, 25, 255)).astype(np.uint8)

        return np.dstack([rgb, alpha])

    def colorize_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map an integer label map → ``(H, W, 3)`` uint8 RGB using Allen colors.

        Each region ID is looked up in the structure colour table (unknown IDs
        fall back to neutral gray; label ``0`` is black). Vectorised via
        ``np.unique(return_inverse=True)`` so the lookup runs once per distinct
        ID rather than per pixel.

        Args:
            labels: Integer region-ID map of any shape ``(H, W)``.

        Returns:
            uint8 ``(H, W, 3)`` RGB array.
        """
        unique_ids, inverse = np.unique(labels, return_inverse=True)
        colors = np.array(
            [self._color_dict.get(int(uid), (128, 128, 128)) for uid in unique_ids],
            dtype=np.uint8,
        )
        return colors[inverse].reshape(*labels.shape, 3)

    def slice_outline(
        self,
        anchoring: list[float],
        out_w: int,
        out_h: int,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Slice the annotation as region-boundary outlines → RGBA (H, W, 4).

        Edges are detected between regions with different labels where at least
        one neighbour belongs to the annotated brain.  Out-of-atlas pixels are
        fully transparent.

        Args:
            color: RGB line color. Defaults to white ``(255, 255, 255)``.
        """
        labels, in_bounds = self._sample(anchoring, out_w, out_h)
        out_h2, out_w2 = labels.shape
        brain = (labels > 0) & in_bounds

        edges = np.zeros((out_h2, out_w2), dtype=bool)

        # Horizontal edges (between col i and i+1) — mark left pixel only (1px)
        diff_h = labels[:, :-1] != labels[:, 1:]
        keep_h = diff_h & (brain[:, :-1] | brain[:, 1:])
        edges[:, :-1] |= keep_h

        # Vertical edges (between row i and i+1) — mark top pixel only (1px)
        diff_v = labels[:-1, :] != labels[1:, :]
        keep_v = diff_v & (brain[:-1, :] | brain[1:, :])
        edges[:-1, :] |= keep_v

        rgba = np.zeros((out_h2, out_w2, 4), dtype=np.uint8)
        rgba[edges & in_bounds] = [*color, 220]
        return rgba

    # ------------------------------------------------------------------
    # Region lookup

    def get_region_info(
        self, anchoring: list[float], s: float, t: float
    ) -> tuple[str, tuple[int, int, int]]:
        """Return (name, rgb) for the atlas region at normalised section position (s, t).

        Returns:
            Tuple of human-readable region name and its RGB colour.
            Outside-atlas → ("Outside atlas", (20, 20, 20)).
            Outside-brain → ("Outside brain", (40, 40, 40)).
        """
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        voxel = o + s * u + t * v

        # Match VisuAlign/QUINT voxel selection (see _quicknii_floor_indices) so the
        # region named under the cursor is the one VisuAlign/PyNutil would report.
        lr_f, ap_f, dv_f = _quicknii_floor_indices(voxel[0], voxel[1], voxel[2])
        ap, dv, lr = int(ap_f), int(dv_f), int(lr_f)

        ap_max, dv_max, lr_max = self._annotation.shape
        if not (0 <= ap < ap_max and 0 <= dv < dv_max and 0 <= lr < lr_max):
            return "Outside atlas", (20, 20, 20)

        label = int(self._annotation[ap, dv, lr])
        if label == 0:
            return "Outside brain", (40, 40, 40)

        info = self._bg.structures.get(label, {})
        name = str(info.get("name", f"Region {label}"))
        color = self._color_dict.get(label, (128, 128, 128))
        return name, color

    def get_region_name(self, anchoring: list[float], s: float, t: float) -> str:
        """Return the atlas region name at normalised section position (s, t)."""
        name, _ = self.get_region_info(anchoring, s, t)
        return name

    # ------------------------------------------------------------------
    # Reference slice (navigator views)

    def slice_reference(self, anchoring: list[float], out_w: int, out_h: int) -> np.ndarray:
        """Slice the MRI/Nissl reference volume → RGB uint8 (H, W, 3)."""
        grid = make_atlas_sample_grid(anchoring, out_w, out_h)
        lr_f, ap_f, dv_f = _quicknii_floor_indices(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2])
        ap = np.clip(ap_f.astype(int), 0, self._reference.shape[0] - 1)
        dv = np.clip(dv_f.astype(int), 0, self._reference.shape[1] - 1)
        lr = np.clip(lr_f.astype(int), 0, self._reference.shape[2] - 1)
        gray = (
            (self._reference[ap, dv, lr].astype(np.float32) * self._reference_scale)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return np.stack([gray, gray, gray], axis=-1)

    def slice_reference_rgba(self, anchoring: list[float], out_w: int, out_h: int) -> np.ndarray:
        """Slice the MRI/Nissl reference volume → RGBA uint8 (H, W, 4).

        Alpha is 255 within the atlas bounds and 0 outside — label-based
        transparency is not applied here because unannotated regions (fiber
        tracts, ventricles) are the dark features the template is meant to show.
        """
        labels, in_bounds = self._sample(anchoring, out_w, out_h)
        grid = make_atlas_sample_grid(anchoring, out_w, out_h)
        lr_f, ap_f, dv_f = _quicknii_floor_indices(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2])
        ap = np.clip(ap_f.astype(int), 0, self._reference.shape[0] - 1)
        dv = np.clip(dv_f.astype(int), 0, self._reference.shape[1] - 1)
        lr = np.clip(lr_f.astype(int), 0, self._reference.shape[2] - 1)
        gray = (
            (self._reference[ap, dv, lr].astype(np.float32) * self._reference_scale)
            .clip(0, 255)
            .astype(np.uint8)
        )
        rgb = np.stack([gray, gray, gray], axis=-1)
        alpha = np.where(~in_bounds, 0, 255).astype(np.uint8)
        return np.dstack([rgb, alpha])

    def default_anchoring(
        self,
        axis: int = 1,
        aspect_ratio: float = 1.0,
    ) -> list[float]:
        """Return a centered anchoring perpendicular to ``axis`` for this atlas.

        The plane is constructed in the two in-plane axes (derived from
        :func:`verso.engine.registration._in_plane_axes`) with ``u`` spanning
        the full in-plane width and ``v`` sized to ``aspect_ratio``.
        """
        from verso.engine.registration import _in_plane_axes

        u_axis, v_axis = _in_plane_axes(axis)
        ap_dim, dv_dim, lr_dim = self._annotation.shape
        # BrainGlobe shape (AP, DV, LR) indexed by QuickNII axis (ML=0, AP=1, DV=2).
        qn_dims = (lr_dim, ap_dim, dv_dim)
        u_dim = float(qn_dims[u_axis])
        v_dim = float(qn_dims[v_axis])

        u_span = u_dim
        v_span = u_span / aspect_ratio if aspect_ratio > 0 else v_dim
        v_span = min(v_span, v_dim)

        origin = [0.0, 0.0, 0.0]
        origin[axis] = float(qn_dims[axis]) / 2.0
        origin[u_axis] = 0.0
        origin[v_axis] = (v_dim - v_span) / 2.0

        u_vec = [0.0, 0.0, 0.0]
        u_vec[u_axis] = u_span
        v_vec = [0.0, 0.0, 0.0]
        v_vec[v_axis] = v_span

        return [*origin, *u_vec, *v_vec]

    def voxel_to_mm(self, voxel: float) -> float:
        return voxel * self.resolution_um / 1000.0

    def mm_to_voxel(self, mm: float) -> float:
        return mm * 1000.0 / self.resolution_um

    def extent_mm_along(self, axis: int) -> float:
        """Return the atlas extent in mm along the given QuickNII voxel axis."""
        ap_dim, dv_dim, lr_dim = self._annotation.shape
        qn_dims = (lr_dim, ap_dim, dv_dim)
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        return qn_dims[axis] * self.resolution_um / 1000.0

    # ------------------------------------------------------------------
    # Orthogonal navigator views

    def get_orthogonal_slice(self, axis: int, idx: int) -> np.ndarray:
        """Return a uint8 RGB array for a navigator axis-aligned view."""
        ref = self._reference  # (AP=528, DV=320, LR=456)
        if axis == 0:
            idx = int(np.clip(idx, 0, ref.shape[2] - 1))
            gray = ref[:, :, idx].T.astype(np.float32)
        elif axis == 1:
            idx = int(np.clip(idx, 0, ref.shape[0] - 1))
            gray = ref[idx, :, :].astype(np.float32)
        else:
            idx = int(np.clip(idx, 0, ref.shape[1] - 1))
            gray = ref[:, idx, :].astype(np.float32)

        hi = float(gray.max())
        if hi > 0:
            gray = (gray / hi * 255).clip(0, 255)
        return np.stack([gray.astype(np.uint8)] * 3, axis=-1)

    def cut_center(self, anchoring: list[float]) -> np.ndarray:
        """Return the (LR, AP, DV) center of the cut plane in atlas voxels."""
        o = np.array(anchoring[:3])
        u = np.array(anchoring[3:6])
        v = np.array(anchoring[6:9])
        return o + u / 2.0 + v / 2.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._annotation.shape  # type: ignore[return-value]
