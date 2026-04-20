"""Atlas volume loading, slicing, and color mapping via brainglobe-atlasapi.

Coordinate mapping (QuickNII voxel → brainglobe Allen Mouse 25µm):
  QuickNII  (x=LR, y=AP, z=DV)  →  annotation[round(y), round(z), round(x)]
  The annotation volume has shape (AP=528, DV=320, LR=456) for allen_mouse_25um.
"""

from __future__ import annotations

import numpy as np

from verso.engine.registration import make_atlas_sample_grid


class AtlasVolume:
    """Wraps a brainglobe atlas for 2D slicing and color mapping."""

    def __init__(self, atlas_name: str) -> None:
        from brainglobe_atlasapi import BrainGlobeAtlas
        self._bg = BrainGlobeAtlas(atlas_name, check_latest=False)
        self.atlas_name = atlas_name
        self.resolution_um: float = float(self._bg.resolution[0])
        self._annotation: np.ndarray = self._bg.annotation  # (AP, DV, LR)
        self._reference: np.ndarray = self._bg.reference    # (AP, DV, LR)
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
        # Keep floats for bounds check — converting first can overflow int32 for extreme anchorings
        ap_f = np.round(grid[:, :, 1])
        dv_f = np.round(grid[:, :, 2])
        lr_f = np.round(grid[:, :, 0])

        ap_max, dv_max, lr_max = self._annotation.shape
        in_bounds: np.ndarray = (
            (ap_f >= 0) & (ap_f < ap_max) &
            (dv_f >= 0) & (dv_f < dv_max) &
            (lr_f >= 0) & (lr_f < lr_max)
        )

        # Now safe to clip and cast — all values are within [0, dim-1]
        ap = np.clip(ap_f, 0, ap_max - 1).astype(np.int32)
        dv = np.clip(dv_f, 0, dv_max - 1).astype(np.int32)
        lr = np.clip(lr_f, 0, lr_max - 1).astype(np.int32)
        return self._annotation[ap, dv, lr], in_bounds

    # ------------------------------------------------------------------
    # Public slice methods

    def slice_annotation(
        self, anchoring: list[float], out_w: int, out_h: int
    ) -> np.ndarray:
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

        alpha = np.where(
            ~in_bounds, 0,
            np.where(labels == 0, 25, 255)
        ).astype(np.uint8)

        return np.dstack([rgb, alpha])

    def slice_outline(
        self, anchoring: list[float], out_w: int, out_h: int
    ) -> np.ndarray:
        """Slice the annotation as white region-boundary outlines → RGBA (H, W, 4).

        Edges are detected between regions with different labels where at least
        one neighbour belongs to the annotated brain.  Out-of-atlas pixels are
        fully transparent.
        """
        labels, in_bounds = self._sample(anchoring, out_w, out_h)
        out_h2, out_w2 = labels.shape
        brain = (labels > 0) & in_bounds

        edges = np.zeros((out_h2, out_w2), dtype=bool)

        # Horizontal edges (between col i and i+1)
        diff_h = labels[:, :-1] != labels[:, 1:]
        keep_h = diff_h & (brain[:, :-1] | brain[:, 1:])
        edges[:, :-1] |= keep_h
        edges[:, 1:] |= keep_h

        # Vertical edges (between row i and i+1)
        diff_v = labels[:-1, :] != labels[1:, :]
        keep_v = diff_v & (brain[:-1, :] | brain[1:, :])
        edges[:-1, :] |= keep_v
        edges[1:, :] |= keep_v

        rgba = np.zeros((out_h2, out_w2, 4), dtype=np.uint8)
        rgba[edges & in_bounds] = [255, 255, 255, 220]
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

        ap = int(round(float(voxel[1])))
        dv = int(round(float(voxel[2])))
        lr = int(round(float(voxel[0])))

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

    def get_region_name(
        self, anchoring: list[float], s: float, t: float
    ) -> str:
        """Return the atlas region name at normalised section position (s, t)."""
        name, _ = self.get_region_info(anchoring, s, t)
        return name

    # ------------------------------------------------------------------
    # Reference slice (navigator views)

    def slice_reference(
        self, anchoring: list[float], out_w: int, out_h: int
    ) -> np.ndarray:
        """Slice the MRI/Nissl reference volume → RGB uint8 (H, W, 3)."""
        grid = make_atlas_sample_grid(anchoring, out_w, out_h)
        ap = np.clip(np.round(grid[:, :, 1]).astype(int), 0, self._reference.shape[0] - 1)
        dv = np.clip(np.round(grid[:, :, 2]).astype(int), 0, self._reference.shape[1] - 1)
        lr = np.clip(np.round(grid[:, :, 0]).astype(int), 0, self._reference.shape[2] - 1)
        gray = self._reference[ap, dv, lr]
        return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    def default_anchoring(self, aspect_ratio: float = 1.0) -> list[float]:
        """Centered coronal anchoring for this atlas."""
        ap_dim, dv_dim, lr_dim = self._annotation.shape
        lr_span = float(lr_dim)
        dv_span = lr_span / aspect_ratio if aspect_ratio > 0 else float(dv_dim)
        dv_span = min(dv_span, float(dv_dim))
        oz = (dv_dim - dv_span) / 2.0
        return [0.0, ap_dim / 2.0, oz, lr_span, 0.0, 0.0, 0.0, 0.0, dv_span]

    @property
    def ap_axis(self) -> int:
        return 1

    def ap_voxel_to_mm(self, voxel: float) -> float:
        return voxel * self.resolution_um / 1000.0

    def ap_mm_to_voxel(self, mm: float) -> float:
        return mm * 1000.0 / self.resolution_um

    @property
    def ap_extent_mm(self) -> float:
        return self._annotation.shape[0] * self.resolution_um / 1000.0

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
