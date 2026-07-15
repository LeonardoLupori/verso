"""Per-region, per-channel intensity accumulation.

Mean and total (sum) only — no median (plan §4.1). Both come from two running
``np.bincount`` sums per channel, so nothing stores per-region pixel lists and the
result is exact. The same :class:`IntensityAccumulator` serves the pooled and the
per-slice paths (one accumulator each) and is reused by the area-annotation
analysis with a different scope mask.
"""

from __future__ import annotations

import numpy as np


def match_to_raw(
    labels: np.ndarray,
    scope: np.ndarray,
    raw_shape: tuple[int, int],
    hemi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Nearest-resize ``labels``/``scope``/``hemi`` to the raw image's ``(H, W)``.

    The full-res maps are computed at the section's *stored* pixel dimensions;
    normally those equal the on-disk image's, but a rounding mismatch would break
    boolean indexing, so the label/scope/hemi maps are conformed to the raw array
    (nearest — integer maps). Raw pixels are never resized, keeping intensities exact.
    ``hemi`` is passed through untouched (still ``None``) when not splitting.
    """
    target = (int(raw_shape[0]), int(raw_shape[1]))
    if labels.shape == target:
        return labels, scope, hemi
    from PIL import Image

    from verso.engine.quantification.region_map import upsample_mask

    lab_im = Image.fromarray(labels.astype(np.int32), mode="I")
    lab_im = lab_im.resize((target[1], target[0]), Image.Resampling.NEAREST)
    labels = np.asarray(lab_im, dtype=np.int32)
    scope = upsample_mask(scope, target)
    if hemi is not None:
        hemi_im = Image.fromarray(hemi.astype(np.uint8), mode="L")
        hemi_im = hemi_im.resize((target[1], target[0]), Image.Resampling.NEAREST)
        hemi = np.asarray(hemi_im, dtype=np.uint8)
    return labels, scope, hemi


class IntensityAccumulator:
    """Pools ``n_pixels`` and per-channel totals per region across sections.

    Keys are ``(region_id, hemi)`` where ``hemi`` is ``None`` when hemispheres are
    not split, or the raw atlas hemisphere value (``1``/``2``/``0``) when they are.
    The label→``"l"``/``"r"``/``"none"`` conversion is deferred to the row builders
    (which hold the atlas), so this accumulator stays atlas-free.
    """

    def __init__(self, n_channels: int) -> None:
        self.n_channels = int(n_channels)
        self.counts: dict[tuple[int, int | None], int] = {}
        self.totals: dict[tuple[int, int | None], np.ndarray] = {}

    def add(
        self,
        labels: np.ndarray,
        scope: np.ndarray,
        raw: np.ndarray,
        hemi: np.ndarray | None = None,
    ) -> None:
        """Accumulate one section's in-scope pixels.

        Args:
            labels: ``(H, W)`` int32 region-ID map.
            scope: ``(H, W)`` bool — pixels to include.
            raw: ``(H, W, C)`` raw pixels (native dtype); ``C`` must be
                ``>= n_channels`` (extra channels ignored).
            hemi: ``(H, W)`` uint8 hemisphere map, or ``None``. When given, pixels
                are accumulated separately per hemisphere value present in scope.
        """
        if hemi is None:
            self._add_bucket(labels, scope, raw, None)
            return
        for hval in np.unique(hemi[scope]):
            self._add_bucket(labels, scope & (hemi == hval), raw, int(hval))

    def _add_bucket(
        self, labels: np.ndarray, scope: np.ndarray, raw: np.ndarray, hemi_val: int | None
    ) -> None:
        lab = labels[scope].ravel()
        if lab.size == 0:
            return
        pix = raw[scope].reshape(lab.size, -1)[:, : self.n_channels].astype(np.float64)
        maxid = int(lab.max())
        count = np.bincount(lab, minlength=maxid + 1)
        totals = np.empty((maxid + 1, self.n_channels), dtype=np.float64)
        for c in range(self.n_channels):
            totals[:, c] = np.bincount(lab, weights=pix[:, c], minlength=maxid + 1)

        for rid in np.nonzero(count)[0]:
            key = (int(rid), hemi_val)
            self.counts[key] = self.counts.get(key, 0) + int(count[rid])
            if key in self.totals:
                self.totals[key] += totals[rid]
            else:
                self.totals[key] = totals[rid].copy()

    def totals_as_lists(self) -> dict[tuple[int, int | None], list[float]]:
        """Return ``(region_id, hemi) -> [tot_ch0, …]`` for row assembly."""
        return {k: t.tolist() for k, t in self.totals.items()}
