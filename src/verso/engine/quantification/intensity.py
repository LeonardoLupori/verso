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
    labels: np.ndarray, scope: np.ndarray, raw_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-resize ``labels``/``scope`` to the raw image's ``(H, W)`` if needed.

    The full-res label map is computed at the section's *stored* pixel dimensions;
    normally those equal the on-disk image's, but a rounding mismatch would break
    boolean indexing, so the label/scope maps are conformed to the raw array
    (nearest — integer maps). Raw pixels are never resized, keeping intensities exact.
    """
    target = (int(raw_shape[0]), int(raw_shape[1]))
    if labels.shape == target:
        return labels, scope
    from PIL import Image

    from verso.engine.quantification.region_map import upsample_mask

    lab_im = Image.fromarray(labels.astype(np.int32), mode="I")
    lab_im = lab_im.resize((target[1], target[0]), Image.Resampling.NEAREST)
    return np.asarray(lab_im, dtype=np.int32), upsample_mask(scope, target)


class IntensityAccumulator:
    """Pools ``n_pixels`` and per-channel totals per region across sections."""

    def __init__(self, n_channels: int) -> None:
        self.n_channels = int(n_channels)
        self.counts: dict[int, int] = {}
        self.totals: dict[int, np.ndarray] = {}

    def add(self, labels: np.ndarray, scope: np.ndarray, raw: np.ndarray) -> None:
        """Accumulate one section's in-scope pixels.

        Args:
            labels: ``(H, W)`` int32 region-ID map.
            scope: ``(H, W)`` bool — pixels to include.
            raw: ``(H, W, C)`` raw pixels (native dtype); ``C`` must be
                ``>= n_channels`` (extra channels ignored).
        """
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
            r = int(rid)
            self.counts[r] = self.counts.get(r, 0) + int(count[r])
            if r in self.totals:
                self.totals[r] += totals[r]
            else:
                self.totals[r] = totals[r].copy()

    def totals_as_lists(self) -> dict[int, list[float]]:
        """Return ``region_id -> [tot_ch0, …]`` for row assembly."""
        return {r: t.tolist() for r, t in self.totals.items()}
