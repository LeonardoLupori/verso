"""Dot (point-series) quantification: per-dot table + per-region counts.

Region assignment and the mask gate both read the shared full-resolution
``labels``/``scope`` maps (so dots, pixels, and counts are consistent). CCF
coordinates come from :meth:`VersoRegistration.coord_image_to_atlas` and are
re-ordered to the Allen convention (``x=AP, y=DV, z=LR`` microns).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from verso.engine.quantification.tables import channel_column

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume
    from verso.engine.model.project import Section
    from verso.engine.registration import VersoRegistration


def _circle_means(
    raw: np.ndarray, x: float, y: float, diameter_px: float, channel_idx: list[int]
) -> dict[int, float]:
    """Mean raw pixel value in a disk of ``diameter_px`` centred on ``(x, y)``.

    Returns ``{channel_idx: mean}``. Diameter is in original pixels; the default
    (1 px) reduces to the single pixel under the dot.
    """
    from skimage.draw import disk

    h, w = raw.shape[:2]
    rr, cc = disk((y, x), max(diameter_px / 2.0, 0.5), shape=(h, w))
    out: dict[int, float] = {}
    if rr.size == 0:
        rr = np.array([round(y)])
        cc = np.array([round(x)])
    for c in channel_idx:
        out[c] = float(np.mean(raw[rr, cc, c]))
    return out


def process_section_dots(
    reg: VersoRegistration,
    atlas: AtlasVolume,
    section: Section,
    points_xy: np.ndarray,
    labels: np.ndarray,
    scope: np.ndarray,
    *,
    raw: np.ndarray | None = None,
    intensity_channels: list[int] | None = None,
    channel_names: list[str] | None = None,
    dot_diameter_px: float = 1.0,
) -> tuple[list[dict], dict[int, int]]:
    """Quantify one section's dots.

    Args:
        reg: Registration facade (for CCF coordinates).
        atlas: Atlas (for region acronyms).
        section: The section the dots belong to.
        points_xy: ``(N, 2)`` original-resolution pixel coordinates.
        labels: ``(H, W)`` full-res region-ID map (on-disk frame).
        scope: ``(H, W)`` bool slice-mask scope. Dots outside it are dropped (RULE).
        raw: ``(H, W, C)`` raw pixels, required only if ``intensity_channels`` given.
        intensity_channels: Channel indices to measure ``mean_intensity`` for.
        channel_names: Channel display names (for column naming), indexed by channel.
        dot_diameter_px: Disk diameter (original px) for ``mean_intensity``.

    Returns:
        ``(per_dot_records, n_dots_by_region)`` — one dict per kept dot and the
        per-region kept-dot counts. Dots outside the slice mask (or image bounds)
        are dropped from both.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    records: list[dict] = []
    n_dots: dict[int, int] = {}
    if pts.size == 0:
        return records, n_dots

    image = Path(section.original_path).name
    h, w = labels.shape
    ccf = reg.coord_image_to_atlas(section.id, pts, space="full", units="um")  # (N,3) LR,AP,DV
    ic = intensity_channels or []

    for i in range(len(pts)):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        xi, yi = round(x), round(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue  # outside the image → cannot be inside the slice mask
        if not scope[yi, xi]:
            continue  # RULE: only dots inside the slice mask are counted
        rid = int(labels[yi, xi])
        acronym, _ = atlas.region_meta(rid)
        row: dict = {
            "x": x,
            "y": y,
            "image": image,
            "x_ccf": float(ccf[i, 1]),  # AP
            "y_ccf": float(ccf[i, 2]),  # DV
            "z_ccf": float(ccf[i, 0]),  # LR
            "region_id": rid,
            "acronym": acronym,
        }
        if ic and raw is not None:
            means = _circle_means(raw, x, y, dot_diameter_px, ic)
            for c in ic:
                cname = channel_names[c] if channel_names and c < len(channel_names) else f"Ch {c}"
                row[channel_column("mean_intensity", cname)] = means[c]
        records.append(row)
        n_dots[rid] = n_dots.get(rid, 0) + 1

    return records, n_dots


def add_region_counts(counts: dict[int, int], labels: np.ndarray, scope: np.ndarray) -> None:
    """Accumulate the region pixel footprint (within ``scope``) into ``counts``."""
    lab = labels[scope].ravel()
    if lab.size == 0:
        return
    binned = np.bincount(lab)
    for rid in np.nonzero(binned)[0]:
        r = int(rid)
        counts[r] = counts.get(r, 0) + int(binned[r])
