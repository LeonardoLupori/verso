"""Row assembly and column-naming helpers for quantification outputs.

All quantification results are **lists of plain dicts** (one dict per row) so the
engine keeps no pandas dependency — ``pd.DataFrame(rows)`` reconstructs a frame
for users who want one, and :mod:`verso.engine.io.quant_export` writes them with
the stdlib ``csv`` module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verso.engine.io.annotation_io import slugify

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume


def channel_column(prefix: str, channel_name: str) -> str:
    """Column name for a per-channel statistic, e.g. ``mean_ch_DAPI``.

    The channel name is slugified (filesystem/CSV-safe) with the same helper used
    for annotation folders, so ``"Ch 0"`` → ``mean_ch_Ch_0``.
    """
    return f"{prefix}_ch_{slugify(channel_name)}"


def intensity_rows(
    counts: dict[int, int],
    totals: dict[int, list[float]],
    atlas: AtlasVolume,
    channel_names: list[str],
) -> list[dict]:
    """Build per-region intensity rows from pooled accumulators.

    Args:
        counts: ``region_id -> n_pixels``.
        totals: ``region_id -> [sum_ch0, sum_ch1, …]`` (aligned with ``channel_names``).
        atlas: For region acronym/name lookup.
        channel_names: Channel display names, in output column order.

    Returns:
        One dict per region, sorted by ``region_id``, with keys ``region_id``,
        ``acronym``, ``name``, ``n_pixels``, then ``mean_ch_<Name>`` /
        ``tot_ch_<Name>`` per channel.
    """
    rows: list[dict] = []
    for rid in sorted(counts):
        acronym, name = atlas.region_meta(rid)
        n = counts[rid]
        row: dict = {"region_id": rid, "acronym": acronym, "name": name, "n_pixels": n}
        tot = totals[rid]
        for c, cname in enumerate(channel_names):
            total = float(tot[c])
            row[channel_column("mean", cname)] = (total / n) if n else 0.0
            row[channel_column("tot", cname)] = total
        rows.append(row)
    return rows


def dots_region_rows(
    counts: dict[int, int],
    n_dots: dict[int, int],
    atlas: AtlasVolume,
) -> list[dict]:
    """Build the per-region dots table (counts + per-pixel density).

    Args:
        counts: ``region_id -> n_pixels`` (region footprint within the mask).
        n_dots: ``region_id -> dot count``.
        atlas: For region acronym/name lookup.

    Returns:
        One dict per region (union of the two key sets), sorted by ``region_id``,
        with ``region_id, acronym, name, n_pixels, n_dots, dots_density``
        (``dots_density = n_dots / n_pixels``, 0 when the region has no pixels).
    """
    rows: list[dict] = []
    for rid in sorted(set(counts) | set(n_dots)):
        acronym, name = atlas.region_meta(rid)
        n_px = counts.get(rid, 0)
        nd = n_dots.get(rid, 0)
        rows.append(
            {
                "region_id": rid,
                "acronym": acronym,
                "name": name,
                "n_pixels": n_px,
                "n_dots": nd,
                "dots_density": (nd / n_px) if n_px else 0.0,
            }
        )
    return rows
