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


def sort_region_keys(keys) -> list[tuple[int, int | None]]:
    """Sort ``(region_id, hemi)`` keys by region then hemisphere value.

    ``hemi`` is ``None`` (not splitting) or an int atlas hemisphere value; ``None``
    is ordered first so it never compares against an int.
    """
    return sorted(keys, key=lambda k: (k[0], -1 if k[1] is None else k[1]))


def _region_row_head(atlas: AtlasVolume, key: tuple[int, int | None]) -> dict:
    """Leading ``region_id``/``acronym``/``name`` (+ ``hemisphere`` when split)."""
    rid, hemi = key
    acronym, name = atlas.region_meta(rid)
    head: dict = {"region_id": rid, "acronym": acronym, "name": name}
    if hemi is not None:
        head["hemisphere"] = atlas.hemisphere_label(hemi)
    return head


def intensity_rows(
    counts: dict[tuple[int, int | None], int],
    totals: dict[tuple[int, int | None], list[float]],
    atlas: AtlasVolume,
    channel_names: list[str],
) -> list[dict]:
    """Build per-region intensity rows from pooled accumulators.

    Args:
        counts: ``(region_id, hemi) -> n_pixels``.
        totals: ``(region_id, hemi) -> [sum_ch0, sum_ch1, …]`` (aligned with
            ``channel_names``).
        atlas: For region acronym/name lookup.
        channel_names: Channel display names, in output column order.

    Returns:
        One dict per ``(region, hemisphere)`` bucket, sorted by region then
        hemisphere, with keys ``region_id``, ``acronym``, ``name``,
        (``hemisphere`` when split), ``n_pixels``, then ``mean_ch_<Name>`` /
        ``tot_ch_<Name>`` per channel.
    """
    rows: list[dict] = []
    for key in sort_region_keys(counts):
        n = counts[key]
        row = _region_row_head(atlas, key)
        row["n_pixels"] = n
        tot = totals[key]
        for c, cname in enumerate(channel_names):
            total = float(tot[c])
            row[channel_column("mean", cname)] = (total / n) if n else 0.0
            row[channel_column("tot", cname)] = total
        rows.append(row)
    return rows


def dots_region_rows(
    counts: dict[tuple[int, int | None], int],
    n_dots: dict[tuple[int, int | None], int],
    atlas: AtlasVolume,
) -> list[dict]:
    """Build the per-region dots table (counts + per-pixel density).

    Args:
        counts: ``(region_id, hemi) -> n_pixels`` (region footprint within the mask).
        n_dots: ``(region_id, hemi) -> dot count``.
        atlas: For region acronym/name lookup.

    Returns:
        One dict per ``(region, hemisphere)`` bucket (union of the two key sets),
        sorted by region then hemisphere, with ``region_id, acronym, name,
        (hemisphere,) n_pixels, n_dots, dots_density``
        (``dots_density = n_dots / n_pixels``, 0 when the bucket has no pixels).
    """
    rows: list[dict] = []
    for key in sort_region_keys(set(counts) | set(n_dots)):
        n_px = counts.get(key, 0)
        nd = n_dots.get(key, 0)
        row = _region_row_head(atlas, key)
        row["n_pixels"] = n_px
        row["n_dots"] = nd
        row["dots_density"] = (nd / n_px) if n_px else 0.0
        rows.append(row)
    return rows
