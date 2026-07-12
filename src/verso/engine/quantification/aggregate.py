"""Mid / coarse aggregation via the bundled Allen structure-set membership table.

Maps each fine region to its representative in a set by walking brainglobe's
``structure_id_path`` from the region up to the root and taking the **nearest**
member ancestor (see plan §5) — order-independent and correct even if a set were
not a strict partition. Membership comes from ``resources/region_sets.json``
(generated offline by ``scripts/generate_region_sets.py``); this module touches
only brainglobe at runtime.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from verso.engine.quantification.tables import channel_column

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume

_log = logging.getLogger(__name__)

#: Aggregation level names, coarsest last.
LEVELS = ("mid", "coarse")


def load_region_sets() -> dict[str, dict]:
    """Load the bundled ``{level: {set_id, members}}`` membership table."""
    from importlib.resources import files

    text = (files("verso.resources") / "region_sets.json").read_text(encoding="utf-8")
    return json.loads(text)


class RegionAggregator:
    """Resolves fine region IDs to their mid/coarse representatives (memoized)."""

    def __init__(self, atlas: AtlasVolume, region_sets: dict[str, dict] | None = None) -> None:
        self._atlas = atlas
        sets = region_sets if region_sets is not None else load_region_sets()
        # Ignore metadata keys like "_comment"; a level is any dict with "members".
        self._members: dict[str, set[int]] = {
            level: {int(m) for m in spec["members"]}
            for level, spec in sets.items()
            if isinstance(spec, dict) and "members" in spec
        }
        self._cache: dict[str, dict[int, int | None]] = {level: {} for level in self._members}

    def levels(self) -> list[str]:
        """Return the available level names (e.g. ``["mid", "coarse"]``)."""
        return list(self._members)

    def representative(self, level: str, region_id: int) -> int | None:
        """Nearest member ancestor of ``region_id`` at ``level`` (or ``None``).

        ``None`` means the region has no ancestor in the set (the ``unassigned``
        bucket). Region ``0`` (background) is always ``None``.
        """
        rid = int(region_id)
        cache = self._cache[level]
        if rid in cache:
            return cache[rid]
        members = self._members[level]
        # structure_id_path is root -> ... -> self; the nearest member ancestor is
        # the last (deepest) hit, so ordering never matters.
        hits = [a for a in self._atlas.structure_id_path(rid) if a in members]
        if len(hits) > 1:
            _log.warning(
                "Region %s has %d ancestors in the %s set (overlapping members?); "
                "using the nearest.",
                rid,
                len(hits),
                level,
            )
        rep = hits[-1] if hits else None
        cache[rid] = rep
        return rep


def _sorted_group_keys(keys) -> list:
    """Sort group keys with ``None`` (unassigned) last."""
    return sorted(keys, key=lambda k: (k is None, k if k is not None else 0))


def _meta(atlas: AtlasVolume, key: int | None) -> tuple[object, str, str]:
    """Return ``(region_id_out, acronym, name)`` for a group key (``None`` = unassigned)."""
    if key is None:
        return "", "unassigned", "unassigned"
    acronym, name = atlas.region_meta(key)
    return key, acronym, name


def regroup_intensity(
    counts: dict[int, int],
    totals: dict[int, list[float]],
    agg: RegionAggregator,
    level: str,
    atlas: AtlasVolume,
    channel_names: list[str],
) -> list[dict]:
    """Regroup pooled intensity accumulators to ``level`` (means recomputed)."""
    gc: dict[int | None, int] = {}
    gt: dict[int | None, np.ndarray] = {}
    for rid, n in counts.items():
        key = agg.representative(level, rid)
        gc[key] = gc.get(key, 0) + n
        arr = np.asarray(totals[rid], dtype=np.float64)
        gt[key] = gt[key] + arr if key in gt else arr.copy()

    rows: list[dict] = []
    for key in _sorted_group_keys(gc):
        n = gc[key]
        tot = gt[key]
        rid_out, acronym, name = _meta(atlas, key)
        row: dict = {"region_id": rid_out, "acronym": acronym, "name": name, "n_pixels": n}
        for c, cname in enumerate(channel_names):
            row[channel_column("mean", cname)] = (float(tot[c]) / n) if n else 0.0
            row[channel_column("tot", cname)] = float(tot[c])
        rows.append(row)
    return rows


def regroup_dots_region(
    counts: dict[int, int],
    n_dots: dict[int, int],
    agg: RegionAggregator,
    level: str,
    atlas: AtlasVolume,
) -> list[dict]:
    """Regroup the per-region dots table to ``level`` (density recomputed)."""
    gc: dict[int | None, int] = {}
    gd: dict[int | None, int] = {}
    for rid in set(counts) | set(n_dots):
        key = agg.representative(level, rid)
        gc[key] = gc.get(key, 0) + counts.get(rid, 0)
        gd[key] = gd.get(key, 0) + n_dots.get(rid, 0)

    rows: list[dict] = []
    for key in _sorted_group_keys(set(gc) | set(gd)):
        n_px = gc.get(key, 0)
        nd = gd.get(key, 0)
        rid_out, acronym, name = _meta(atlas, key)
        rows.append(
            {
                "region_id": rid_out,
                "acronym": acronym,
                "name": name,
                "n_pixels": n_px,
                "n_dots": nd,
                "dots_density": (nd / n_px) if n_px else 0.0,
            }
        )
    return rows


def add_dot_aggregation_columns(
    records: list[dict],
    agg: RegionAggregator,
    atlas: AtlasVolume,
    levels: tuple[str, ...],
) -> None:
    """Add ``<level>_region_id`` / ``<level>_acronym`` columns to per-dot records."""
    for r in records:
        rid = int(r["region_id"])
        for level in levels:
            rep = agg.representative(level, rid)
            rid_out, acronym, _ = _meta(atlas, rep)
            r[f"{level}_region_id"] = rid_out
            r[f"{level}_acronym"] = acronym
