#!/usr/bin/env python3
"""Regenerate the packaged Allen structure-set membership table.

This is a one-off build tool — it is *not* shipped or imported at runtime. It
writes ``src/verso/resources/region_sets.json``, the membership table VERSO's
mid/coarse aggregation reads (see ``.claude/quantification.md`` §5 and
``verso.engine.quantification.aggregate``). Committing this script keeps the
provenance visible and lets the table be refreshed if Allen changes the sets.

Two Allen structure sets define the aggregation levels (matching
``wholeBrain_PNN_analysis/wholebrain_tools/aba.py``):

* ``167587189`` — 316 mid-ontology structures      -> level ``"mid"``
* ``687527670`` — 12 major divisions               -> level ``"coarse"``

Fiber tracts (id ``1009``) are appended to **both** levels so fibre-tract regions
aggregate to a real bucket instead of ``unassigned``.

Usage (needs AllenSDK)::

    uv run --with allensdk python scripts/generate_region_sets.py

If AllenSDK is unavailable, pass ``--http`` to fetch the same data from the Allen
RMA HTTP API instead (stdlib only)::

    python scripts/generate_region_sets.py --http
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MID_SET_ID = 167587189
COARSE_SET_ID = 687527670
FIBER_TRACTS_ID = 1009

_OUT = Path(__file__).resolve().parent.parent / "src" / "verso" / "resources" / "region_sets.json"


def _members_allensdk(set_id: int) -> list[int]:
    """Member region IDs of a structure set via AllenSDK (mirrors aba.py)."""
    from allensdk.api.queries.ontologies_api import OntologiesApi

    nodes = OntologiesApi().get_structures_with_sets(structure_graph_ids=1)
    return sorted({int(n["id"]) for n in nodes if set_id in n.get("structure_set_ids", [])})


def _members_http(set_id: int) -> list[int]:
    """Member region IDs of a structure set via the Allen RMA HTTP API (stdlib)."""
    import urllib.request

    url = (
        "http://api.brain-map.org/api/v2/data/query.json?"
        "criteria=model::Structure,rma::criteria,"
        f"structure_sets[id$in{set_id}],"
        "rma::options[num_rows$eqall][only$eq'structures.id']"
    )
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    if not data.get("success"):
        raise RuntimeError(f"Allen API query failed for set {set_id}: {data.get('msg')}")
    return sorted({int(r["id"]) for r in data["msg"]})


def build(use_http: bool) -> dict:
    members = _members_http if use_http else _members_allensdk

    def with_fiber_tracts(ids: list[int]) -> list[int]:
        return sorted(set(ids) | {FIBER_TRACTS_ID})

    return {
        "_comment": (
            "Allen structure-set membership for VERSO aggregation. "
            "Regenerate with scripts/generate_region_sets.py. "
            "Fiber tracts (1009) appended to both levels."
        ),
        "mid": {"set_id": MID_SET_ID, "members": with_fiber_tracts(members(MID_SET_ID))},
        "coarse": {"set_id": COARSE_SET_ID, "members": with_fiber_tracts(members(COARSE_SET_ID))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http", action="store_true", help="Use the Allen RMA HTTP API instead of AllenSDK."
    )
    parser.add_argument("-o", "--out", type=Path, default=_OUT, help="Output JSON path.")
    args = parser.parse_args()

    data = build(args.http)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")
    print(
        f"Wrote {args.out}\n"
        f"  mid:    {len(data['mid']['members'])} members\n"
        f"  coarse: {len(data['coarse']['members'])} members"
    )


if __name__ == "__main__":
    main()
