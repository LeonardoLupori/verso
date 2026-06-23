#!/usr/bin/env python3
"""Build the packaged curated anchor-point resource for auto control points.

This is a one-off build/curation tool — it is *not* shipped at runtime. It reads
the raw hand-traced anchor lines (3D polylines in Allen CCF 25 µm voxel space,
axis order ``[z=AP, y=DV, x=ML]``) and produces the runtime resource
``src/verso/resources/anchor_points.json`` by:

1. Mirroring every non-midline line across the ML midline so both hemispheres
   are represented. Lines whose name contains ``mid`` lie on the midline and are
   kept as-is.
2. Spline-interpolating + smoothing each line into a dense polyline, so the
   runtime only has to intersect dense polylines with the section plane (no
   spline fitting at runtime).

Baking mirroring + smoothing here keeps the runtime engine code trivial.

Usage:
    uv run python scripts/build_anchor_points.py [RAW_JSON] [-o OUT_JSON] [-s SMOOTHING]

``RAW_JSON`` defaults to ``../scratch_elastix/data/anchor_points.json`` relative
to this script's repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep

# Smoothing passed to splprep (matches the prototype's default).
DEFAULT_SMOOTHING = 20.0

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW = _REPO_ROOT.parent / "scratch_elastix" / "data" / "anchor_points.json"
_DEFAULT_OUT = _REPO_ROOT / "src" / "verso" / "resources" / "anchor_points.json"


def interpolate_line(points: list[list[float]], smoothing: float) -> np.ndarray:
    """Spline-interpolate clicked points (sorted by Z) into a dense curve.

    points : (N, 3) of [z, y, x]; returns (M, 3) float samples along the curve.
    Mirrors ``scratch_elastix/annotate.py::interpolate_line``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(pts) == 0:
        return pts.reshape(0, 3)
    pts = pts[np.argsort(pts[:, 0])]
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)
    pts = pts[keep]
    if len(pts) == 1:
        return pts.copy()

    seg_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    n_samples = int(np.clip(round(seg_len * 2), 50, 5000))
    k = min(3, len(pts) - 1)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], k=k, s=smoothing)
        u = np.linspace(0.0, 1.0, n_samples)
        samples = np.array(splev(u, tck)).T
    except Exception:
        samples = pts
    return samples


def mirror_lines(lines: dict[str, list], shape: list[int]) -> dict[str, list]:
    """Duplicate + ML-mirror every non-midline line. Midline lines kept as-is."""
    ml_max = shape[2] - 1
    out: dict[str, list] = {}
    for name, pts in lines.items():
        out[name] = pts
        if "mid" not in name.lower():
            out[f"{name}_mirror"] = [[z, y, ml_max - x] for z, y, x in pts]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", nargs="?", default=str(_DEFAULT_RAW), help="raw anchor_points.json")
    ap.add_argument("-o", "--out", default=str(_DEFAULT_OUT), help="output resource path")
    ap.add_argument("-s", "--smoothing", type=float, default=DEFAULT_SMOOTHING)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    shape = raw["shape"]  # [AP, DV, ML]
    lines = mirror_lines(raw["lines"], shape)

    dense: dict[str, list[list[float]]] = {}
    for name, pts in lines.items():
        if len(pts) < 2:
            continue
        samples = interpolate_line(pts, smoothing=args.smoothing)
        dense[name] = np.round(samples, 2).tolist()

    out = {
        "axis_order": "z(AP),y(DV),x(ML)",
        "resolution_um": raw.get("resolution_um", 25),
        "shape": shape,
        "lines": dense,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    n_pts = sum(len(v) for v in dense.values())
    print(f"Wrote {len(dense)} lines ({n_pts} points) -> {out_path}")


if __name__ == "__main__":
    main()
