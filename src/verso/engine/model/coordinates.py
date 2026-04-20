"""Coordinate space definitions and transform helpers.

Three spaces exist in VERSO:

1. Full-resolution space  — original image pixels (e.g. 20000 × 15000)
2. Working-resolution space — thumbnail at 1200 px on long side; all interactive
   operations (control points, masks) happen here.
3. Atlas space — 3D voxel coordinates of the reference atlas volume.

Transform chain:
    full ←→ working  (uniform scale factor stored per section)
    working ←→ atlas (anchoring matrix, see Alignment dataclass)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Full ↔ Working
# ---------------------------------------------------------------------------

def full_to_working(x: float, y: float, scale: float) -> tuple[float, float]:
    """Scale full-resolution pixel coords to working-resolution coords."""
    return x * scale, y * scale


def working_to_full(x: float, y: float, scale: float) -> tuple[float, float]:
    """Scale working-resolution pixel coords back to full-resolution coords."""
    return x / scale, y / scale


# ---------------------------------------------------------------------------
# Working ↔ Normalized section coords
# ---------------------------------------------------------------------------

def pixel_to_normalized(px: float, py: float, width: int, height: int) -> tuple[float, float]:
    """Convert working-resolution pixel coords to normalized section coords [0, 1]²."""
    return px / width, py / height


def normalized_to_pixel(s: float, t: float, width: int, height: int) -> tuple[float, float]:
    """Convert normalized section coords to working-resolution pixel coords."""
    return s * width, t * height


# ---------------------------------------------------------------------------
# Working / Normalized ↔ Atlas
# ---------------------------------------------------------------------------

def normalized_to_atlas(s: float, t: float, anchoring: list[float]) -> np.ndarray:
    """Map normalized section coords (s, t) → atlas voxel coords (x, y, z).

    Uses the QuickNII anchoring convention:
        anchoring = [ox, oy, oz, ux, uy, uz, vx, vy, vz]
        atlas_coord = o + s·u + t·v
    """
    a = np.asarray(anchoring, dtype=np.float64)
    o, u, v = a[0:3], a[3:6], a[6:9]
    return o + s * u + t * v


def atlas_to_normalized(
    xyz: np.ndarray, anchoring: list[float]
) -> tuple[float, float]:
    """Map atlas voxel coords → normalized section coords (s, t).

    Solves the least-squares system: xyz - o = s·u + t·v.
    Returns (s, t); caller should check that both are in [0, 1].
    """
    a = np.asarray(anchoring, dtype=np.float64)
    o, u, v = a[0:3], a[3:6], a[6:9]
    A = np.column_stack([u, v])  # (3, 2)
    b = np.asarray(xyz, dtype=np.float64) - o
    result, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1])
