"""Export warped images, CSVs, and point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verso.engine.model.project import Project, Section


def export_warped_image(section: Section, out_path: Path) -> None:
    """Write the full-resolution warped section image to *out_path*.

    Args:
        section: Section whose warp control points will be applied.
        out_path: Destination file path.
    """
    raise NotImplementedError


def export_region_csv(project: Project, out_path: Path) -> None:
    """Write per-region cell counts / areas to a CSV at *out_path*.

    Args:
        project: Project containing quantification results.
        out_path: Destination ``.csv`` file path.
    """
    raise NotImplementedError


def export_point_cloud(
    points: np.ndarray, labels: np.ndarray, out_path: Path
) -> None:
    """Write an atlas-space point cloud to *out_path*.

    Args:
        points: Float array of shape (N, 3) in atlas space (LR, AP, DV).
        labels: Integer region-label array of shape (N,).
        out_path: Destination file path (``.csv`` or ``.npy``).
    """
    raise NotImplementedError
