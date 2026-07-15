"""Export an aligned, un-warped TIFF stack of section images.

This inverts the registration: instead of warping the atlas overlay onto the
(deformed) section, it resamples each section *into the atlas frame*. The
result is a multi-page TIFF where every page is the section's intensity data
laid onto a clean, axis-aligned atlas slice ("the closest atlas section"), with
both the nonlinear warp and the affine rotation/shear/stretch undone. Every page
shares the same atlas-voxel grid, so the stack is mutually co-registered and
overlays the reference atlas annotation directly.

Inverse pipeline, per output pixel (a voxel on the canonical plane):

    atlas voxel ──(inverse affine)──▶ atlas-overlay (s, t)
                ──(inverse warp)─────▶ section image (s', t')
                ──(sample)───────────▶ section intensity

The inverse affine is a single pseudo-inverse of the section's anchoring applied
across the whole output grid; the inverse warp reuses
:func:`verso.engine.warping.warp_points_atlas_to_section` (atlas → section
direction); the final sample is one :func:`cv2.remap` per channel.

Two optional post-steps:
  * background masking — set pixels outside the section's slice mask (and outside
    the section's coverage) to a flat colour (black or white);
  * slice-index merge — max-project pages that share a ``slice_index`` into one,
    reconstructing a single physical section that was imaged as several pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from verso.engine.anchoring import anchoring_center, anchoring_to_vectors, make_atlas_sample_grid
from verso.engine.atlas import AtlasVolume
from verso.engine.io.image_io import ensure_working_copy
from verso.engine.model.project import Project, Section
from verso.engine.preprocessing import apply_flip, load_mask
from verso.engine.warping import warp_points_atlas_to_section


@dataclass
class ExportStackOptions:
    """User-selected parameters for an aligned-stack export.

    The slicing axis is taken from ``Project.interpolation_axis``; all
    working-resolution channels are always preserved.
    """

    # Output resolution relative to the atlas voxel grid: 1 == one output pixel
    # per atlas voxel along the plane, higher == bigger.
    scale: float = 4.0
    # Export every section vs. a caller-supplied subset (see GUI handler).
    all_sections: bool = True
    # ``None`` keeps the full section (uncovered area stays black). "black" or
    # "white" sets every pixel outside the slice mask (and outside coverage) to
    # that flat colour.
    background: str | None = None
    # Max-project pages that share a ``slice_index`` into one page.
    merge_by_slice_index: bool = False


def _has_usable_anchoring(section: Section) -> bool:
    """True if the section's anchoring defines a non-degenerate plane."""
    anchoring = section.alignment.current_anchoring
    if not anchoring or len(anchoring) != 9:
        return False
    _, u, v = anchoring_to_vectors(anchoring)
    return int(np.linalg.matrix_rank(np.column_stack([u, v]))) == 2


def build_canonical_remap(
    section: Section,
    atlas: AtlasVolume,
    axis: int,
    scale: float,
    work_w: int,
    work_h: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build ``cv2.remap`` maps pulling the section onto its canonical atlas plane.

    Args:
        section: Section whose registration (anchoring + control points) is inverted.
        atlas: Atlas volume, used for plane dimensions only.
        axis: Slicing axis in anchoring voxel order (0 = LR, 1 = AP, 2 = DV).
        scale: Output resolution multiplier relative to the atlas voxel grid.
        work_w: Working-image width in pixels (sample source).
        work_h: Working-image height in pixels.

    Returns:
        ``(map_x, map_y, out_w, out_h)`` where the maps are ``(out_h, out_w)``
        float32 arrays for :func:`cv2.remap`. Output pixels not covered by the
        section are set to ``-1`` so ``BORDER_CONSTANT`` yields the border value
        and ``map_x >= 0`` recovers the coverage mask.
    """
    width_vox, height_vox = atlas.axis_plane_dims(axis)
    out_w = max(1, round(width_vox * scale))
    out_h = max(1, round(height_vox * scale))

    # Canonical (straight) atlas plane at this section's registered position.
    position = float(anchoring_center(section.alignment.current_anchoring)[axis])
    canonical = atlas.canonical_plane_anchoring(position, axis)
    grid = make_atlas_sample_grid(canonical, out_w, out_h)  # (H, W, 3) atlas voxels

    # Inverse affine: atlas voxel -> section atlas-overlay (s, t). Solve the
    # least-squares system grid - o = s·u + t·v for every pixel at once via the
    # anchoring's pseudo-inverse (vectorised atlas_to_normalized).
    o, u, v = anchoring_to_vectors(section.alignment.current_anchoring)
    pinv = np.linalg.pinv(np.column_stack([u, v]))  # (2, 3)
    st = (grid - o) @ pinv.T  # (H, W, 2)

    # Coverage: voxels whose (s, t) fall outside the section frame have no data.
    covered = (
        (st[:, :, 0] >= 0.0) & (st[:, :, 0] <= 1.0) & (st[:, :, 1] >= 0.0) & (st[:, :, 1] <= 1.0)
    )

    # Inverse warp: atlas-overlay (s, t) -> section image (s', t').
    cps = section.warp.control_points
    if cps:
        src = np.array([[cp.src_x, cp.src_y] for cp in cps], dtype=np.float64)
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps], dtype=np.float64)
        sec = warp_points_atlas_to_section(st.reshape(-1, 2), src, dst, work_w, work_h)
        sec = sec.reshape(out_h, out_w, 2)
    else:
        sec = st

    map_x = (sec[:, :, 0] * work_w).astype(np.float32)
    map_y = (sec[:, :, 1] * work_h).astype(np.float32)
    map_x[~covered] = -1.0
    map_y[~covered] = -1.0
    return map_x, map_y, out_w, out_h


def export_section_aligned(
    section: Section,
    project: Project,
    atlas: AtlasVolume,
    scale: float,
    *,
    apply_slice_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resample one section onto its canonical atlas plane.

    Args:
        section: Section to export.
        project: Owning project (for ``working_scale`` and ``interpolation_axis``).
        atlas: Atlas volume defining the canonical plane.
        scale: Output resolution multiplier relative to the atlas voxel grid.
        apply_slice_mask: If True, intersect the valid region with the section's
            slice mask (when one exists) so pixels outside the tissue are dropped.

    Returns:
        ``(page, valid)`` where ``page`` is an ``(out_h, out_w, C)`` uint8 image
        with intensities zeroed outside ``valid`` (clean for max-projection), and
        ``valid`` is an ``(out_h, out_w)`` bool mask of pixels carrying real data.
        ``None`` if the section has no usable anchoring or no working image.
    """
    if not _has_usable_anchoring(section):
        return None
    work = ensure_working_copy(section, project.working_scale)
    if work is None:
        return None
    # Control points and the slice mask live in the *displayed* (flipped) space,
    # so apply the same flips render_section_rgb does before sampling.
    work = apply_flip(work, section.preprocessing)
    if work.ndim == 2:
        work = work[:, :, np.newaxis]
    work_h, work_w = work.shape[:2]

    map_x, map_y, _out_w, _out_h = build_canonical_remap(
        section, atlas, project.interpolation_axis_index, scale, work_w, work_h
    )
    covered = map_x >= 0.0

    valid = covered
    mask_path = section.preprocessing.slice_mask_path
    if apply_slice_mask and mask_path and Path(mask_path).exists():
        mask = apply_flip(
            load_mask(mask_path, (work_h, work_w)).astype(np.uint8), section.preprocessing
        )
        mask_warp = cv2.remap(
            mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = covered & (mask_warp > 0)

    channels = [
        cv2.remap(
            np.ascontiguousarray(work[:, :, c]),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for c in range(work.shape[2])
    ]
    page = np.dstack(channels)
    page[~valid] = 0  # neutral background so max-projection / merge stays clean
    return page, valid


def finalize_aligned_pages(
    entries: list[tuple[int, np.ndarray, np.ndarray]],
    options: ExportStackOptions,
) -> list[np.ndarray]:
    """Apply slice-index merge and background colour to per-section results.

    Args:
        entries: ``(slice_index, page, valid)`` tuples in page order, where
            ``page`` is intensity-zeroed outside ``valid`` (as returned by
            :func:`export_section_aligned`).
        options: Export options (``merge_by_slice_index``, ``background``).

    Returns:
        Finished ``(H, W, C)`` uint8 pages ready to write.
    """
    if options.merge_by_slice_index:
        merged: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        order: list[int] = []
        for slice_index, page, valid in entries:
            if slice_index not in merged:
                merged[slice_index] = (page.copy(), valid.copy())
                order.append(slice_index)
            else:
                acc_page, acc_valid = merged[slice_index]
                np.maximum(acc_page, page, out=acc_page)
                np.logical_or(acc_valid, valid, out=acc_valid)
        pairs = [merged[i] for i in order]
    else:
        pairs = [(page, valid) for _, page, valid in entries]

    if options.background is not None:
        bg = 255 if options.background == "white" else 0
        for page, valid in pairs:
            page[~valid] = bg

    return [page for page, _ in pairs]


def write_aligned_stack(
    pages: list[np.ndarray],
    channel_names: list[str],
    out_path: Path,
) -> Path:
    """Write aligned section pages as a multi-page OME-TIFF (axes ``ZCYX``).

    Args:
        pages: List of ``(H, W, C)`` uint8 arrays, all the same shape.
        channel_names: Channel names for the OME metadata.
        out_path: Destination ``.ome.tif`` path.

    Returns:
        The written path.
    """
    import tifffile

    if not pages:
        raise ValueError("No sections produced output — nothing to write.")

    data = np.stack(pages, axis=0)  # (Z, H, W, C)
    data = np.transpose(data, (0, 3, 1, 2))  # (Z, C, H, W)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(out_path),
        data,
        photometric="minisblack",
        metadata={"axes": "ZCYX", "Channel": {"Name": list(channel_names)}},
    )
    return out_path


def export_aligned_stack(
    sections: list[Section],
    project: Project,
    atlas: AtlasVolume,
    options: ExportStackOptions,
    out_path: Path,
) -> tuple[Path, list[str]]:
    """Resample sections onto canonical atlas planes and write a TIFF stack.

    Convenience driver for scripting; the GUI loops :func:`export_section_aligned`
    itself so it can show progress and allow cancellation.

    Args:
        sections: Sections to export, in the desired page order.
        project: Owning project.
        atlas: Atlas volume defining the canonical plane.
        options: Export options (scale, background, merge).
        out_path: Destination ``.ome.tif`` path.

    Returns:
        ``(written_path, skipped)`` where ``skipped`` lists section ids that were
        omitted (no usable anchoring or unreadable working image).
    """
    entries: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for section in sections:
        result = export_section_aligned(
            section, project, atlas, options.scale, apply_slice_mask=options.background is not None
        )
        if result is None:
            skipped.append(section.id)
        else:
            page, valid = result
            entries.append((section.slice_index, page, valid))

    pages = finalize_aligned_pages(entries, options)
    channel_names = [c.name for c in project.channels] or [
        f"Ch {i}" for i in range(pages[0].shape[2] if pages else 0)
    ]
    write_aligned_stack(pages, channel_names, out_path)
    return out_path, skipped
