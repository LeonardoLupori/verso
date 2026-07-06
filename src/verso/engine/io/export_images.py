"""Export section images with a high-quality atlas overlay.

Produces publication-grade output by smoothing the atlas annotation in *label
space*: each region's signed-distance field (SDF) is blurred, upscaled, and the
per-pixel argmax over regions reconstructs a single, shared, artifact-free set
of smooth boundaries with region IDs preserved exactly. From that smoothed label
map we render either region outlines or filled Allen-coloured regions.

The interactive GUI overlay runs at low resolution for responsiveness; this
module exists so exports can use a higher ``scale`` (relative to the atlas voxel
grid) without the voxel/warp staircase artifacts that a plain nearest-neighbour
upscale would show.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from verso.engine.atlas import AtlasVolume
from verso.engine.io.image_io import _resize_multichannel, load_image, to_multichannel
from verso.engine.model.project import Project, Section
from verso.engine.preprocessing import apply_flip, composite_channels
from verso.engine.warping import build_backward_remap

# Smoothing slider (0–100) maps to a Gaussian sigma applied to the per-region
# SDF at *base* (atlas-voxel) resolution, so the smoothing strength is
# independent of the export ``scale``. ~5 voxels is already very heavy
# smoothing for a 25 µm mouse atlas, so 100 → 5.0 covers the useful range.
_MAX_SMOOTHING_SIGMA = 5.0

# Output long side (px) used when a section has no usable anchoring, so the
# section image still exports at a reasonable resolution even though no atlas
# voxel count is available to derive it from.
_NO_ANCHOR_LONG_SIDE = 2000


@dataclass
class ExportOptions:
    """User-selected parameters for an images-with-overlay export."""

    burn_overlay: bool = True
    overlay_color: tuple[int, int, int] = (255, 255, 255)
    overlay_opacity: float = 1.0  # 0..1
    # Output resolution relative to the atlas voxel grid: 1 == one output pixel
    # per atlas voxel along the plane, higher == bigger. Output long side is
    # ``round(scale * atlas_plane_voxels)`` with the section aspect preserved.
    scale: float = 4.0
    # 0 (no blur) .. 100 (very smooth contours). Mapped to an SDF Gaussian
    # sigma in atlas-voxel units (see :data:`_MAX_SMOOTHING_SIGMA`).
    smoothing: float = 30.0
    # "outline" — region boundaries in ``overlay_color``;
    # "filled"  — Allen-coloured semi-transparent regions.
    overlay_style: str = "outline"
    outline_thickness: int = 1


def _target_dims(orig_w: int, orig_h: int, long_side: int) -> tuple[int, int]:
    """Return (out_w, out_h) preserving aspect ratio with longest side == long_side."""
    if orig_w >= orig_h:
        out_w = int(long_side)
        out_h = max(1, round(orig_h * (long_side / orig_w)))
    else:
        out_h = int(long_side)
        out_w = max(1, round(orig_w * (long_side / orig_h)))
    return out_w, out_h


def render_section_rgb(section: Section, project: Project, long_side: int) -> np.ndarray:
    """Composite the section's channels into a single RGB image at *long_side*.

    Loads the full-resolution original, applies the section's flip flags, then
    composites visible channels using the project's :class:`ChannelSpec` list
    so the output matches what the user sees in the GUI.

    Returns:
        uint8 ``(H, W, 3)`` array with longest side equal to *long_side*.
    """
    raw = load_image(section.original_path)
    img = to_multichannel(raw)  # uint8 (H, W, C), percentile-stretched per channel
    img = apply_flip(img, section.preprocessing)

    orig_h, orig_w = img.shape[:2]
    out_w, out_h = _target_dims(orig_w, orig_h, long_side)
    if (out_w, out_h) != (orig_w, orig_h):
        img = _resize_multichannel(img, (out_w, out_h))

    return composite_channels(img, project.channels)


def _smooth_label_map(
    labels: np.ndarray,
    out_w: int,
    out_h: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth + upscale an integer label map via per-region signed-distance fields.

    Each region's SDF (positive inside, negative outside) is optionally Gaussian
    blurred, upscaled with cubic interpolation, and the per-pixel argmax over all
    regions reconstructs a single set of smooth, shared boundaries with labels
    preserved exactly.

    For speed each region is processed inside its bounding box plus a margin
    (``ceil(3·sigma) + 2``): a region's SDF can only win the argmax within ~its
    own boundary neighbourhood, so cropping is exact where it matters while
    avoiding a full-frame distance-transform + cubic resize per region.

    Args:
        labels: ``(H, W)`` integer region-ID map (base / atlas-voxel resolution).
        out_w: Output width in pixels (``> H/W`` for an upscale).
        out_h: Output height in pixels.
        sigma: Gaussian sigma applied to each SDF at base resolution. ``0`` skips
            blurring (boundaries still get sub-pixel-smoothed by the cubic upscale).

    Returns:
        ``(smoothed_ids, best_lab)`` where ``smoothed_ids`` is the ``(out_h,
        out_w)`` label map with real region IDs and ``best_lab`` is the compact
        ``(out_h, out_w)`` argmax index map (handy for fast boundary diffs).
    """
    from scipy import ndimage

    ids, compact = np.unique(labels, return_inverse=True)
    compact = compact.reshape(labels.shape).astype(np.int32)
    base_h, base_w = labels.shape
    sx = out_w / base_w
    sy = out_h / base_h
    margin = int(np.ceil(3.0 * sigma)) + 2 if sigma > 0 else 2

    best_val = np.full((out_h, out_w), -1e30, np.float32)
    best_lab = np.zeros((out_h, out_w), np.int32)

    for li in range(len(ids)):
        mask = compact == li
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:
            continue
        r0 = max(0, int(rows[0]) - margin)
        r1 = min(base_h, int(rows[-1]) + 1 + margin)
        c0 = max(0, int(cols[0]) - margin)
        c1 = min(base_w, int(cols[-1]) + 1 + margin)

        m = mask[r0:r1, c0:c1]
        sdf = (ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)).astype(
            np.float32
        )
        if sigma > 0:
            sdf = cv2.GaussianBlur(sdf, (0, 0), sigma)

        # Output window covering this crop. Use round() at both edges so adjacent
        # crops tile the output grid without gaps or overlaps.
        or0 = round(r0 * sy)
        or1 = round(r1 * sy)
        oc0 = round(c0 * sx)
        oc1 = round(c1 * sx)
        ow = oc1 - oc0
        oh = or1 - or0
        if ow <= 0 or oh <= 0:
            continue
        up = cv2.resize(sdf, (ow, oh), interpolation=cv2.INTER_CUBIC)

        win_val = best_val[or0:or1, oc0:oc1]
        win_lab = best_lab[or0:or1, oc0:oc1]
        better = up > win_val
        win_val[better] = up[better]
        win_lab[better] = li

    return ids[best_lab], best_lab


def _boundary_mask(compact: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Boolean edge mask between differing labels, keeping brain-adjacent edges.

    Mirrors :meth:`AtlasVolume.slice_outline`: a 1-px edge is marked where two
    neighbouring pixels carry different labels and at least one is annotated
    brain, so the empty-background frame is not outlined.
    """
    edges = np.zeros(compact.shape, dtype=bool)
    diff_h = compact[:, :-1] != compact[:, 1:]
    edges[:, :-1] |= diff_h & (brain[:, :-1] | brain[:, 1:])
    diff_v = compact[:-1, :] != compact[1:, :]
    edges[:-1, :] |= diff_v & (brain[:-1, :] | brain[1:, :])
    return edges


def render_overlay_rgba(
    section: Section,
    atlas: AtlasVolume,
    out_w: int,
    out_h: int,
    *,
    scale: float = 4.0,
    smoothing: float = 30.0,
    overlay_style: str = "outline",
    color: tuple[int, int, int] = (255, 255, 255),
    opacity: float = 1.0,
    thickness: int = 1,
) -> np.ndarray:
    """Render the atlas overlay as a transparent RGBA image at the export size.

    Pipeline: sample atlas labels at base (atlas-voxel) resolution → backward-warp
    the label map through the Delaunay map → SDF-smooth + upscale to the output
    size → render region outlines (anti-aliased) or filled Allen-coloured regions.

    The overlay is *not* flipped here. Like the GUI (which flips only the section
    background, never the atlas overlay), the atlas orientation is fully encoded
    by the anchoring, which was solved against the already-flipped display. The
    section RGB is flipped separately in :func:`render_section_rgb`, so the two
    compose correctly.

    Args:
        out_w: Output width in pixels (section aspect; set by the caller).
        out_h: Output height in pixels.
        scale: Output resolution relative to the atlas voxel grid. Sampling and
            warping happen at ``round(out/scale)``; the SDF step upscales back up.
        smoothing: 0–100 smoothing strength (see :data:`_MAX_SMOOTHING_SIGMA`).
        overlay_style: ``"outline"`` or ``"filled"``.
        color: Outline RGB colour (ignored for ``"filled"``).
        opacity: 0–1 overlay opacity.
        thickness: Outline thickness in output pixels (ignored for ``"filled"``).

    Returns:
        uint8 ``(out_h, out_w, 4)`` RGBA. Background pixels have alpha 0.
    """
    anchoring = section.alignment.anchoring
    if not anchoring or all(v == 0.0 for v in anchoring):
        return np.zeros((out_h, out_w, 4), dtype=np.uint8)

    base_w = max(1, round(out_w / scale))
    base_h = max(1, round(out_h / scale))

    # Out-of-volume voxels are clipped to label 0 (background); the brain mask
    # (label > 0) is what excludes them downstream, so in_bounds is not needed.
    labels, _ = atlas.sample_labels(anchoring, base_w, base_h)

    # Apply the warp using the same backward remap as the display pipeline so the
    # exported overlay matches what the user sees in the Warp view. Nearest-
    # neighbour keeps the integer labels exact; the SDF step smooths the resulting
    # staircase together with the atlas voxel staircase.
    cps = section.warp.control_points
    if cps:
        src = np.array([[cp.src_x, cp.src_y] for cp in cps], dtype=np.float64)
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps], dtype=np.float64)
        map_x, map_y = build_backward_remap(base_h, base_w, src, dst, out_w, out_h)
        labels = cv2.remap(
            labels.astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.int32)

    sigma = max(0.0, min(smoothing, 100.0)) / 100.0 * _MAX_SMOOTHING_SIGMA
    smooth_ids, _ = _smooth_label_map(labels, out_w, out_h, sigma)

    alpha_scalar = min(max(opacity, 0.0), 1.0)
    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    if overlay_style == "filled":
        rgb = atlas.colorize_labels(smooth_ids)
        alpha = np.where(smooth_ids > 0, round(alpha_scalar * 255), 0).astype(np.uint8)
        canvas[..., :3] = rgb
        canvas[..., 3] = alpha
        return canvas

    # Outline: edge mask from the smoothed label map, dilated to the requested
    # thickness, then lightly feathered for anti-aliasing.
    _, compact = np.unique(smooth_ids, return_inverse=True)
    compact = compact.reshape(smooth_ids.shape).astype(np.int32)
    brain = smooth_ids > 0
    edges = _boundary_mask(compact, brain)

    line = edges.astype(np.float32)
    if thickness > 1:
        k = int(thickness)
        line = cv2.dilate(line, np.ones((k, k), np.float32))
    line = cv2.GaussianBlur(line, (0, 0), 0.8)
    line = np.clip(line, 0.0, 1.0)

    canvas[..., 0] = int(color[0])
    canvas[..., 1] = int(color[1])
    canvas[..., 2] = int(color[2])
    canvas[..., 3] = np.round(line * alpha_scalar * 255).astype(np.uint8)
    return canvas


def _burn_overlay(rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend *overlay_rgba* onto *rgb* and return uint8 ``(H, W, 3)``."""
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    out = rgb.astype(np.float32) * (1.0 - alpha) + overlay_rgb * alpha
    return out.clip(0, 255).astype(np.uint8)


def _save_png(image: np.ndarray, path: Path) -> None:
    """Write a uint8 array as a PNG via PIL (RGB or RGBA based on channel count)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 4:
        Image.fromarray(image, mode="RGBA").save(str(path), format="PNG")
    else:
        Image.fromarray(image, mode="RGB").save(str(path), format="PNG")


def _output_long_side(section: Section, scale: float) -> int:
    """Output long side in pixels: ``scale × atlas-plane voxel count``.

    The anchoring's ``u``/``v`` vectors are lengths in atlas voxels, so their
    magnitudes give the voxel count spanned along the plane. Falls back to a
    fixed default when the section has no usable anchoring.
    """
    anchoring = section.alignment.anchoring
    if not anchoring or all(v == 0.0 for v in anchoring):
        return _NO_ANCHOR_LONG_SIDE
    u = np.asarray(anchoring[3:6], dtype=np.float64)
    v = np.asarray(anchoring[6:9], dtype=np.float64)
    atlas_long = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    if atlas_long <= 0:
        return _NO_ANCHOR_LONG_SIDE
    return max(1, round(scale * atlas_long))


def export_section(
    section: Section,
    project: Project,
    atlas: AtlasVolume,
    options: ExportOptions,
    out_dir: Path,
) -> list[Path]:
    """Render and write the export files for a single section.

    Returns the list of written paths (one file in burn mode, two in
    separate-overlay mode).
    """
    long_side = _output_long_side(section, options.scale)
    rgb = render_section_rgb(section, project, long_side)
    out_h, out_w = rgb.shape[:2]
    overlay = render_overlay_rgba(
        section,
        atlas,
        out_w,
        out_h,
        scale=options.scale,
        smoothing=options.smoothing,
        overlay_style=options.overlay_style,
        color=options.overlay_color,
        opacity=options.overlay_opacity,
        thickness=options.outline_thickness,
    )

    stem = Path(section.original_path).stem
    base = f"{section.slice_index:03d}_{stem}"
    written: list[Path] = []

    if options.burn_overlay:
        out_path = out_dir / f"{base}.png"
        _save_png(_burn_overlay(rgb, overlay), out_path)
        written.append(out_path)
    else:
        section_path = out_dir / f"{base}.png"
        overlay_path = out_dir / f"{base}_overlay.png"
        _save_png(rgb, section_path)
        _save_png(overlay, overlay_path)
        written.extend([section_path, overlay_path])

    return written
