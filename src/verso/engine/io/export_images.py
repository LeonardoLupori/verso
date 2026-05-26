"""Export section images with a high-quality atlas overlay.

Produces publication-grade output by re-sampling the atlas annotation at the
requested resolution, extracting region boundaries as sub-pixel contours
(``skimage.measure.find_contours``), warping those contours into section
space through the existing Delaunay map, and rendering them with anti-aliased
polylines (``cv2.polylines``) onto a transparent RGBA canvas.

The interactive GUI overlay runs at ~512 px for responsiveness; this module
exists so exports can use whatever long-side the user picks without losing
contour smoothness.
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

# Cap the atlas sampling grid used for contour extraction. The annotation
# volume is at 25 µm voxels, so sampling much finer than this just multiplies
# the same staircase boundary — wasted work for find_contours, the Delaunay
# warp, and the smoothing kernel. 1000 px on the long side is ~2–3× the atlas
# voxel grid for a typical mouse-brain coronal section, which is plenty for
# the smoothing slider to operate on while keeping vertex counts modest.
_ATLAS_SAMPLING_LONG_SIDE = 1000


@dataclass
class ExportOptions:
    """User-selected parameters for an images-with-overlay export."""

    burn_overlay: bool = True
    overlay_color: tuple[int, int, int] = (255, 255, 255)
    overlay_opacity: float = 1.0  # 0..1
    long_side: int = 4000
    outline_thickness: int = 1
    # Gaussian sigma in atlas-sampling pixels (capped resolution, ~1000 px on
    # the long side). 0 disables smoothing.
    contour_smoothing: float = 1.5


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


def _label_contours(labels: np.ndarray) -> list[np.ndarray]:
    """Extract sub-pixel boundary polylines for every labelled region.

    Returns one polyline per connected boundary as a (K, 2) array of
    (row, col) pixel-edge coordinates in the same convention as
    ``skimage.measure.find_contours``.
    """
    from skimage import measure

    polylines: list[np.ndarray] = []
    unique = np.unique(labels)
    for lbl in unique:
        if lbl == 0:
            continue
        mask = (labels == lbl).astype(np.float32)
        for contour in measure.find_contours(mask, 0.5):
            if len(contour) >= 2:
                polylines.append(contour)
    return polylines


def _smooth_polyline(poly: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a (K, 2) polyline along its arc.

    Closed contours (first vertex ≈ last) use periodic boundary conditions so
    the seam doesn't develop a kink; open contours use ``nearest`` so endpoints
    stay put.
    """
    if sigma <= 0.0 or len(poly) < 3:
        return poly
    from scipy.ndimage import gaussian_filter1d

    closed = np.allclose(poly[0], poly[-1])
    mode = "wrap" if closed else "nearest"
    # Smooth row/col independently along the vertex axis.
    src = poly[:-1] if closed else poly
    smoothed = np.empty_like(src)
    smoothed[:, 0] = gaussian_filter1d(src[:, 0], sigma=sigma, mode=mode)
    smoothed[:, 1] = gaussian_filter1d(src[:, 1], sigma=sigma, mode=mode)
    if closed:
        smoothed = np.vstack([smoothed, smoothed[:1]])
    return smoothed


def _apply_flip_norm(points_norm: np.ndarray, *, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Mirror normalised section-space coords to match a flipped image."""
    if not flip_h and not flip_v:
        return points_norm
    out = points_norm.copy()
    if flip_h:
        out[:, 0] = 1.0 - out[:, 0]
    if flip_v:
        out[:, 1] = 1.0 - out[:, 1]
    return out


def render_overlay_rgba(
    section: Section,
    atlas: AtlasVolume,
    out_w: int,
    out_h: int,
    color: tuple[int, int, int] = (255, 255, 255),
    opacity: float = 1.0,
    thickness: int = 1,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Render the atlas overlay as a transparent RGBA image at the export size.

    Pipeline: sample atlas labels at output resolution → find_contours per
    region → forward-warp vertices through the Delaunay map → flip to match
    the displayed orientation → draw anti-aliased polylines.

    Returns:
        uint8 ``(out_h, out_w, 4)`` RGBA. Background pixels have alpha 0.
    """
    anchoring = section.alignment.anchoring
    if not anchoring or all(v == 0.0 for v in anchoring):
        return np.zeros((out_h, out_w, 4), dtype=np.uint8)

    # Atlas sampling is decoupled from the output canvas: contour extraction
    # works in (sample_w, sample_h) space, polylines are normalised to [0, 1],
    # and we draw at full (out_w, out_h) resolution. This keeps find_contours
    # and the smoothing kernel tied to the atlas voxel grid rather than the
    # arbitrary export size.
    scale = min(1.0, _ATLAS_SAMPLING_LONG_SIDE / max(out_w, out_h))
    sample_w = max(1, round(out_w * scale))
    sample_h = max(1, round(out_h * scale))

    labels, _ = atlas.sample_labels(anchoring, sample_w, sample_h)

    # Apply the warp using the same backward remap as the display pipeline so
    # that exported contours match what the user sees in the Warp view.
    # Forward-mapping contour points (warp_points_atlas_to_section) uses the
    # opposite triangulation direction and produces different results.
    cps = section.warp.control_points
    if cps:
        src = np.array([[cp.src_x, cp.src_y] for cp in cps], dtype=np.float64)
        dst = np.array([[cp.dst_x, cp.dst_y] for cp in cps], dtype=np.float64)
        map_x, map_y = build_backward_remap(sample_h, sample_w, src, dst)
        labels = cv2.remap(
            labels.astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.int32)

    polylines = _label_contours(labels)

    flip_h = bool(section.preprocessing.flip_horizontal)
    flip_v = bool(section.preprocessing.flip_vertical)

    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    alpha = int(round(min(max(opacity, 0.0), 1.0) * 255))
    line_color = (int(color[0]), int(color[1]), int(color[2]), alpha)
    # When smoothing is off, drop anti-aliasing so the rendered line stays
    # pixel-faithful to the marching-squares contour rather than getting
    # softened back into apparent curves.
    line_type = cv2.LINE_AA if smoothing > 0.0 else cv2.LINE_8

    for poly in polylines:
        poly = _smooth_polyline(poly, smoothing)
        # After the backward warp, the label map is in section space, so
        # contour pixel coords directly normalise to section-space fractions.
        # find_contours returns (row, col) ≈ (y, x).
        section_norm = np.column_stack(
            [
                (poly[:, 1] + 0.5) / sample_w,  # x
                (poly[:, 0] + 0.5) / sample_h,  # y
            ]
        )
        section_norm = _apply_flip_norm(section_norm, flip_h=flip_h, flip_v=flip_v)

        pts = np.empty((len(section_norm), 1, 2), dtype=np.int32)
        pts[:, 0, 0] = np.round(section_norm[:, 0] * out_w).astype(np.int32)
        pts[:, 0, 1] = np.round(section_norm[:, 1] * out_h).astype(np.int32)

        cv2.polylines(
            canvas,
            [pts],
            isClosed=False,
            color=line_color,
            thickness=int(thickness),
            lineType=line_type,
        )

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
    rgb = render_section_rgb(section, project, options.long_side)
    out_h, out_w = rgb.shape[:2]
    overlay = render_overlay_rgba(
        section,
        atlas,
        out_w,
        out_h,
        color=options.overlay_color,
        opacity=options.overlay_opacity,
        thickness=options.outline_thickness,
        smoothing=options.contour_smoothing,
    )

    stem = Path(section.original_path).stem
    base = f"{section.serial_number:03d}_{stem}"
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
