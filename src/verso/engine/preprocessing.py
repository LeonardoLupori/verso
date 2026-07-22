"""Non-destructive preprocessing: flip, masking, contrast.

All operations return parameters to store in project.json; they never
modify the original image on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verso.engine.model.project import Preprocessing

_FOREGROUND_SENSITIVITY = 0.25
_MAX_HOLE_SIZE = 3500
_EROSION_AMOUNT = 3


def apply_flip(image: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    """Return a flipped copy of *image* according to *preprocessing* flags.

    Args:
        image: H×W or H×W×C uint8 array.
        preprocessing: Preprocessing parameters from the project model.

    Returns:
        Flipped image array (same dtype and shape).
    """
    out = image
    if preprocessing.flip_horizontal:
        out = np.fliplr(out)
    if preprocessing.flip_vertical:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside *mask*.

    Args:
        image: H×W or H×W×C uint8 array.
        mask: Boolean H×W array; True = keep, False = zero.

    Returns:
        Masked image array.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if image.shape[:2] != mask_bool.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image shape {image.shape[:2]}"
        )
    out = image.copy()
    if out.ndim == 2:
        out[~mask_bool] = 0
    else:
        out[~mask_bool, :] = 0
    return out


def composite_channels(
    image: np.ndarray,
    channels,
) -> np.ndarray:
    """Composite a multichannel image into a displayable RGB.

    For each visible :class:`~verso.engine.model.project.ChannelSpec`:
      * apply an ``imadjust([0, scale], [0, 1])`` brightness boost,
      * apply a ``gamma`` correction (``out = 255·(in/255)**gamma``; 1 = linear),
      * tint the resulting plane by the spec's RGB color,
      * max-blend into the output (matches Fiji "Composite" mode and avoids
        over-saturation).

    If the channel list has fewer entries than the image has planes, extra
    planes are ignored. If it has more entries than the image has planes, the
    excess specs are ignored.

    Args:
        image: uint8 ``(H, W)`` or ``(H, W, C)`` array.
        channels: iterable of :class:`ChannelSpec`-like objects with
            ``scale``, ``color`` (RGB tuple), and ``visible`` attributes.

    Returns:
        uint8 ``(H, W, 3)`` RGB image.
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"Expected (H, W) or (H, W, C), got shape {image.shape}")

    h, w, c = image.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    specs = list(channels)
    n = min(c, len(specs))
    for i in range(n):
        spec = specs[i]
        if not getattr(spec, "visible", True):
            continue
        scale = float(spec.scale)
        if scale <= 0:
            continue
        plane = image[:, :, i].astype(np.float32) / min(scale, 1.0)
        np.clip(plane, 0, 255, out=plane)
        gamma = float(getattr(spec, "gamma", 1.0))
        if gamma > 0 and gamma != 1.0:
            plane = 255.0 * np.power(plane / 255.0, gamma)
        for k in range(3):
            tinted = plane * (spec.color[k] / 255.0)
            np.maximum(out[:, :, k], tinted, out=out[:, :, k])

    return out.clip(0, 255).astype(np.uint8)


def channel_lut(spec) -> np.ndarray:
    """Return a ``(256, 4)`` uint8 RGBA lookup table for a channel spec.

    Encodes the same ``gamma(clip(plane / scale, 0, 255)) × (color / 255)``
    transform that ``composite_channels`` applies per pixel, but precomputed for every
    possible uint8 input value. Used by the GUI canvas to feed pyqtgraph's
    ``ImageItem.setLookupTable``, so brightness/color changes become a
    1 KB table swap instead of a full image recomposite.
    """
    scale = max(float(spec.scale), 1e-6)
    intensities = np.arange(256, dtype=np.float32)
    luminance = np.clip(intensities / min(scale, 1.0), 0.0, 255.0)
    gamma = float(getattr(spec, "gamma", 1.0))
    if gamma > 0 and gamma != 1.0:
        luminance = 255.0 * np.power(luminance / 255.0, gamma)
    color = np.asarray(spec.color, dtype=np.float32) / 255.0
    rgb = np.clip(luminance[:, None] * color[None, :], 0.0, 255.0).astype(np.uint8)
    alpha = np.full((256, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a PNG mask and resize to working-resolution shape (H, W) bool."""
    from PIL import Image

    target_h, target_w = shape
    with Image.open(str(path)) as im:
        im = im.convert("L")
        if im.size != (target_w, target_h):
            im = im.resize((target_w, target_h), Image.Resampling.NEAREST)
        arr = np.asarray(im)
    return arr > 0


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Write a bool HxW mask as a 1-bit PNG."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask, dtype=bool).view(np.uint8) * 255
    Image.fromarray(arr, mode="L").convert("1").save(str(path), format="PNG")


def mask_to_rgba(
    mask: np.ndarray,
    negative: bool,
    opacity: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Convert a bool mask to an RGBA overlay.

    ``negative=False`` highlights True mask pixels. ``negative=True``
    highlights False mask pixels. Saved masks remain True=tissue/foreground.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    visible = ~mask_bool if negative else mask_bool
    alpha = round(min(max(opacity, 0.0), 1.0) * 255)

    rgba = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = color[0]
    rgba[:, :, 1] = color[1]
    rgba[:, :, 2] = color[2]
    rgba[:, :, 3] = visible.astype(np.uint8) * alpha
    return rgba


def morph_mask(mask: np.ndarray, pixels: int, operation: str) -> np.ndarray:
    """Erode or expand a binary mask by *pixels* using a disk structuring element.

    Args:
        mask: Boolean H×W array.
        pixels: Radius in mask pixels (1–20).
        operation: ``"erode"`` or ``"expand"``.

    Returns:
        New bool H×W array with the morphological operation applied.
    """
    import cv2

    radius = max(int(pixels), 1)
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    src = np.asarray(mask, dtype=np.uint8) * 255
    result = cv2.erode(src, kernel) if operation == "erode" else cv2.dilate(src, kernel)
    return result > 0


def apply_freehand_stroke(
    mask: np.ndarray,
    polygon_xy: np.ndarray,
    add: bool,
) -> np.ndarray:
    """Fill a freehand polygon into a copy of the mask."""
    from skimage.draw import polygon

    out = np.asarray(mask, dtype=bool).copy()
    pts = np.asarray(polygon_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return out

    rows, cols = polygon(pts[:, 1], pts[:, 0], shape=out.shape)
    out[rows, cols] = bool(add)
    return out


def apply_brush_stroke(
    mask: np.ndarray,
    points_xy: np.ndarray,
    radius: int,
    add: bool,
) -> np.ndarray:
    """Paint filled disks of ``radius`` along a brush stroke into a copy of the mask.

    Unlike :func:`apply_freehand_stroke`, a single point paints one disk. Sparse
    drag samples are densified along each segment so fast strokes leave no gaps.
    ``add=True`` sets painted pixels to True; ``add=False`` clears them. Radius is
    measured in mask pixels.
    """
    from skimage.draw import disk

    out = np.asarray(mask, dtype=bool).copy()
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return out
    radius = max(int(radius), 1)
    value = bool(add)

    def stamp(cx: float, cy: float) -> None:
        rr, cc = disk((cy, cx), radius, shape=out.shape)
        out[rr, cc] = value

    stamp(pts[0, 0], pts[0, 1])
    step = max(radius / 2.0, 1.0)
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        n = int(dist / step)
        for k in range(1, n + 1):
            t = k / (n + 1)
            stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        stamp(x1, y1)
    return out


def detect_foreground(image: np.ndarray) -> np.ndarray:
    """Auto-segment tissue from bright or dark background.

    Multichannel inputs are reduced to a single plane via max-projection,
    so any channel with strong signal contributes to the foreground mask.
    Background polarity is estimated from border luminance. The returned
    mask uses True=tissue/foreground.
    """
    from skimage import filters, morphology

    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3:
        gray = image.max(axis=2).astype(np.float32)
    else:
        raise ValueError("Expected a 2-D or 3-D image")
    if gray.max() > 1.0:
        gray /= 255.0

    if gray.size == 0:
        return np.ones(gray.shape, dtype=bool)

    border = _border_values(gray)
    bright_background = float(np.median(border)) >= 0.5

    smoothed = filters.gaussian(gray, sigma=3)
    threshold = filters.threshold_li(smoothed)
    foreground = smoothed < threshold if bright_background else smoothed > threshold
    foreground = morphology.erosion(foreground, morphology.disk(_EROSION_AMOUNT))
    foreground = _largest_component(foreground)
    foreground = morphology.remove_small_holes(foreground, max_size=_MAX_HOLE_SIZE)

    if not _usable_mask(foreground):
        return np.ones(gray.shape, dtype=bool)
    return foreground


def _border_values(gray: np.ndarray) -> np.ndarray:
    top = gray[0, :]
    bottom = gray[-1, :]
    left = gray[:, 0]
    right = gray[:, -1]
    return np.concatenate([top, bottom, left, right])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from skimage import measure

    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros(mask.shape, dtype=bool)

    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda region: region.area)
    return labels == largest.label


def _usable_mask(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    total = int(mask.size)
    if total == 0:
        return False
    return max(8, int(total * 0.001)) <= area <= int(total * 0.98)
