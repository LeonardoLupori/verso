"""Non-destructive preprocessing: flip, masking, contrast.

All operations return parameters to store in project.json; they never
modify the original image on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verso.engine.model.project import Preprocessing

_FOREGROUND_SENSITIVITY = 0.25


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


def apply_channel_luminance(rgb: np.ndarray, red: float, green: float) -> np.ndarray:
    """Apply maskEditor-style per-channel display luminance.

    This matches MATLAB ``imadjust(channel, [0, scale], [0, 1])`` for the red
    and green channels: lower non-zero scale values brighten/saturate the
    channel. A scale of 0 hides that channel for display.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape HxWx3")

    out = rgb.astype(np.float32, copy=True)
    for channel, scale in ((0, red), (1, green)):
        if scale <= 0:
            out[:, :, channel] = 0
        else:
            out[:, :, channel] = out[:, :, channel] / min(float(scale), 1.0)
    return out.clip(0, 255).astype(np.uint8)


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
    """Write a bool HxW mask as a single-channel PNG."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(str(path), format="PNG")


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
    alpha = int(round(min(max(opacity, 0.0), 1.0) * 255))

    rgba = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = color[0]
    rgba[:, :, 1] = color[1]
    rgba[:, :, 2] = color[2]
    rgba[:, :, 3] = visible.astype(np.uint8) * alpha
    return rgba


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


def detect_foreground(rgb: np.ndarray) -> np.ndarray:
    """Auto-segment tissue from bright or dark background.

    Background polarity is estimated from border luminance. The returned mask
    uses True=tissue/foreground.
    """
    from scipy import ndimage as ndi
    from skimage import morphology
    from skimage.color import rgb2gray

    if rgb.ndim == 2:
        gray = rgb.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
    elif rgb.ndim == 3:
        gray = rgb2gray(rgb[:, :, :3])
    else:
        raise ValueError("Expected a 2-D or RGB image")

    if gray.size == 0:
        return np.ones(gray.shape, dtype=bool)

    border = _border_values(gray)
    bright_background = float(np.median(border)) >= 0.5

    try:
        threshold = _sensitive_threshold(
            gray,
            bright_background=bright_background,
            background_level=float(np.median(border)),
            sensitivity=_FOREGROUND_SENSITIVITY,
        )
    except ValueError:
        return np.ones(gray.shape, dtype=bool)

    foreground = gray < threshold if bright_background else gray > threshold
    foreground = morphology.closing(foreground, morphology.disk(3))
    foreground = ndi.binary_fill_holes(foreground)
    min_size = max(16, int(gray.size * 0.001))
    foreground = morphology.remove_small_objects(
        foreground.astype(bool),
        max_size=min_size - 1,
    )
    foreground = _largest_component(foreground)

    if not _usable_mask(foreground):
        return np.ones(gray.shape, dtype=bool)
    return foreground


def _sensitive_threshold(
    gray: np.ndarray,
    *,
    bright_background: bool,
    background_level: float,
    sensitivity: float,
) -> float:
    from skimage import filters

    threshold = float(filters.threshold_otsu(gray))
    sensitivity = min(max(float(sensitivity), 0.0), 1.0)
    if bright_background:
        return threshold + max(0.0, background_level - threshold) * sensitivity
    return threshold - max(0.0, threshold - background_level) * sensitivity


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
