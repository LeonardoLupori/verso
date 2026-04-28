"""Image loading and resolution management.

Resolution tiers
----------------
Full resolution   Original file on disk (e.g. 20000×15000). Never loaded
                  for interactive use; only for final export.
Working           Downscaled by WORKING_SCALE (default 0.2) from the original.
                  This preserves the pixel-to-atlas-voxel ratio across all
                  sections — matching the QuickNII strategy of keeping a
                  constant scale so the anchoring "width"/"height" fields
                  remain meaningful. Saved to project thumbnails/ folder.
Filmstrip         Longest side clamped to FILMSTRIP_MAX_SIDE (150 px).
                  Generated from the working copy on demand; not persisted.

The working scale is settable: change WORKING_SCALE before creating thumbnails
or pass an explicit ``scale`` argument to :func:`ensure_working_copy`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Default scale factor for working copies (fraction of original resolution).
# A value of 0.2 preserves the pixel:atlas-voxel ratio for typical mouse-brain
# slides scanned at ~1 µm/px with a 25 µm atlas (25 × 0.2 = 5 px/voxel).
# Change this constant or pass an explicit scale to ensure_working_copy().
WORKING_SCALE: float = 0.2
FILMSTRIP_MAX_SIDE = 150


def parse_section_serial_number(path: str | Path, fallback: int) -> int:
    """Extract the absolute section number from a microscopy filename.

    VERSO expects names such as ``MOUSE_123_CODEs.tif`` where the section number
    is the first underscore-delimited numeric field. If no such field exists,
    callers can fall back to list order.
    """
    stem = Path(path).stem
    match = re.match(r"^[^_]+_(\d+)(?:_|$)", stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|_)(\d+)(?:_|$)", stem)
    if match:
        return int(match.group(1))
    return fallback


def thumbnail_filename(path: str | Path) -> str:
    """Return the working-thumbnail filename for a source image path."""
    return f"{Path(path).stem}-thumb.png"


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Read any supported image file as a numpy array (original dtype/shape).

    TIFF files (including multi-channel OME-TIFF) are read with tifffile.
    All other formats are read with PIL.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    return np.asarray(Image.open(str(path)))


def image_dimensions(path: str | Path) -> tuple[int, int]:
    """Return image dimensions as ``(width, height)`` without full decoding."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = series.shape
                axes = series.axes
            if "X" in axes and "Y" in axes:
                return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
            if len(shape) >= 3 and shape[-1] <= 4:
                return int(shape[-2]), int(shape[-3])
            if len(shape) >= 2:
                return int(shape[-1]), int(shape[-2])
        except Exception:
            pass

    from PIL import Image
    with Image.open(str(path)) as im:
        return im.size


def imadjust(
    rgb: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
) -> np.ndarray:
    """Mild percentile contrast stretch, similar to MATLAB imadjust.

    For colour thumbnails, red and green are stretched independently using
    MATLAB-like 1/99.8 percentiles while blue is left at its original 0-255
    range. Grayscale RGB images are adjusted uniformly across all channels.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return _stretch_uint8(rgb, low_pct, high_pct)

    if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
        rgb[:, :, 0], rgb[:, :, 2]
    ):
        adjusted = _stretch_uint8(rgb[:, :, 0], low_pct, high_pct)
        return np.stack([adjusted, adjusted, adjusted], axis=-1)

    out = rgb.copy()
    for channel in (0, 1):
        out[:, :, channel] = _stretch_uint8(rgb[:, :, channel], low_pct, high_pct)
    return out


def _stretch_uint8(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    lo = float(np.percentile(image, low_pct))
    hi = float(np.percentile(image, high_pct))
    if hi <= lo:
        return image.copy()
    out = (image.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return out.clip(0, 255).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Linearly scale any numeric dtype into [0, 255] uint8."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    return img.clip(0, 255).astype(np.uint8)


def select_channel(image: np.ndarray, channel: int = 0) -> np.ndarray:
    """Return a single 2-D plane from a possibly multi-dimensional array."""
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        c = image.shape[2]
        if c <= 4:
            # Channels-last (H, W, C) — the common case for RGB/RGBA
            return image[:, :, min(channel, c - 1)]
        # Might be channels-first (C, H, W) with C very large — unusual.
        # Treat the first axis as channels.
        return image[min(channel, image.shape[0] - 1)]

    # 4-D or higher: collapse all leading axes by taking index 0
    while image.ndim > 2:
        image = image[0]
    return image


def to_rgb(image: np.ndarray, channel: int | None = None) -> np.ndarray:
    """Convert any image array to uint8 H×W×3 suitable for display.

    Args:
        image: Raw numpy array in any dtype and layout.
        channel: Which channel to pick for multi-channel images (0-based).
            If None, uses channel 0.

    Returns:
        uint8 array of shape (H, W, 3).
    """
    img = image
    ch = channel if channel is not None else 0

    # Collapse to 2-D or channels-last 3-D
    if img.ndim == 4 or (img.ndim == 3 and img.shape[2] > 4):
        img = select_channel(img, ch)
    elif img.ndim == 3 and img.shape[2] <= 4:
        pass  # RGB / RGBA — leave as-is
    # else: already 2-D

    img = normalize_to_uint8(img)

    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 4:
        return img[:, :, :3]   # drop alpha
    if img.shape[2] == 3:
        return img
    # Unlikely but safe
    return np.stack([img[:, :, 0]] * 3, axis=-1)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def resize_to_max_side(
    image: np.ndarray, max_side: int
) -> tuple[np.ndarray, float]:
    """Resize image so its longest side ≤ max_side (Lanczos).

    Args:
        image: uint8 H×W or H×W×3 array.
        max_side: Maximum pixels on the long side.

    Returns:
        Tuple (resized_image, scale_factor) where scale = new / original.
    """
    from PIL import Image as PILImage

    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0

    scale = max_side / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    if image.ndim == 2:
        pil = PILImage.fromarray(image, mode="L")
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(pil), scale

    pil = PILImage.fromarray(image[:, :, :3])
    pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil), scale


def resize_by_scale(
    image: np.ndarray, scale: float
) -> tuple[np.ndarray, float]:
    """Resize image by a fixed scale factor (Lanczos).

    Unlike :func:`resize_to_max_side`, this preserves the pixel-to-voxel ratio
    consistently across all sections (QuickNII strategy).

    Args:
        image: uint8 H×W or H×W×3 array.
        scale: Scale factor in (0, 1]. Values > 1 upscale (unusual).

    Returns:
        Tuple (resized_image, scale_factor) where scale_factor == scale.
    """
    from PIL import Image as PILImage

    if scale == 1.0:
        return image, 1.0

    h, w = image.shape[:2]
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    if image.ndim == 2:
        pil = PILImage.fromarray(image, mode="L")
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(pil), scale

    pil = PILImage.fromarray(image[:, :, :3])
    pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil), scale


# ---------------------------------------------------------------------------
# High-level helpers used by the GUI
# ---------------------------------------------------------------------------

def load_for_display(
    path: str | Path,
    max_side: int = FILMSTRIP_MAX_SIDE,
    channel: int | None = None,
) -> tuple[np.ndarray, float]:
    """Load a file and return a display-ready uint8 RGB array capped at max_side.

    Returns:
        (rgb_image, scale_factor)  — scale = output_long_side / original_long_side
    """
    raw = load_image(path)
    rgb = to_rgb(raw, channel)
    return resize_to_max_side(rgb, max_side)


def ensure_working_copy(
    section,
    scale: float | None = None,
) -> np.ndarray | None:
    """Return the working-resolution RGB image for a section.

    Uses a fixed *scale* factor (QuickNII strategy) so the pixel-to-atlas-voxel
    ratio stays constant across sections.  Defaults to :data:`WORKING_SCALE`.

    If the thumbnail already exists on disk it is loaded directly (no
    re-scaling — the persisted file is assumed to be at the correct resolution).
    Otherwise the original is loaded, scaled, saved to ``thumbnail_path``, and
    ``section.scale`` is updated.

    Args:
        section: A :class:`~verso.engine.model.project.Section` instance.
        scale: Override the global :data:`WORKING_SCALE`. Pass ``None`` to use
            the module default.

    Returns:
        uint8 H×W×3 array, or ``None`` if no source image is available.
    """
    working_scale = scale if scale is not None else WORKING_SCALE

    thumb = Path(section.thumbnail_path)
    if thumb.exists():
        try:
            raw = load_image(thumb)
            return to_rgb(raw)
        except Exception:
            pass  # fall through to original

    orig = Path(section.original_path)
    if not orig.exists():
        return None

    try:
        raw = load_image(orig)
        rgb = to_rgb(raw)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load image '{orig.name}': {exc}\n\n"
            "If this is a compressed TIFF, make sure 'imagecodecs' is installed."
        ) from exc

    rgb, actual_scale = resize_by_scale(rgb, working_scale)
    section.scale = actual_scale
    rgb = imadjust(rgb)

    # Persist as PNG for fast future loads
    try:
        thumb.parent.mkdir(parents=True, exist_ok=True)
        _save_png(rgb, thumb)
    except OSError:
        pass  # Can't write thumbnail — that's OK, just won't cache

    return rgb


def registration_dimensions(section) -> tuple[int, int]:
    """Return dimensions of the image used for interactive registration.

    VERSO registers against the working copy when it exists. Falling back to
    the original keeps imported QuickNII/VisuAlign JSON usable even when there
    is no VERSO thumbnail cache.
    """
    thumb = Path(section.thumbnail_path)
    if thumb.exists():
        return image_dimensions(thumb)
    return image_dimensions(section.original_path)


def load_filmstrip_thumbnail(section) -> np.ndarray | None:
    """Return a tiny (≤ FILMSTRIP_MAX_SIDE) RGB image for the filmstrip."""
    thumb = Path(section.thumbnail_path)
    src = thumb if thumb.exists() else Path(section.original_path)
    if not src.exists():
        return None
    rgb, _ = load_for_display(src, max_side=FILMSTRIP_MAX_SIDE)
    return rgb


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image[:, :, :3] if image.ndim == 3 else image)
    pil.save(str(path), format="PNG")
