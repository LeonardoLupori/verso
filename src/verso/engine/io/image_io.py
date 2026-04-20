"""Image loading and resolution management.

Resolution tiers
----------------
Full resolution   Original file on disk (e.g. 20000×15000). Never loaded
                  for interactive use; only for final export.
Working           Longest side clamped to WORKING_MAX_SIDE (1200 px).
                  Saved to project thumbnails/ folder. Used in all views.
Filmstrip         Longest side clamped to FILMSTRIP_MAX_SIDE (150 px).
                  Generated from the working copy on demand; not persisted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORKING_MAX_SIDE = 1200
FILMSTRIP_MAX_SIDE = 150


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


# ---------------------------------------------------------------------------
# High-level helpers used by the GUI
# ---------------------------------------------------------------------------

def load_for_display(
    path: str | Path,
    max_side: int = WORKING_MAX_SIDE,
    channel: int | None = None,
) -> tuple[np.ndarray, float]:
    """Load a file and return a display-ready uint8 RGB array at working res.

    Returns:
        (rgb_image, scale_factor)  — scale = working_long_side / original_long_side
    """
    raw = load_image(path)
    rgb = to_rgb(raw, channel)
    return resize_to_max_side(rgb, max_side)


def ensure_working_copy(section) -> np.ndarray | None:
    """Return the working-resolution RGB image for a section.

    Loads the existing thumbnail if present; otherwise generates it from the
    original, saves it to thumbnail_path, and updates section.scale.

    Returns None if neither file exists.
    """
    thumb = Path(section.thumbnail_path)
    if thumb.exists():
        try:
            rgb, _ = load_for_display(thumb, max_side=WORKING_MAX_SIDE)
            return rgb
        except Exception:
            pass  # fall through to original

    orig = Path(section.original_path)
    if not orig.exists():
        return None

    try:
        rgb, scale = load_for_display(orig, max_side=WORKING_MAX_SIDE)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load image '{orig.name}': {exc}\n\n"
            "If this is a compressed TIFF, make sure 'imagecodecs' is installed."
        ) from exc

    section.scale = scale

    # Persist as PNG for fast future loads
    try:
        thumb.parent.mkdir(parents=True, exist_ok=True)
        _save_png(rgb, thumb)
    except OSError:
        pass  # Can't write thumbnail — that's OK, just won't cache

    return rgb


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
