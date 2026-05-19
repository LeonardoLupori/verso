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


def composite_channels(
    image: np.ndarray,
    channels,
) -> np.ndarray:
    """Composite a multichannel image into a displayable RGB.

    For each visible :class:`~verso.engine.model.project.ChannelSpec`:
      * apply an ``imadjust([0, scale], [0, 1])`` brightness boost,
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
        for k in range(3):
            tinted = plane * (spec.color[k] / 255.0)
            np.maximum(out[:, :, k], tinted, out=out[:, :, k])

    return out.clip(0, 255).astype(np.uint8)


def compute_channel_layer(
    image: np.ndarray,
    channel_index: int,
    spec,
) -> np.ndarray | None:
    """Compute the tinted (H, W, 3) float32 contribution for a single channel.

    Unlike composite_channels(), this does NOT check spec.visible so callers
    can store the layer and selectively include it via composite_from_layers().

    Returns:
        (H, W, 3) float32 array, or None if spec.scale <= 0 (disabled channel).
    """
    scale = float(spec.scale)
    if scale <= 0:
        return None
    plane = image[:, :, channel_index].astype(np.float32) / min(scale, 1.0)
    np.clip(plane, 0, 255, out=plane)
    color = np.array(spec.color, dtype=np.float32) / 255.0
    return plane[:, :, np.newaxis] * color


def composite_from_layers(
    layers: list,
    channels,
) -> np.ndarray:
    """Max-blend precomputed per-channel layers into a displayable RGB image.

    Use together with compute_channel_layer() to avoid re-running per-pixel
    math when only channel visibility changes — only the max-reduce runs.

    Args:
        layers: one (H, W, 3) float32 array per channel (None = disabled).
        channels: channel list matching *layers*, consulted only for ``visible``.

    Returns:
        uint8 (H, W, 3) RGB image.
    """
    specs = list(channels)
    visible = [
        layers[i]
        for i in range(min(len(layers), len(specs)))
        if layers[i] is not None and getattr(specs[i], "visible", True)
    ]
    if not visible:
        ref = next((layer for layer in layers if layer is not None), None)
        shape = ref.shape if ref is not None else (1, 1, 3)
        return np.zeros(shape, dtype=np.uint8)
    out = np.maximum.reduce(visible)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def detect_foreground(image: np.ndarray) -> np.ndarray:
    """Auto-segment tissue from bright or dark background.

    Multichannel inputs are reduced to a single plane via max-projection,
    so any channel with strong signal contributes to the foreground mask.
    Background polarity is estimated from border luminance. The returned
    mask uses True=tissue/foreground.
    """
    from scipy import ndimage as ndi
    from skimage import morphology

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


# ---------------------------------------------------------------------------
# L/R hemisphere masks
#
# Tri-valued uint8 arrays:  0 = unlabeled, 1 = left, 2 = right.
# Saved unflipped on disk; flip + value-swap are applied at display time via
# flip_lr_mask().  See gui/views/prep_view.py for the consuming flow.
# ---------------------------------------------------------------------------


def save_lr_mask(mask: np.ndarray, path: str | Path) -> None:
    """Write a tri-valued (0/1/2) uint8 H×W mask as a single-channel PNG.

    Values are written literally — viewing the file in an external image
    viewer will look almost black, which is acceptable for a non-user-facing
    artefact.  Use NEAREST resize on load to preserve discrete values.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(str(path), format="PNG")


def load_lr_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a tri-valued PNG and NEAREST-resize to *shape* = (H, W)."""
    from PIL import Image

    target_h, target_w = shape
    with Image.open(str(path)) as im:
        im = im.convert("L")
        if im.size != (target_w, target_h):
            im = im.resize((target_w, target_h), Image.Resampling.NEAREST)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


def rasterize_lr_line(
    p0: tuple[float, float],
    p1: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a uint8 (H, W) mask split by the line p0→p1.

    Side is determined by the sign of the 2-D cross product
    ``(p1-p0) × (q-p0)``: negative → 1 (left), positive → 2 (right).
    Pixels exactly on the line are assigned 2 (consistent with the strict
    sign convention used by the hover-tint helper).
    """
    h, w = shape
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    x0 = float(p0[0])
    y0 = float(p0[1])

    # Pixel-centre coordinates: (x = col, y = row).
    cols = np.arange(w, dtype=np.float32)
    rows = np.arange(h, dtype=np.float32)
    qx = cols[np.newaxis, :]
    qy = rows[:, np.newaxis]

    cross = dx * (qy - y0) - dy * (qx - x0)
    out = np.where(cross < 0.0, np.uint8(1), np.uint8(2))
    return out.astype(np.uint8)


def flip_lr_mask(
    mask: np.ndarray,
    *,
    horizontal: bool,
    vertical: bool,
) -> np.ndarray:
    """Flip a tri-valued L/R mask for display.

    Horizontal flip mirrors the array AND swaps the value labels 1↔2
    (a left pixel that is mirrored across the vertical axis becomes a
    right pixel).  Vertical flip mirrors the array but leaves the labels
    intact (vertical flip does not change anatomical handedness).
    """
    out = np.asarray(mask, dtype=np.uint8)
    if horizontal:
        out = np.fliplr(out)
        # Swap 1↔2; leave 0 untouched.
        swapped = np.zeros_like(out)
        swapped[out == 1] = 2
        swapped[out == 2] = 1
        out = swapped
    if vertical:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def lr_mask_to_rgba(
    mask: np.ndarray,
    opacity: float,
    left_color: tuple[int, int, int] = (220, 60, 60),
    right_color: tuple[int, int, int] = (60, 130, 220),
) -> np.ndarray:
    """Convert a tri-valued L/R mask to an RGBA overlay.

    ``0`` pixels are fully transparent; ``1`` pixels are tinted *left_color*,
    ``2`` pixels are tinted *right_color*.  Alpha is scaled by *opacity*.
    """
    arr = np.asarray(mask, dtype=np.uint8)
    alpha = int(round(min(max(opacity, 0.0), 1.0) * 255))

    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    is_left = arr == 1
    is_right = arr == 2
    rgba[is_left, 0] = left_color[0]
    rgba[is_left, 1] = left_color[1]
    rgba[is_left, 2] = left_color[2]
    rgba[is_left, 3] = alpha
    rgba[is_right, 0] = right_color[0]
    rgba[is_right, 1] = right_color[1]
    rgba[is_right, 2] = right_color[2]
    rgba[is_right, 3] = alpha
    return rgba


def line_side_polygons(
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip the image rectangle (0, 0) → (width, height) by the line p0→p1.

    Returns ``(left_polygon, right_polygon)`` — two (N, 2) float arrays of
    polygon vertices in image-pixel coords.  "Left" follows the cross-product
    convention used by :func:`rasterize_lr_line` (negative side).

    Implementation: Sutherland–Hodgman convex polygon clip against each
    half-plane defined by the line.  Each clip is O(8) for the 4-vertex rect.

    If the line lies entirely outside the rect (or the polygon clips to
    zero vertices), the corresponding output is an empty (0, 2) array.
    """
    rect = np.array(
        [(0.0, 0.0), (float(width), 0.0),
         (float(width), float(height)), (0.0, float(height))],
        dtype=np.float64,
    )

    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    x0 = float(p0[0])
    y0 = float(p0[1])

    def cross(q: np.ndarray) -> float:
        return dx * (q[1] - y0) - dy * (q[0] - x0)

    def clip(poly: np.ndarray, keep_negative: bool) -> np.ndarray:
        if len(poly) == 0:
            return poly
        # sign chosen so that "in" is always ca <= 0:
        #   keep_negative → ca = cross(a),       in when cross <= 0
        #   keep_positive → ca = -cross(a),      in when cross >= 0
        sign = 1.0 if keep_negative else -1.0
        result: list[tuple[float, float]] = []
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            ca = cross(a) * sign
            cb = cross(b) * sign
            a_in = ca <= 0.0  # boundary counted as inside
            b_in = cb <= 0.0
            if a_in:
                result.append((float(a[0]), float(a[1])))
            if a_in != b_in:
                # Intersection of segment a→b with the line p0→p1
                denom = ca - cb
                if abs(denom) > 1e-12:
                    t = ca / denom
                else:
                    t = 0.0
                ix = float(a[0]) + t * (float(b[0]) - float(a[0]))
                iy = float(a[1]) + t * (float(b[1]) - float(a[1]))
                result.append((ix, iy))
        if not result:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array(result, dtype=np.float64)

    # "Left" = cross product is negative (matches rasterize_lr_line).
    left_poly = clip(rect, keep_negative=True)
    right_poly = clip(rect, keep_negative=False)
    return left_poly, right_poly
