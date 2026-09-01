"""Tonal adjustment (black point / white point / gamma) for exported images.

A :class:`Levels` is applied to the composited RGB section *before* any colormap
(see :mod:`verso.engine.colormaps`), so the black/white points choose which part
of the intensity range the colormap spends its colors on.

The transform is the same one :func:`verso.engine.preprocessing.composite_channels`
uses for its per-channel gamma — ``255 · norm**gamma`` with ``gamma == 1``
linear — applied here through a 256-entry LUT so a full-resolution export costs
one table lookup per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bounds for the gamma control, mirrored by the GUI slider.
MIN_GAMMA = 0.1
MAX_GAMMA = 5.0


@dataclass
class Levels:
    """Black point, white point and gamma for an exported section image.

    Attributes:
        black: Input value mapped to 0. Values at or below it clip to black.
        white: Input value mapped to 255. Values at or above it clip to white.
        gamma: Exponent applied to the normalized value; ``1.0`` is linear,
            below 1 brightens the midtones, above 1 darkens them.
    """

    black: int = 0
    white: int = 255
    gamma: float = 1.0

    @property
    def is_identity(self) -> bool:
        """True when this leaves an image untouched (the export default)."""
        return self.black == 0 and self.white == 255 and self.gamma == 1.0

    def normalized(self) -> Levels:
        """Return a copy with the fields clamped to usable ranges.

        Keeps a degenerate window (``white <= black``) from producing a
        divide-by-zero: the white point is pushed to at least ``black + 1``.
        """
        black = int(min(max(self.black, 0), 255))
        white = int(min(max(self.white, 0), 255))
        if white <= black:
            white = min(255, black + 1)
            black = min(black, white - 1)
        gamma = float(min(max(self.gamma, MIN_GAMMA), MAX_GAMMA))
        return Levels(black=black, white=white, gamma=gamma)


def levels_lut(levels: Levels) -> np.ndarray:
    """Return the ``(256,)`` uint8 lookup table implementing *levels*.

    Args:
        levels: The adjustment to encode. Clamped via :meth:`Levels.normalized`.

    Returns:
        uint8 ``(256,)`` table indexed by input intensity.
    """
    lv = levels.normalized()
    x = np.arange(256, dtype=np.float32)
    norm = np.clip((x - lv.black) / float(lv.white - lv.black), 0.0, 1.0)
    if lv.gamma != 1.0:
        norm = np.power(norm, lv.gamma)
    return np.round(norm * 255.0).astype(np.uint8)


def apply_levels(image: np.ndarray, levels: Levels | None) -> np.ndarray:
    """Apply *levels* to a uint8 image, per channel.

    ``None`` or an identity adjustment returns *image* unchanged, so the default
    export path does no work.

    Args:
        image: uint8 array of any shape (typically ``(H, W, 3)`` RGB).
        levels: The adjustment, or ``None``.

    Returns:
        uint8 array of the same shape.
    """
    if levels is None or levels.is_identity:
        return image
    return levels_lut(levels)[image]


def sample_background_level(rgb: np.ndarray, x: int, y: int, *, radius: int = 2) -> int:
    """Return the background level sampled around ``(x, y)``.

    Takes the median over a ``(2·radius + 1)`` square so a single noisy pixel
    cannot set the level. Used by the export dialog's "pick background" tool,
    where the user clicks an empty area of the preview.

    Each pixel is reduced by its **brightest** channel, not by an average of the
    three. :func:`apply_levels` runs one LUT over R, G and B alike, so it is that
    brightest channel that decides whether the background actually clips: a
    green-tinted section with a background of ``(0, 34, 0)`` needs a level of 34,
    while the RGB mean would report 11 and leave the background visible.

    Args:
        rgb: uint8 ``(H, W, 3)`` RGB image.
        x: Column of the clicked pixel.
        y: Row of the clicked pixel.
        radius: Half-width of the sampled square, in pixels.

    Returns:
        The median per-pixel channel maximum (0–255) over the patch.

    Raises:
        IndexError: If ``(x, y)`` falls outside *rgb*.
    """
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise IndexError(f"({x}, {y}) is outside the {w}x{h} image")
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    patch = rgb[y0:y1, x0:x1]
    per_pixel = patch.reshape(-1, patch.shape[-1]).max(axis=1)
    return round(float(np.median(per_pixel)))


def is_bright_background(rgb: np.ndarray, level: int) -> bool:
    """True when *level* sits in the bright half of *rgb*'s range.

    Distinguishes a brightfield/histology section — bright background, dark
    stain — from a fluorescence one. It decides which end of the levels window a
    picked background belongs to: on a bright-background image the background is
    the *maximum*, and clamping it to the minimum instead would crush the whole
    section to black.

    The midpoint comes from the 1st/99th percentiles rather than min/max so a
    few stray hot or dead pixels cannot move it.

    Args:
        rgb: uint8 ``(H, W, 3)`` RGB image.
        level: A sampled background level, as returned by
            :func:`sample_background_level`.

    Returns:
        Whether *level* is above the image's midrange.
    """
    per_pixel = rgb.reshape(-1, rgb.shape[-1]).max(axis=1)
    low, high = np.percentile(per_pixel, [1.0, 99.0])
    return level > (float(low) + float(high)) / 2.0
