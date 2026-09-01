"""Colormap lookup tables for recoloring exported section images.

Built on OpenCV's built-in colormaps so VERSO gains no new dependency. A
colormap turns the composited RGB section into a single-channel luminance image
and maps it through a 256-entry RGB LUT, optionally reversed.

``NONE`` is the identity: the section is written exactly as before, so the
default export path is unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Sentinel meaning "leave the composited RGB untouched".
NONE = "none"

#: Display name → OpenCV ``COLORMAP_*`` code, in the order shown in the GUI.
#: Perceptually uniform maps first, then the classic ones.
_CV_COLORMAPS: dict[str, int] = {
    "Gray": cv2.COLORMAP_BONE,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Magma": cv2.COLORMAP_MAGMA,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Cividis": cv2.COLORMAP_CIVIDIS,
    "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET,
    "Hot": cv2.COLORMAP_HOT,
}


def colormap_names() -> list[str]:
    """Return the selectable colormap names, ``NONE`` first.

    Returns:
        Names accepted by :func:`apply_colormap`, in GUI display order.
    """
    return [NONE, *_CV_COLORMAPS]


def colormap_lut(name: str, *, invert: bool = False) -> np.ndarray:
    """Return the ``(256, 3)`` uint8 RGB lookup table for *name*.

    Args:
        name: A name from :func:`colormap_names` (case-insensitive).
        invert: Reverse the LUT, so dark input maps to the bright end.

    Returns:
        uint8 ``(256, 3)`` RGB table indexed by luminance.

    Raises:
        KeyError: If *name* is not a known colormap.
    """
    match = {k.lower(): k for k in _CV_COLORMAPS}.get(name.strip().lower())
    if match is None:
        raise KeyError(f"Unknown colormap: {name!r}")
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    # applyColorMap returns BGR; reverse the last axis to get RGB.
    lut = cv2.applyColorMap(ramp, _CV_COLORMAPS[match])[:, 0, ::-1].copy()
    return lut[::-1].copy() if invert else lut


def apply_colormap(rgb: np.ndarray, name: str, *, invert: bool = False) -> np.ndarray:
    """Recolor a uint8 RGB image through a colormap.

    The image is first reduced to luminance, so a multi-channel composite
    collapses to one intensity scale before being mapped. ``name`` of
    :data:`NONE` (or empty) returns *rgb* unchanged.

    Args:
        rgb: uint8 ``(H, W, 3)`` RGB image.
        name: A name from :func:`colormap_names`.
        invert: Reverse the colormap.

    Returns:
        uint8 ``(H, W, 3)`` RGB image.
    """
    if not name or name.strip().lower() == NONE:
        return rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return colormap_lut(name, invert=invert)[gray]
