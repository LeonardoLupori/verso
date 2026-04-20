"""GUI utility helpers (no engine imports here)."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a uint8 H×W×3 (or H×W) numpy array to a QPixmap.

    The array must be uint8 and C-contiguous.
    """
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)

    h, w = image.shape[:2]

    if image.ndim == 2:
        qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif image.shape[2] == 3:
        qimg = QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    elif image.shape[2] == 4:
        qimg = QImage(image.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    else:
        return QPixmap()

    # .copy() detaches from the numpy buffer so the pixmap survives array GC
    return QPixmap.fromImage(qimg.copy())
