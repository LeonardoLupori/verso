"""GUI utility helpers (no engine imports here)."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMessageBox, QWidget


def require[T](value: T | None) -> T:
    """Narrow a value that the type stubs mark Optional but is never None here.

    PyQt accessors such as ``QWidget.layout()``, ``QMainWindow.menuBar()`` and
    ``QMainWindow.statusBar()`` are typed to return ``... | None`` even though
    they always return a live object in our usage. Wrapping the call narrows the
    type for the checker and fails loudly if the assumption is ever violated.
    """
    if value is None:  # pragma: no cover - defensive
        raise RuntimeError("expected a non-None value")
    return value


def ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a uint8 HxWx3 (or HxW) numpy array to a QPixmap.

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


def warn_errors(parent: QWidget, title: str, errors: list[str], lead: str) -> None:
    """Show a warning listing up to 8 errors, with an "…and N more" tail.

    ``lead`` is the text shown above the error list. No-op when ``errors`` is
    empty so callers can invoke unconditionally.
    """
    if not errors:
        return
    preview = "\n".join(errors[:8])
    suffix = "" if len(errors) <= 8 else f"\n...and {len(errors) - 8} more"
    QMessageBox.warning(parent, title, f"{lead}\n\n{preview}{suffix}")
