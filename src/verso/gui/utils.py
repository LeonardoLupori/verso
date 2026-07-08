"""GUI utility helpers (no engine imports here)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QMessageBox, QWidget

_ICONS_DIR = Path(__file__).parent / "icons"


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


def colored_svg_pixmap(name: str, color: str, size: int) -> QPixmap:
    """Rasterize ``icons/<name>`` (with ``currentColor`` strokes) tinted ``color``.

    Rendered via ``QSvgRenderer`` into a ``size``x``size`` image, which scales
    the SVG's ``viewBox`` to the target crisply regardless of the file's
    intrinsic ``width``/``height`` (lucide icons ship a 24px box). Toolbar-sized
    icons wrap this in ``QIcon(colored_svg_pixmap(name, color, size))`` directly
    (e.g. ``_common.py::colored_icon``) — a ``QIcon`` only scales its source
    pixmap down cleanly, never up, so a size around 64 stays crisp even for
    16-24px buttons.
    """
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(image)


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


def warn_if_missing_dimensions(parent: QWidget, sections: list) -> bool:
    """Guard atlas-registration ops against sections with no image dimensions.

    Registration needs each section's working-resolution image size, cached as
    ``resolution_thumbnail_wh`` at import. A ``(0, 0)`` value means the project
    file is corrupt or was never fully imported — proceeding would raise deep in
    the QuickNII math. When any section is affected, a dialog naming the exact
    slices and the missing attribute is shown and this returns ``False`` so the
    caller can abort; otherwise it returns ``True``.
    """
    missing = [s for s in sections if min(s.resolution_thumbnail_wh) <= 0]
    if not missing:
        return True
    listed = "\n".join(f"  •  slice {s.slice_index}  (id {s.id})" for s in missing[:8])
    tail = "" if len(missing) <= 8 else f"\n  …and {len(missing) - 8} more"
    QMessageBox.critical(
        parent,
        "Project may be corrupt",
        "Cannot compute atlas registration: the working-resolution image "
        "dimensions (the 'resolution_thumbnail_wh' attribute) are missing for "
        f"these section(s):\n\n{listed}{tail}\n\n"
        "This attribute is populated when images are imported, so the project "
        "file may be corrupt or was never fully imported. Re-importing the "
        "affected images will regenerate it.",
    )
    return False
