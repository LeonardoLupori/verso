"""Horizontal filmstrip of section thumbnails for Prep and Align/Warp views."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus

_THUMB_SIZE = 120  # px long side

# Status border colours
_COLOUR = {
    AlignmentStatus.NOT_STARTED: "#888888",
    AlignmentStatus.IN_PROGRESS: "#E6A817",
    AlignmentStatus.COMPLETE: "#4CAF50",
}
_BORDER_W = 3


class _ThumbButton(QLabel):
    """Single thumbnail tile in the filmstrip."""

    clicked = pyqtSignal(int)  # section index

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 2 * _BORDER_W, _THUMB_SIZE + 2 * _BORDER_W)
        self._set_placeholder(AlignmentStatus.NOT_STARTED)

    def set_thumbnail(self, pixmap: QPixmap, status: AlignmentStatus) -> None:
        scaled = pixmap.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._apply_border(status)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_border(None)  # re-draw with current status

    def _set_placeholder(self, status: AlignmentStatus) -> None:
        px = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        px.fill(QColor("#2a2a2a"))
        self.setPixmap(px)
        self._apply_border(status)

    def _apply_border(self, status: AlignmentStatus | None) -> None:
        colour = "#FFFFFF" if self._selected else _COLOUR.get(status or AlignmentStatus.NOT_STARTED, "#888888")
        self.setStyleSheet(
            f"border: {_BORDER_W}px solid {colour}; background: transparent;"
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)


class Filmstrip(QWidget):
    """Scrollable horizontal strip of section thumbnails."""

    section_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: list[_ThumbButton] = []
        self._current: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(_THUMB_SIZE + 2 * _BORDER_W + 16)  # thumb + border + scrollbar
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: #1e1e1e;")
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(4)
        self._row.addStretch()

        scroll.setWidget(self._container)
        outer.addWidget(scroll)
        self._scroll = scroll

    def populate(self, sections: list) -> None:
        """Rebuild thumbnails from a list of Section objects."""
        # Clear old buttons
        for btn in self._buttons:
            self._row.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        for i, section in enumerate(sections):
            btn = _ThumbButton(i)
            btn.clicked.connect(self._on_thumb_clicked)
            status = section.warp.status
            if status == AlignmentStatus.NOT_STARTED:
                status = section.alignment.status

            # Try to load a real thumbnail image
            try:
                from verso.engine.io.image_io import load_filmstrip_thumbnail
                from verso.gui.utils import ndarray_to_pixmap
                thumb_arr = load_filmstrip_thumbnail(section)
                if thumb_arr is not None:
                    btn.set_thumbnail(ndarray_to_pixmap(thumb_arr), status)
                else:
                    btn._apply_border(status)
            except Exception:
                btn._apply_border(status)

            self._row.insertWidget(self._row.count() - 1, btn)
            self._buttons.append(btn)

        self._highlight(self._current)

    def set_current(self, index: int) -> None:
        self._highlight(index)
        self._current = index
        # Scroll to make it visible
        if 0 <= index < len(self._buttons):
            self._scroll.ensureWidgetVisible(self._buttons[index])

    def _highlight(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.set_selected(i == index)

    def _on_thumb_clicked(self, index: int) -> None:
        self.set_current(index)
        self.section_selected.emit(index)
