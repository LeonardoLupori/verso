"""Horizontal filmstrip of section thumbnails for Prep and Align/Warp views."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_THUMB_SIZE = 100  # px long side
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
        self._set_placeholder()

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._apply_border()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_border()

    def _set_placeholder(self) -> None:
        px = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        px.fill(QColor("#2a2a2a"))
        self.setPixmap(px)
        self._apply_border()

    def _apply_border(self) -> None:
        colour = "#FFFFFF" if self._selected else "#555555"
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

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
        outer.addStretch()
        self._scroll = scroll

    def populate(self, sections: list, channels: list | None = None) -> None:
        """Rebuild thumbnails from a list of Section objects.

        Args:
            sections: section list to render.
            channels: project-level :class:`ChannelSpec` list used to composite
                the cached multichannel thumbnail to RGB.
        """
        channels = channels or []

        for btn in self._buttons:
            self._row.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        for i, section in enumerate(sections):
            btn = _ThumbButton(i)
            btn.clicked.connect(self._on_thumb_clicked)

            try:
                from verso.engine.io.image_io import load_filmstrip_thumbnail
                from verso.gui.utils import ndarray_to_pixmap

                thumb_arr = load_filmstrip_thumbnail(section, channels)
                if thumb_arr is not None:
                    btn.set_thumbnail(ndarray_to_pixmap(thumb_arr))
            except Exception:
                pass

            self._row.insertWidget(self._row.count() - 1, btn)
            self._buttons.append(btn)

        self._highlight(self._current)

    def set_current(self, index: int) -> None:
        self._highlight(index)
        self._current = index
        if 0 <= index < len(self._buttons):
            QTimer.singleShot(0, lambda idx=index: self._center_on_index(idx))

    def _center_on_index(self, index: int) -> None:
        if not (0 <= index < len(self._buttons)):
            return
        btn = self._buttons[index]
        bar = self._scroll.horizontalScrollBar()
        viewport_w = self._scroll.viewport().width()
        button_center = btn.x() + btn.width() // 2
        target = button_center - viewport_w // 2
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))

    def _highlight(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.set_selected(i == index)

    def _on_thumb_clicked(self, index: int) -> None:
        self.set_current(index)
        self.section_selected.emit(index)
