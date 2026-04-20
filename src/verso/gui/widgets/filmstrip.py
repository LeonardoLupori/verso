"""Horizontal filmstrip of section thumbnails for Prep and Align/Warp views."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus

_THUMB_SIZE = 120  # px long side

# Status border colours
_COLOUR = {
    AlignmentStatus.NOT_STARTED: "#888888",
    AlignmentStatus.IN_PROGRESS: "#888888",
    AlignmentStatus.COMPLETE: "#4CAF50",
}
_BORDER_W = 3
_BADGE_SIZE = 16


class _ThumbButton(QLabel):
    """Single thumbnail tile in the filmstrip."""

    clicked = pyqtSignal(int)  # section index

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._status = AlignmentStatus.NOT_STARTED
        self._align_stored = False  # alignment.COMPLETE when in align-view mode
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 2 * _BORDER_W, _THUMB_SIZE + 2 * _BORDER_W)

        # Small "stored" badge — green checkmark shown in align-view mode
        self._badge = QLabel("✓", self)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedSize(_BADGE_SIZE, _BADGE_SIZE)
        self._badge.setStyleSheet(
            "background: #4CAF50; color: #fff; font-size: 9px; font-weight: bold;"
            " border: none; border-radius: 2px;"
        )
        self._badge.move(
            self.width() - _BADGE_SIZE - _BORDER_W,
            self.height() - _BADGE_SIZE - _BORDER_W,
        )
        self._badge.hide()

        self._set_placeholder()

    def set_thumbnail(self, pixmap: QPixmap, status: AlignmentStatus) -> None:
        self._status = status
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

    def set_align_stored(self, is_stored: bool) -> None:
        """Toggle the stored-alignment badge (shown only in align-view mode)."""
        self._align_stored = is_stored
        self._badge.setVisible(is_stored)
        self._apply_border()

    def _set_placeholder(self) -> None:
        px = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        px.fill(QColor("#2a2a2a"))
        self.setPixmap(px)
        self._apply_border()

    def _apply_border(self) -> None:
        if self._align_stored:
            # In align-view mode: stored sections always show green, selection adds white
            colour = "#4CAF50"
            width = _BORDER_W + 1 if self._selected else _BORDER_W
        elif self._selected:
            colour = "#FFFFFF"
            width = _BORDER_W
        else:
            colour = _COLOUR.get(self._status, "#888888")
            width = _BORDER_W
        self.setStyleSheet(
            f"border: {width}px solid {colour}; background: transparent;"
        )
        # Keep badge on top
        self._badge.raise_()

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
        self._sections: list = []
        self._current: int = 0
        self._align_mode: bool = False
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
        self._sections = sections

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

            try:
                from verso.engine.io.image_io import load_filmstrip_thumbnail
                from verso.gui.utils import ndarray_to_pixmap
                thumb_arr = load_filmstrip_thumbnail(section)
                if thumb_arr is not None:
                    btn.set_thumbnail(ndarray_to_pixmap(thumb_arr), status)
                else:
                    btn._status = status
                    btn._apply_border()
            except Exception:
                btn._status = status
                btn._apply_border()

            self._row.insertWidget(self._row.count() - 1, btn)
            self._buttons.append(btn)

        self._apply_align_indicators()
        self._highlight(self._current)

    def set_align_mode(self, is_align: bool) -> None:
        """Switch between align-view mode (green stored badges) and normal mode."""
        self._align_mode = is_align
        self._apply_align_indicators()

    def refresh_stored(self) -> None:
        """Re-apply stored-alignment indicators after alignment changes."""
        self._apply_align_indicators()

    def _apply_align_indicators(self) -> None:
        for i, btn in enumerate(self._buttons):
            if i < len(self._sections) and self._align_mode:
                is_stored = (
                    self._sections[i].alignment.status == AlignmentStatus.COMPLETE
                )
                btn.set_align_stored(is_stored)
            else:
                btn.set_align_stored(False)

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
