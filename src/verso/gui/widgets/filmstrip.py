"""Horizontal filmstrip of section thumbnails for Prep and Align/Warp views."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from verso.engine.io.image_io import WORKING_SCALE
from verso.gui.utils import require

_THUMB_SIZE = 80  # px long side
_BORDER_W = 3
_ROW_MARGIN = 4  # padding around the thumbnail row
_DOT_DIAMETER = 10  # status dot in the top-right corner
_DOT_MARGIN = 4


class _ThumbnailLoader(QObject):
    """Background worker: loads one filmstrip thumbnail per section.

    Emits ``thumbnail_ready(index, pixmap)`` for each section as it finishes.
    Checks ``_abort`` before every section so a new ``populate()`` call can
    cancel an in-flight load quickly.
    """

    thumbnail_ready = pyqtSignal(int, QPixmap)  # (section_index, pixmap)
    finished = pyqtSignal()

    def __init__(self, sections: list, channels: list, working_scale: float) -> None:
        super().__init__()
        self._sections = list(sections)  # snapshot — avoid races with caller
        self._channels = list(channels)
        self._working_scale = working_scale
        self._abort = False

    def stop(self) -> None:
        """Request cancellation. Safe to call from any thread."""
        self._abort = True

    def run(self) -> None:
        from verso.engine.io.image_io import load_filmstrip_thumbnail
        from verso.gui.utils import ndarray_to_pixmap

        for i, section in enumerate(self._sections):
            if self._abort:
                break
            try:
                arr = load_filmstrip_thumbnail(section, self._working_scale, self._channels)
                if arr is not None:
                    self.thumbnail_ready.emit(i, ndarray_to_pixmap(arr))
            except Exception:
                pass
        self.finished.emit()


class _HScrollArea(QScrollArea):
    """Scroll area whose mouse wheel scrolls horizontally, never vertically."""

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = require(self.horizontalScrollBar())
        bar.setValue(bar.value() - delta)
        event.accept()


class _ThumbButton(QLabel):
    """Single thumbnail tile in the filmstrip."""

    clicked = pyqtSignal(int)  # section index

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._status_color: str | None = None
        self._thumbnail: QPixmap | None = None  # original (unscaled) loaded tile
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 2 * _BORDER_W, _THUMB_SIZE + 2 * _BORDER_W)
        self._set_placeholder()

    def set_status_color(self, color: str | None) -> None:
        """Set the top-right status-dot colour (None hides the dot)."""
        if color == self._status_color:
            return
        self._status_color = color
        self.update()

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._thumbnail = pixmap
        scaled = pixmap.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._apply_border()

    def thumbnail(self) -> QPixmap | None:
        """Return the original loaded tile, or None if still a placeholder."""
        return self._thumbnail

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
        self.setStyleSheet(f"border: {_BORDER_W}px solid {colour}; background: transparent;")

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        if self._status_color is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(QColor(self._status_color))
        x = self.width() - _BORDER_W - _DOT_MARGIN - _DOT_DIAMETER
        y = _BORDER_W + _DOT_MARGIN
        painter.drawEllipse(x, y, _DOT_DIAMETER, _DOT_DIAMETER)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)


class Filmstrip(QWidget):
    """Scrollable horizontal strip of section thumbnails."""

    section_selected = pyqtSignal(int)
    thumbnail_loaded = pyqtSignal(int)  # a section's tile finished loading

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: list[_ThumbButton] = []
        self._current: int = 0
        self._loader_thread: QThread | None = None
        self._loader: _ThumbnailLoader | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        scroll = _HScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: #1e1e1e;")
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(_ROW_MARGIN, _ROW_MARGIN, _ROW_MARGIN, _ROW_MARGIN)
        self._row.setSpacing(4)
        self._row.addStretch()

        # Height that exactly fits one row of thumbnails plus the horizontal
        # scrollbar — derived from the real scrollbar extent so there is no
        # vertical slop and the bottom dock has no resize handle.
        scrollbar_h = require(scroll.horizontalScrollBar()).sizeHint().height()
        strip_height = _THUMB_SIZE + 2 * _BORDER_W + 2 * _ROW_MARGIN + scrollbar_h
        self.setFixedHeight(strip_height)
        scroll.setFixedHeight(strip_height)

        scroll.setWidget(self._container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)
        self._scroll = scroll

    def populate(
        self,
        sections: list,
        channels: list | None = None,
        working_scale: float = WORKING_SCALE,
    ) -> None:
        """Rebuild thumbnails from a list of Section objects.

        Creates all placeholder buttons immediately (non-blocking), then loads
        thumbnails in a background thread, updating each button as it finishes.

        Args:
            sections: section list to render.
            channels: project-level :class:`ChannelSpec` list used to composite
                the cached multichannel thumbnail to RGB.
            working_scale: project's :attr:`Project.working_scale`, forwarded to
                thumbnail regeneration so every section keeps one scale.
        """
        channels = channels or []
        self._cancel_loader()

        for btn in self._buttons:
            self._row.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        if not sections:
            self._highlight(self._current)
            return

        # Create all placeholder buttons immediately — no I/O.
        for i in range(len(sections)):
            btn = _ThumbButton(i)
            btn.clicked.connect(self._on_thumb_clicked)
            self._row.insertWidget(self._row.count() - 1, btn)
            self._buttons.append(btn)
        self._highlight(self._current)

        # Load thumbnails in the background; update each button as it arrives.
        loader = _ThumbnailLoader(sections, channels, working_scale)
        thread = QThread()  # No parent — we control lifetime explicitly via shutdown()
        loader.moveToThread(thread)
        thread.started.connect(loader.run)
        loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        loader.finished.connect(thread.quit)
        self._loader = loader
        self._loader_thread = thread
        thread.start()

    def _cancel_loader(self) -> None:
        """Stop any running loader, blocking until the current image load finishes."""
        if self._loader is not None:
            self._loader.stop()
        if self._loader_thread is not None:
            try:
                if self._loader_thread.isRunning():
                    self._loader_thread.quit()
                    self._loader_thread.wait()
            except RuntimeError:
                pass  # C++ object already deleted — thread has already finished
        self._loader = None
        self._loader_thread = None

    def shutdown(self) -> None:
        """Stop the background loader. Must be called before the widget is destroyed."""
        self._cancel_loader()

    def _on_thumbnail_ready(self, index: int, pixmap: QPixmap) -> None:
        """Slot: called on the main thread when the loader finishes one thumbnail."""
        # Guard against stale signals arriving after a new populate() cleared the list.
        if 0 <= index < len(self._buttons):
            self._buttons[index].set_thumbnail(pixmap)
            self.thumbnail_loaded.emit(index)

    def thumbnail_pixmap(self, index: int) -> QPixmap | None:
        """Return the already-loaded tile for a section, or None if not ready.

        Reuses the pixmap held by the filmstrip button — no disk I/O or
        recompositing — so callers (e.g. the Overview properties preview) get the
        section image for free.
        """
        if 0 <= index < len(self._buttons):
            return self._buttons[index].thumbnail()
        return None

    def set_current(self, index: int) -> None:
        self._highlight(index)
        self._current = index
        if 0 <= index < len(self._buttons):
            QTimer.singleShot(0, lambda idx=index: self._center_on_index(idx))

    def _center_on_index(self, index: int) -> None:
        if not (0 <= index < len(self._buttons)):
            return
        btn = self._buttons[index]
        bar = require(self._scroll.horizontalScrollBar())
        viewport_w = require(self._scroll.viewport()).width()
        button_center = btn.x() + btn.width() // 2
        target = button_center - viewport_w // 2
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))

    def set_statuses(self, colors: Sequence[str | None]) -> None:
        """Set every thumbnail's status-dot colour from a per-section list."""
        for i, btn in enumerate(self._buttons):
            btn.set_status_color(colors[i] if i < len(colors) else None)

    def set_status_color(self, index: int, color: str | None) -> None:
        """Set a single thumbnail's status-dot colour."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].set_status_color(color)

    def _highlight(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.set_selected(i == index)

    def _on_thumb_clicked(self, index: int) -> None:
        self.set_current(index)
        self.section_selected.emit(index)
