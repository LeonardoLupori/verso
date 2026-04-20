"""Prep view — canvas for section preprocessing (masks, flipping)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import Section
from verso.gui.widgets.canvas import ImageCanvas


class PrepView(QWidget):
    """Canvas view for the Prep (mask drawing / flip) step."""

    section_modified = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section: Section | None = None
        self._raw_image = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left tool palette
        self._toolbar = self._make_toolbar()
        layout.addWidget(self._toolbar)

        # Central canvas
        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas, stretch=1)

    def _make_toolbar(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(48)
        container.setStyleSheet("background: #2a2a2a;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 8, 4, 8)
        v.setSpacing(4)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tool_group = QButtonGroup()
        tools = [
            ("✏", "Pen", "pen"),
            ("⌫", "Eraser", "eraser"),
            ("⬡", "Polygon", "polygon"),
            ("⬤", "Fill", "fill"),
        ]
        _btn_ss = (
            "QToolButton { color: #ccc; border-radius: 4px; }"
            "QToolButton:checked { background: #1e5a8a; }"
            "QToolButton:hover { background: #444; }"
        )
        from PyQt6.QtGui import QFont
        _icon_font = QFont()
        _icon_font.setPointSize(13)

        for icon, tip, name in tools:
            btn = QToolButton()
            btn.setText(icon)
            btn.setFont(_icon_font)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(_btn_ss)
            btn.setProperty("tool_name", name)
            self._tool_group.addButton(btn)
            v.addWidget(btn)

        if self._tool_group.buttons():
            self._tool_group.buttons()[0].setChecked(True)

        v.addStretch()

        _undo_font = QFont()
        _undo_font.setPointSize(14)
        undo_btn = QToolButton()
        undo_btn.setText("↩")
        undo_btn.setFont(_undo_font)
        undo_btn.setToolTip("Undo")
        undo_btn.setFixedSize(36, 36)
        undo_btn.setStyleSheet(
            "QToolButton { color: #ccc; border-radius: 4px; }"
            "QToolButton:hover { background: #444; }"
        )
        v.addWidget(undo_btn)

        return container

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def load_section(self, section: Section | None) -> None:
        self._section = section
        self._raw_image = None
        self._canvas.clear()
        if section is None:
            return
        from verso.engine.io.image_io import ensure_working_copy
        from PyQt6.QtWidgets import QMessageBox
        try:
            self._raw_image = ensure_working_copy(section)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return
        self._display_image()

    def _display_image(self) -> None:
        if self._raw_image is None:
            return
        import numpy as np
        img = self._raw_image
        if self._section and self._section.preprocessing.flip_horizontal:
            img = np.fliplr(img)
        self._canvas.set_background(np.ascontiguousarray(img))

    def refresh_display(self) -> None:
        """Re-render from cache — call after any preprocessing parameter change."""
        self._display_image()
