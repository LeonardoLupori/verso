"""Properties page for the Overview view."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from verso.engine.model.project import Section

# Long side of the section preview, in px. Matches the small filmstrip tier so
# the already-loaded pixmap is reused as-is (no recompositing or upscaling).
_PREVIEW_MAX_SIDE = 150


class OverviewPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._title = QLabel("No section selected")
        self._title.setWordWrap(True)
        self._title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title)

        # Tiny section preview, reusing the filmstrip's cached tile.
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(_PREVIEW_MAX_SIDE)
        self._preview.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #383838; border-radius: 6px;"
        )
        layout.addWidget(self._preview)

        info_box = QGroupBox("Section info")
        info_layout = QFormLayout(info_box)
        self._lbl_file = QLabel("-")
        self._lbl_file.setWordWrap(True)
        self._lbl_serial = QLabel("-")
        info_layout.addRow("File:", self._lbl_file)
        info_layout.addRow("Serial #:", self._lbl_serial)
        layout.addWidget(info_box)

        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._title.setText("No section selected")
            self._lbl_file.setText("-")
            self._lbl_serial.setText("-")
            self.set_preview(None)
            return
        self._title.setText(os.path.basename(section.original_path))
        self._lbl_file.setText(section.original_path)
        self._lbl_serial.setText(str(section.slice_index))

    def set_preview(self, pixmap: QPixmap | None) -> None:
        """Show a small section image (or a placeholder when unavailable)."""
        if pixmap is None or pixmap.isNull():
            self._preview.clear()
            self._preview.setText("—")
            return
        scaled = pixmap.scaled(
            _PREVIEW_MAX_SIDE,
            _PREVIEW_MAX_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)
