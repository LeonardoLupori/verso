"""Properties page for the Overview view."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import Section

# Long side of the section preview, in px. Matches the small filmstrip tier so
# the already-loaded pixmap is reused as-is (no recompositing or upscaling).
_PREVIEW_MAX_SIDE = 150


class _ElidingLabel(QLabel):
    """Single-line label that elides overflow with "…" and keeps the full
    text in its tooltip, so long paths never widen or wrap the panel."""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        # Let the label shrink below its text's natural width (it elides).
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text if text != "-" else "")
        self._update_elision()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elision()

    def _update_elision(self) -> None:
        metrics = self.fontMetrics()
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, self.width())
        super().setText(elided)


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
        self._lbl_file = _ElidingLabel("-")
        self._lbl_id = QLabel("-")
        self._lbl_slice = QLabel("-")
        self._lbl_res_original = QLabel("-")
        self._lbl_res_thumbnail = QLabel("-")
        self._lbl_control_points = QLabel("-")
        info_layout.addRow("Section ID:", self._lbl_id)
        info_layout.addRow("Slice index:", self._lbl_slice)
        info_layout.addRow("# Control points:", self._lbl_control_points)
        info_layout.addRow("Thumbnail resolution:", self._lbl_res_thumbnail)
        info_layout.addRow("Original resolution:", self._lbl_res_original)
        info_layout.addRow("Original File:", self._lbl_file)
        layout.addWidget(info_box)

        layout.addStretch()

    @staticmethod
    def _dims_text(size_wh: tuple[int, int]) -> str:
        """Return "W × H px" for a (width, height) pair, or "-" if unset."""
        w, h = size_wh
        if w <= 0 or h <= 0:
            return "-"
        return f"{w} × {h} px"

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._title.setText("No section selected")
            self._lbl_file.setText("-")
            self._lbl_id.setText("-")
            self._lbl_slice.setText("-")
            self._lbl_res_original.setText("-")
            self._lbl_res_thumbnail.setText("-")
            self._lbl_control_points.setText("-")
            self.set_preview(None)
            return
        self._title.setText(os.path.basename(section.original_path))
        self._lbl_file.setText(section.original_path)
        self._lbl_id.setText(section.id)
        self._lbl_slice.setText(str(section.slice_index))
        self._lbl_res_original.setText(self._dims_text(section.resolution_original_wh))
        self._lbl_res_thumbnail.setText(self._dims_text(section.resolution_thumbnail_wh))
        self._lbl_control_points.setText(str(len(section.warp.control_points)))

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
