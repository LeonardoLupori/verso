"""Properties page for the Overview view."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from verso.engine.model.project import Section


class OverviewPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._title = QLabel("No section selected")
        self._title.setWordWrap(True)
        self._title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title)

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
            return
        self._title.setText(os.path.basename(section.original_path))
        self._lbl_file.setText(section.original_path)
        self._lbl_serial.setText(str(section.slice_index))
