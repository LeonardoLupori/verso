"""Properties page for the Annotate view.

D2 scaffolding: an empty placeholder. The annotation manager (create / import /
delete, the annotation list, per-annotation colour/opacity/visibility/title, and
the Save annotations bar) is added in a later deliverable.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from verso.engine.model.project import Section


class AnnotatePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        placeholder = QLabel("Annotations")
        placeholder.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(placeholder)
        layout.addStretch()

    def update_section(self, section: Section | None) -> None:
        """No section-derived controls yet (annotations are project-global)."""
