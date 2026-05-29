"""Properties page for the Align view."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from verso.engine.model.project import Section
from verso.gui.widgets.properties.sections import AlignActionsBox, APPlotBox, OverlayBox


class AlignPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        self.overlay = OverlayBox()
        self.actions = AlignActionsBox()
        self.ap_plot = APPlotBox()

        layout.addWidget(self.overlay)
        layout.addWidget(self.actions)
        layout.addWidget(self.ap_plot)
        layout.addStretch()

    def update_section(self, _section: Section | None) -> None:
        pass
