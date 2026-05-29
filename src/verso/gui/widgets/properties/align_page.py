"""Properties page for the Align view."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from verso.engine.model.project import Section
from verso.gui.widgets.properties.sections import APPlotBox, OverlayBox, SaveBarBox


class AlignPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        self.overlay = OverlayBox()
        self.ap_plot = APPlotBox()

        layout.addWidget(self.overlay)
        layout.addWidget(self.ap_plot)
        layout.addStretch()

        self.save_bar = SaveBarBox()
        outer.addWidget(self.save_bar)

    def update_section(self, _section: Section | None) -> None:
        pass
