"""Properties page for the Warp view."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from verso.engine.model.project import Section
from verso.gui.widgets.properties.sections import (
    ControlPointsBox,
    OverlayBox,
    SaveBarBox,
)


class WarpPage(QWidget):
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        self.overlay = OverlayBox()
        self.cp = ControlPointsBox()

        layout.addWidget(self.overlay)
        layout.addWidget(self.cp)
        layout.addStretch()

        # Wrap the save bar so it gets the same horizontal inset as the
        # group boxes inside the scroll area (without shrinking the scroll
        # viewport, which would clip the inner content).
        self.save_bar = SaveBarBox()
        save_bar_wrap = QHBoxLayout()
        save_bar_wrap.setContentsMargins(0, 0, 0, 0)
        save_bar_wrap.addWidget(self.save_bar)
        outer.addLayout(save_bar_wrap)

    def update_section(self, _section: Section | None) -> None:
        pass
