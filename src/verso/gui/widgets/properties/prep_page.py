"""Properties page for the Prep view (Flip + Mask + Hemisphere)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from verso.engine.model.project import Section
from verso.gui.widgets.properties.sections import (
    FlipBox,
    HemisphereBox,
    MaskBox,
    SaveBarBox,
)


class PrepPage(QWidget):
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

        self.flip = FlipBox()
        self.mask = MaskBox()
        self.hemisphere = HemisphereBox()

        layout.addWidget(self.flip)
        layout.addWidget(self.mask)
        layout.addWidget(self.hemisphere)
        layout.addStretch()

        # Wrap the save bar so it gets the same horizontal inset as the
        # group boxes inside the scroll area (without shrinking the scroll
        # viewport, which would clip the inner content).
        self.save_bar = SaveBarBox()
        save_bar_wrap = QHBoxLayout()
        save_bar_wrap.setContentsMargins(0, 0, 0, 0)
        save_bar_wrap.addWidget(self.save_bar)
        outer.addLayout(save_bar_wrap)

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self.flip.set_flip_h(False)
            self.flip.set_flip_v(False)
            return
        self.flip.set_flip_h(section.preprocessing.flip_horizontal)
        self.flip.set_flip_v(section.preprocessing.flip_vertical)
