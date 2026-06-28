"""Properties page for the Prep view (Flip + Mask)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from verso.engine.model.project import Section
from verso.gui.widgets.properties.sections import (
    FlipBox,
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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        self.flip = FlipBox()
        self.mask_box = MaskBox()

        layout.addWidget(self.flip)
        layout.addWidget(self.mask_box)
        layout.addStretch()

        self.save_bar = SaveBarBox()
        layout.addWidget(self.save_bar)

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self.flip.set_flip_h(False)
            self.flip.set_flip_v(False)
            return
        self.flip.set_flip_h(section.preprocessing.flip_horizontal)
        self.flip.set_flip_v(section.preprocessing.flip_vertical)
