"""Save / Clear bar pinned at the bottom of each view's properties page.

Local edits in a view (Prep / Align / Warp) are now drafts — they only
become part of the slice's persistent state when the user clicks Save or
Clear here (or hits Ctrl+S).  Changing view or slice silently discards
the draft.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from verso.gui.widgets.properties._common import colored_icon


class SaveBarBox(QGroupBox):
    save_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Unsaved changes")

        self._save_btn = QPushButton("Save")
        self._save_btn.setIcon(colored_icon("save.svg", "#ffffff"))
        self._save_btn.setIconSize(QSize(14, 14))
        self._save_btn.setToolTip(
            "Save local changes (also Ctrl+S)"
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setIcon(colored_icon("circle-x.svg", "#ffffff"))
        self._clear_btn.setIconSize(QSize(14, 14))
        self._clear_btn.setToolTip(
            "Wipe this slice's state for the current view and write the project"
        )
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self.clear_requested)

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.addWidget(self._clear_btn, 0, 1)
        layout.addWidget(self._save_btn, 0, 0)

    def set_dirty(self, dirty: bool) -> None:
        self._save_btn.setEnabled(bool(dirty))

    def set_clear_enabled(self, enabled: bool) -> None:
        self._clear_btn.setEnabled(bool(enabled))
