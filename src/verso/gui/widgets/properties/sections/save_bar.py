"""Local-changes bar pinned at the bottom of each view's properties page.

Local edits in a view (Prep / Align / Warp) are drafts — they only become
part of the slice's persistent state when the user clicks **Save** here (or
hits Ctrl+S).  Three buttons act on the current slice/view:

- **Save** — commit the unsaved edits to disk.
- **Clear edits** — revert the unsaved edits back to the last-saved version
  (or to default if this slice/view was never saved); the saved file is left
  untouched.
- **Reset** — wipe both saved and unsaved changes and restore the default.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from verso.gui.widgets.properties._common import colored_icon


class SaveBarBox(QGroupBox):
    save_requested = pyqtSignal()
    revert_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Local changes")

        self._dirty = False
        self._has_persisted = False

        self._save_btn = QPushButton("Save")
        self._save_btn.setIcon(colored_icon("save.svg", "#ffffff"))
        self._save_btn.setIconSize(QSize(14, 14))
        self._save_btn.setToolTip("Save local changes (also Ctrl+S)")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested)

        self._revert_btn = QPushButton("Clear edits")
        self._revert_btn.setIcon(colored_icon("rotate-ccw.svg", "#ffffff"))
        self._revert_btn.setIconSize(QSize(14, 14))
        self._revert_btn.setToolTip(
            "Discard unsaved edits and go back to the last saved version",
        )
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert_requested)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setIcon(colored_icon("circle-x.svg", "#ffffff"))
        self._reset_btn.setIconSize(QSize(14, 14))
        self._reset_btn.setToolTip(
            "Erase saved and unsaved changes for this slice and restore defaults"
        )
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self.reset_requested)

        # Clear edits / Reset are the secondary actions on top; Save is the
        # primary action — full-width — on the bottom row, same height as the rest.
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.addWidget(self._revert_btn, 0, 0)
        layout.addWidget(self._reset_btn, 0, 1)
        layout.addWidget(self._save_btn, 1, 0, 1, 2)

    def set_dirty(self, dirty: bool) -> None:
        """Reflect whether the view has unsaved edits."""
        self._dirty = bool(dirty)
        self._save_btn.setEnabled(self._dirty)
        self._revert_btn.setEnabled(self._dirty)
        self._refresh_reset()

    def set_reset_enabled(self, enabled: bool) -> None:
        """Reflect whether the slice has persisted state to wipe."""
        self._has_persisted = bool(enabled)
        self._refresh_reset()

    def _refresh_reset(self) -> None:
        # Reset is meaningful whenever there is anything non-default to clear,
        # whether it lives on disk (persisted) or only in memory (unsaved).
        self._reset_btn.setEnabled(self._dirty or self._has_persisted)
