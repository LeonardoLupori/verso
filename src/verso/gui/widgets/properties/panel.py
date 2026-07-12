"""Context-sensitive right-side properties panel.

Contains a QStackedWidget with one page per view mode.  Each page exposes
its own sections as public attributes (e.g. ``panel.prep.mask_box``).  Wire
signals directly to the sections — this class no longer re-exports them.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from verso.gui.widgets.properties.align_page import AlignPage
from verso.gui.widgets.properties.annotate_page import AnnotatePage
from verso.gui.widgets.properties.overview_page import OverviewPage
from verso.gui.widgets.properties.prep_page import PrepPage
from verso.gui.widgets.properties.warp_page import WarpPage


class PropertiesPanel(QWidget):
    """Stack of per-mode pages; routes mode switching and section updates."""

    _MODES = ("overview", "prep", "align", "warp", "annotate")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self.overview = OverviewPage()
        self.prep = PrepPage()
        self.align = AlignPage()
        self.warp = WarpPage()
        self.annotate = AnnotatePage()

        for page in (self.overview, self.prep, self.align, self.warp, self.annotate):
            self._stack.addWidget(page)
        layout.addWidget(self._stack)

    def set_mode(self, mode: str) -> None:
        self._stack.setCurrentIndex(self._MODES.index(mode))
        # Push the newly-active page's overlay settings to the canvas so it
        # matches the buttons the user is now looking at.
        if mode == "align":
            self.align.overlay.emit_current_state()
        elif mode == "warp":
            self.warp.overlay.emit_current_state()

    def update_section(self, section, mode: str) -> None:
        page = getattr(self, mode)
        page.update_section(section)
