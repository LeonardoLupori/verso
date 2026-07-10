"""Properties page for the Annotate view — the annotation manager.

Composes two sections:
:class:`~verso.gui.widgets.properties.sections.manage_annotations.ManageAnnotationsBox`
(create/select/toggle/delete, exposed as :attr:`manager`) and
:class:`~verso.gui.widgets.properties.sections.edit_annotations.EditAnnotationsBox`
(edit the active one, exposed as :attr:`selected`). A Save button (mirrored by
Ctrl+S) persists the annotation set.

The page is a pure view: it emits intent signals and renders whatever
:meth:`set_annotations` is handed. All state lives in
:class:`~verso.gui.controllers.annotation_controller.AnnotationController`.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.annotation import Annotation
from verso.engine.model.project import Section
from verso.gui.widgets.properties._common import colored_icon
from verso.gui.widgets.properties.sections import EditAnnotationsBox, ManageAnnotationsBox


class AnnotatePage(QWidget):
    save_requested = pyqtSignal()

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

        self.manager = ManageAnnotationsBox()
        self.selected = EditAnnotationsBox()

        layout.addWidget(self.manager)
        layout.addWidget(self.selected)
        layout.addStretch()

        self._save_box = self._build_save_box()
        layout.addWidget(self._save_box)

        self.set_annotations([], -1)
        self.set_dirty(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Local changes")
        v = QVBoxLayout(box)
        self._save_btn = QPushButton("Save annotations")
        self._save_btn.setIcon(colored_icon("save.svg", "#ffffff"))
        self._save_btn.setIconSize(QSize(14, 14))
        self._save_btn.setToolTip("Write annotations to the project's annotations/ folder")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested)
        v.addWidget(self._save_btn)
        return box

    # ------------------------------------------------------------------
    # Population (driven by AnnotationController)
    # ------------------------------------------------------------------

    def set_annotations(self, annotations: list[Annotation], active_index: int) -> None:
        """Rebuild the table and reflect the active annotation's controls."""
        self.manager.set_annotations(annotations, active_index)

        has_active = 0 <= active_index < len(annotations)
        self.selected.setEnabled(has_active)
        if has_active:
            self.selected.update_selected(annotations[active_index])
        else:
            self.selected.clear()

    def set_dirty(self, dirty: bool) -> None:
        self._save_btn.setEnabled(bool(dirty))

    def update_section(self, section: Section | None) -> None:
        """No section-derived controls (annotations are project-global)."""
