"""Annotation manager section — create, select, toggle, and delete annotations.

A compact action row (New ▾ / Delete) sits above a single-selection table of
every annotation in the project, one row per annotation: its type icon, a
per-row visibility toggle, and its name. Selecting a row drives the active
annotation shown by
:class:`~verso.gui.widgets.properties.sections.edit_annotations.EditAnnotationsBox`.
A pure view: it emits intent signals and renders whatever :meth:`set_annotations`
is handed.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.annotation import Annotation, AreaAnnotation
from verso.gui.widgets.properties._common import colored_icon, eye_icon, make_eye_btn

_COL_TYPE, _COL_NAME, _COL_VISIBLE = range(3)

# Matches the Overview table's dark-theme styling (see overview_view._TABLE_STYLE).
_TABLE_STYLE = """
QTableWidget {
    background: #1e1e1e;
    alternate-background-color: #242424;
    gridline-color: transparent;
    border-radius: 4px;
    color: #d6d6d6;
    font-size: 12px;
    outline: none;
}

QTableWidget::item:selected {
    background: #1e5a8a;
    color: #ffffff;
}
"""


class ManageAnnotationsBox(QGroupBox):
    new_point_requested = pyqtSignal()
    new_area_requested = pyqtSignal()
    import_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    active_changed = pyqtSignal(int)
    visibility_changed = pyqtSignal(int, bool)  # (row, visible)

    def __init__(self) -> None:
        super().__init__("Annotations")
        v = QVBoxLayout(self)
        # Zero out the left/right margins so the table runs edge-to-edge with
        # the group box border instead of leaving a side gutter; the action
        # row keeps the original inset (applied to its own layout below).
        margins = v.contentsMargins()
        v.setContentsMargins(0, margins.top(), 0, 0)

        actions_row = self._build_actions_row()
        actions_row.setContentsMargins(margins.left(), 0, margins.right(), 0)
        v.addLayout(actions_row)

        v.addSpacing(8)

        self._table = self._build_table()
        v.addWidget(self._table)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_actions_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)

        self._new_btn = QPushButton("New")
        self._new_btn.setIcon(colored_icon("circle-plus.svg", "#ffffff"))
        self._new_btn.setIconSize(QSize(14, 14))
        self._new_btn.setToolTip("Create a new annotation")
        self._new_btn.setMenu(self._build_new_menu())

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setIcon(colored_icon("trash-2.svg", "#ffffff"))
        self._delete_btn.setIconSize(QSize(14, 14))
        self._delete_btn.setToolTip("Delete the selected annotation")
        self._delete_btn.clicked.connect(self.delete_requested)

        row.addWidget(self._new_btn)
        row.addStretch()
        row.addWidget(self._delete_btn)
        return row

    def _build_new_menu(self) -> QMenu:
        menu = QMenu(self._new_btn)

        act_area = QAction(colored_icon("area.svg", "#ffffff"), "Area", menu)
        act_area.triggered.connect(self.new_area_requested)
        menu.addAction(act_area)

        points_menu = QMenu("Point series", menu)
        points_menu.setIcon(colored_icon("scatter.svg", "#ffffff"))
        act_new_empty = QAction("New empty", points_menu)
        act_new_empty.triggered.connect(self.new_point_requested)
        act_from_csv = QAction("From CSV…", points_menu)
        act_from_csv.triggered.connect(self.import_requested)
        points_menu.addAction(act_new_empty)
        points_menu.addAction(act_from_csv)
        menu.addMenu(points_menu)

        return menu

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, 3)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(_TABLE_STYLE)
        header = table.horizontalHeader()
        header.setSectionResizeMode(_COL_TYPE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_VISIBLE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(26)
        table.itemSelectionChanged.connect(self._on_selection_changed)
        return table

    # ------------------------------------------------------------------
    # Population (driven by AnnotatePage / AnnotationController)
    # ------------------------------------------------------------------

    def set_annotations(self, annotations: list[Annotation], active_index: int) -> None:
        """Rebuild the table and select *active_index*'s row (no signal)."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(annotations))
        for row, ann in enumerate(annotations):
            is_area = isinstance(ann, AreaAnnotation)
            icon = "area.svg" if is_area else "scatter.svg"

            type_item = QTableWidgetItem()
            type_item.setIcon(colored_icon(icon, "#aaaaaa"))
            type_item.setToolTip("Area" if is_area else "Point series")
            self._table.setItem(row, _COL_TYPE, type_item)

            self._table.setCellWidget(row, _COL_VISIBLE, self._make_visibility_cell(row, ann))

            name_item = QTableWidgetItem(ann.title)
            self._table.setItem(row, _COL_NAME, name_item)

        if 0 <= active_index < len(annotations):
            self._table.setCurrentCell(active_index, _COL_NAME)
        else:
            self._table.clearSelection()
        self._table.blockSignals(False)

        self._delete_btn.setEnabled(0 <= active_index < len(annotations))

    def _make_visibility_cell(self, row: int, ann: Annotation) -> QWidget:
        """An eye toggle centred in its cell, wired to emit ``(row, visible)``."""
        btn = make_eye_btn()
        btn.setChecked(ann.visible)
        btn.setIcon(eye_icon(ann.visible))
        btn.setToolTip("Show / hide this annotation")
        btn.toggled.connect(lambda checked, r=row: self.visibility_changed.emit(r, checked))

        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(btn)
        return cell

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows(_COL_NAME)
        if rows:
            self.active_changed.emit(rows[0].row())
