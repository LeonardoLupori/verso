"""Overview view — table of all sections with progress tracking."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Project, Section

_STEPS = ("Flip", "Slice mask", "L/R mask", "Align", "Warp")

_STATUS_SYMBOL = {
    AlignmentStatus.NOT_STARTED: "",
    AlignmentStatus.IN_PROGRESS: "",
    AlignmentStatus.COMPLETE: "●",
}
_STATUS_COLOR = {
    AlignmentStatus.NOT_STARTED: "#888888",
    AlignmentStatus.IN_PROGRESS: "#E6A817",
    AlignmentStatus.COMPLETE: "#4CAF50",
}

_COL_SERIAL = 0
_COL_FILE = 1
_COL_AP = 2
_COL_STEPS_START = 3  # Flip, Slice, LR, Align, Warp


class OverviewView(QWidget):
    """Table-based overview of all sections and their pipeline status."""

    section_activated = pyqtSignal(int)   # double-click → open in Prep
    section_selected = pyqtSignal(int)    # single click → update properties
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Empty state (shown when no project is loaded)
        self._empty = QWidget()
        empty_layout = QVBoxLayout(self._empty)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("No project loaded")
        lbl.setStyleSheet("font-size: 18px; color: #888;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl)
        sub = QLabel('Use  File → Open QuickNII…  or  File → New Project  to get started.')
        sub.setStyleSheet("color: #666; font-size: 12px;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(sub)
        layout.addWidget(self._empty)

        # Table (hidden until a project is loaded)
        self._table = QTableWidget()
        self._table.setVisible(False)
        self._setup_table()
        layout.addWidget(self._table)

        # Summary bar
        self._summary = QLabel("  —")
        self._summary.setStyleSheet(
            "background: #2a2a2a; color: #aaa; padding: 4px 8px; font-size: 11px;"
        )
        layout.addWidget(self._summary)

    def _setup_table(self) -> None:
        t = self._table
        n_cols = _COL_STEPS_START + len(_STEPS)
        t.setColumnCount(n_cols)
        headers = ["#", "File", "AP (mm)"] + list(_STEPS)
        t.setHorizontalHeaderLabels(headers)

        t.horizontalHeader().setSectionResizeMode(
            _COL_FILE, QHeaderView.ResizeMode.Stretch
        )
        for col in [_COL_SERIAL, _COL_AP] + list(range(_COL_STEPS_START, n_cols)):
            t.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )

        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setDefaultSectionSize(36)
        t.verticalHeader().setVisible(False)
        t.setStyleSheet(
            "QTableWidget { gridline-color: #333; }"
            "QTableWidget::item:selected { background: #1e5a8a; }"
        )

        t.cellDoubleClicked.connect(self._on_double_click)
        t.currentCellChanged.connect(self._on_selection_changed)

    def load_project(self, project: Project) -> None:
        self._project = project
        self._populate()

    def _populate(self) -> None:
        p = self._project
        if p is None or not p.sections:
            self._empty.setVisible(True)
            self._table.setVisible(False)
            self._summary.setText("  —")
            return

        self._empty.setVisible(False)
        self._table.setVisible(True)

        t = self._table
        t.setRowCount(len(p.sections))

        complete = 0
        in_progress = 0

        for row, section in enumerate(p.sections):
            self._fill_row(row, section)
            ws = section.warp.status
            if ws == AlignmentStatus.COMPLETE:
                complete += 1
            elif ws == AlignmentStatus.IN_PROGRESS:
                in_progress += 1

        total = len(p.sections)
        self._summary.setText(
            f"  {total} sections  ·  {complete} complete  ·  {in_progress} in progress"
        )


    def _fill_row(self, row: int, section: Section) -> None:
        t = self._table

        def cell(text: str, align=Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setTextAlignment(align)
            return item

        import os
        t.setItem(row, _COL_SERIAL, cell(str(section.serial_number)))
        file_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        t.setItem(
            row,
            _COL_FILE,
            cell(os.path.basename(section.original_path), file_align),
        )
        ap = section.alignment.ap_position_mm
        t.setItem(row, _COL_AP, cell(f"{ap:.2f}" if ap is not None else "—"))

        # Status columns
        done = AlignmentStatus.COMPLETE
        not_started = AlignmentStatus.NOT_STARTED
        statuses = [
            done if section.preprocessing.flip_horizontal else not_started,
            done if section.preprocessing.slice_mask_path else not_started,
            done if section.preprocessing.lr_mask_path else not_started,
            section.alignment.status,
            section.warp.status,
        ]
        for i, status in enumerate(statuses):
            col = _COL_STEPS_START + i
            if _STEPS[i] == "Warp":
                cp_count = len(section.warp.control_points)
                sym = str(cp_count) if cp_count else ""
            else:
                sym = _STATUS_SYMBOL[status]
            item = cell(sym)
            item.setForeground(QColor(_STATUS_COLOR[status]))
            t.setItem(row, col, item)

    def refresh_row(self, section_index: int) -> None:
        if self._project is None:
            return
        section = self._project.sections[section_index]
        self._fill_row(section_index, section)

    def refresh(self) -> None:
        self._populate()

    def _on_double_click(self, row: int, _col: int) -> None:
        self.section_activated.emit(row)

    def _on_selection_changed(self, row: int, *_) -> None:
        if row >= 0:
            self.section_selected.emit(row)
