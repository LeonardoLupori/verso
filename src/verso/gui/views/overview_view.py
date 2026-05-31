"""Overview view — table of all sections with progress tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
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
from verso.engine.model.status import STATUS_COLOR as _STATUS_COLOR
from verso.engine.model.status import section_step_status

if TYPE_CHECKING:
    from verso.gui.state import AppState

_STEPS = ("Flip", "Slice mask", "L/R mask", "Align", "Warp")

_STATUS_SYMBOL = {
    AlignmentStatus.NOT_STARTED: "",
    AlignmentStatus.IN_PROGRESS: "",
    AlignmentStatus.COMPLETE: "●",
}

_COL_SERIAL = 0
_COL_FILE = 1
_COL_DIMS = 2
_COL_AP = 3
_COL_STEPS_START = 4  # Flip, Slice, LR, Align, Warp


class _DimensionLoader(QObject):
    """Background worker: reads image dimensions for overview table rows.

    Emits ``dimension_ready(row, dims_text)`` for each section as it completes.
    """

    dimension_ready = pyqtSignal(int, str)  # (row_index, "W x H")
    finished = pyqtSignal()

    def __init__(self, sections: list) -> None:
        super().__init__()
        self._sections = list(sections)  # snapshot — avoid races with caller
        self._abort = False

    def stop(self) -> None:
        """Request cancellation. Safe to call from any thread."""
        self._abort = True

    def run(self) -> None:
        from verso.engine.io.image_io import registration_dimensions

        for row, section in enumerate(self._sections):
            if self._abort:
                break
            try:
                w, h = registration_dimensions(section)
                self.dimension_ready.emit(row, f"{w} x {h}")
            except Exception:
                pass
        self.finished.emit()


class OverviewView(QWidget):
    """Table-based overview of all sections and their pipeline status."""

    section_activated = pyqtSignal(int)   # double-click → open in Prep
    section_selected = pyqtSignal(int)    # single click → update properties

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._project: Project | None = None
        self._dim_loader: _DimensionLoader | None = None
        self._dim_thread: QThread | None = None
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
        axis_name = (
            self._project.interpolation_axis if self._project is not None else "AP"
        )
        headers = ["#", "File", "Dimensions", f"{axis_name} (mm)"] + list(_STEPS)
        t.setHorizontalHeaderLabels(headers)

        t.horizontalHeader().setSectionResizeMode(
            _COL_FILE, QHeaderView.ResizeMode.Stretch
        )
        for col in [_COL_SERIAL, _COL_DIMS, _COL_AP] + list(range(_COL_STEPS_START, n_cols)):
            t.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )

        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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

    @staticmethod
    def _make_cell(
        text: str, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        return item

    def _stop_dim_loader(self) -> None:
        if self._dim_loader is not None:
            self._dim_loader.stop()
        if self._dim_thread is not None:
            try:
                if self._dim_thread.isRunning():
                    self._dim_thread.quit()
                    self._dim_thread.wait()
            except RuntimeError:
                pass  # C++ object already deleted — thread has already finished
        self._dim_loader = None
        self._dim_thread = None

    def shutdown(self) -> None:
        """Stop the background loader. Must be called before the widget is destroyed."""
        self._stop_dim_loader()

    def load_project(self, project: Project) -> None:
        self._project = project
        axis_name = project.interpolation_axis if project is not None else "AP"
        self._table.horizontalHeaderItem(_COL_AP).setText(f"{axis_name} (mm)")
        self._populate()

    def _populate(self) -> None:
        p = self._project
        if p is None or not p.sections:
            self._stop_dim_loader()
            self._empty.setVisible(True)
            self._table.setVisible(False)
            self._summary.setText("  —")
            return

        self._stop_dim_loader()

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

        # Read image dimensions in the background; update each cell as it arrives.
        loader = _DimensionLoader(p.sections)
        thread = QThread()  # No parent — we control lifetime explicitly via shutdown()
        loader.moveToThread(thread)
        thread.started.connect(loader.run)
        loader.dimension_ready.connect(self._on_dimension_ready)
        loader.finished.connect(thread.quit)
        self._dim_loader = loader
        self._dim_thread = thread
        thread.start()

    def _fill_row(self, row: int, section: Section) -> None:
        t = self._table

        import os
        t.setItem(row, _COL_SERIAL, self._make_cell(str(section.serial_number)))
        file_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        t.setItem(
            row,
            _COL_FILE,
            self._make_cell(os.path.basename(section.original_path), file_align),
        )
        # Dimensions are loaded asynchronously by _DimensionLoader.
        t.setItem(row, _COL_DIMS, self._make_cell("—"))
        pos = section.alignment.position_mm
        t.setItem(row, _COL_AP, self._make_cell(f"{pos:.2f}" if pos is not None else "—"))

        # Status columns.  Prep splits into flip / slice / L-R sub-columns; the
        # registry tracks a single "prep" dirty flag, but a flushed draft tells
        # us which sub-step actually changed so an unsaved edit to an already-
        # saved mask still shows yellow (matching the filmstrip dot).  Align /
        # Warp map 1:1 to their step status.
        done = AlignmentStatus.COMPLETE
        not_started = AlignmentStatus.NOT_STARTED
        in_progress = AlignmentStatus.IN_PROGRESS
        prep_dirty = self._state.is_dirty(section.id, "prep")

        draft = self._state.get_prep_draft(section.id)
        mask_dirty = draft is not None and draft.mask_dirty
        lr_dirty = draft is not None and draft.lr_dirty
        flip_dirty = draft is not None and (
            draft.base_flip_h != section.preprocessing.flip_horizontal
            or draft.base_flip_v != section.preprocessing.flip_vertical
        )

        def prep_status(is_done: bool, sub_dirty: bool) -> AlignmentStatus:
            if sub_dirty:
                return in_progress
            if is_done:
                return done
            # Not done, no per-sub-step signal (e.g. the section is the one open
            # in Prep, whose draft is checked out) — fall back to the step flag.
            return in_progress if prep_dirty else not_started

        statuses = [
            prep_status(bool(section.preprocessing.flip_horizontal), flip_dirty),
            prep_status(bool(section.preprocessing.slice_mask_path), mask_dirty),
            prep_status(bool(section.preprocessing.lr_mask_path), lr_dirty),
            section_step_status(
                section, "align", dirty=self._state.is_dirty(section.id, "align")
            ),
            section_step_status(
                section, "warp", dirty=self._state.is_dirty(section.id, "warp")
            ),
        ]
        for i, status in enumerate(statuses):
            col = _COL_STEPS_START + i
            if _STEPS[i] == "Warp":
                cp_count = len(section.warp.control_points)
                sym = str(cp_count) if cp_count else ""
            else:
                sym = _STATUS_SYMBOL[status]
            item = self._make_cell(sym)
            item.setForeground(QColor(_STATUS_COLOR[status]))
            t.setItem(row, col, item)

    def _on_dimension_ready(self, row: int, dims_text: str) -> None:
        """Slot: called on the main thread when the loader reads one section's dims."""
        if not self._table.isVisible():
            return
        # Guard against stale signals arriving after a new _populate() cleared the table.
        if 0 <= row < self._table.rowCount():
            self._table.setItem(row, _COL_DIMS, self._make_cell(dims_text))

    def refresh_row(self, section_index: int) -> None:
        if self._project is None:
            return
        section = self._project.sections[section_index]
        self._fill_row(section_index, section)
        # Single-row refresh: read dims synchronously (~5–20 ms, acceptable).
        try:
            from verso.engine.io.image_io import registration_dimensions
            w, h = registration_dimensions(section)
            self._table.setItem(
                section_index, _COL_DIMS, self._make_cell(f"{w} x {h}")
            )
        except Exception:
            pass

    def refresh(self) -> None:
        self._populate()

    def selected_rows(self) -> list[int]:
        """Return the sorted indices of all currently selected sections."""
        if not self._table.isVisible():
            return []
        rows = {idx.row() for idx in self._table.selectionModel().selectedRows()}
        return sorted(rows)

    def _on_double_click(self, row: int, _col: int) -> None:
        self.section_activated.emit(row)

    def _on_selection_changed(self, row: int, *_) -> None:
        if row >= 0:
            self.section_selected.emit(row)
