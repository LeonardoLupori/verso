"""Overview view — table of all sections with progress tracking."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QShowEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Project, Section
from verso.engine.model.status import STATUS_COLOR as _STATUS_COLOR
from verso.engine.model.status import section_step_status
from verso.gui.utils import require

if TYPE_CHECKING:
    from verso.gui.state import AppState

_STEPS = ("Flip", "Slice mask", "Align", "Warp")

# Every step renders the same dot; the colour carries the meaning, matching the
# filmstrip convention (gray = not started, yellow = unsaved edits, green =
# saved).  See verso.engine.model.status.STATUS_COLOR.
_DOT = "●"

_COL_SERIAL = 0
_COL_FILE = 1
_COL_DIMS = 2
_COL_AP = 3
_COL_STEPS_START = 4  # Flip, Slice, Align, Warp

# Modern table styling, tuned to the app's dark theme (#2a2a2a chrome,
# #1e5a8a accent — see main_window / view_chrome).  Flat, gridline-free rows
# with a quiet uppercase header and a soft accent rule beneath it.
_TABLE_STYLE = """
QTableWidget {
    background: #1e1e1e;
    alternate-background-color: #242424;
    gridline-color: transparent;
    border: 1px solid #383838;
    border-radius: 8px;
    color: #d6d6d6;
    font-size: 12px;
    outline: none;
}
QTableWidget::item {
    padding: 6px 10px;
    border: none;
}
QTableWidget::item:selected {
    background: #1e5a8a;
    color: #ffffff;
}
QHeaderView::section {
    background: #262626;
    color: #9aa0a6;
    padding: 9px 10px;
    border: none;
    border-bottom: 2px solid #1e5a8a;
    font-size: 11px;
    font-weight: 600;
}
QTableCornerButton::section {
    background: #262626;
    border: none;
}
"""

_SUMMARY_STYLE = (
    "color: #888; padding: 10px 6px 2px 6px; font-size: 11px; border-top: 1px solid #333;"
)


class _SliceIndexDelegate(QStyledItemDelegate):
    """Renders the slice-index cells as a subtle, editable input field.

    Slice index is the only editable column, so we outline each cell with a
    faint rounded "chip" — quiet at rest, filling and brightening on hover — to
    hint that the value can be changed without drawing the eye.  Selected rows
    defer to the default highlight so the chip never fights the selection.
    """

    # At rest the chip is just a faint outline so it reads as editable without
    # drawing the eye; hovering fills and brightens it to invite the edit.
    _BORDER = QColor("#3f3f3f")
    _BORDER_HOVER = QColor("#3a6d99")
    _FILL_HOVER = QColor("#27414f")
    _TEXT = QColor("#d6d6d6")
    _TEXT_HOVER = QColor("#9fd0f2")

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        # Let the base style paint the cell background (alternating bands and,
        # when selected, the highlight).  On a selected row we stop there so the
        # chip never competes with the selection colour.
        super().paint(painter, option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            return

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The base style already drew the value; cover it so we can recolour on
        # hover, then outline the chip over the (matching) row background.
        chip = QRectF(option.rect).adjusted(7, 7, -7, -7)
        painter.setPen(QPen(self._BORDER_HOVER if hovered else self._BORDER, 1))
        painter.setBrush(self._FILL_HOVER if hovered else Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(chip, 5, 5)

        painter.setPen(self._TEXT_HOVER if hovered else self._TEXT)
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, index.data() or "")
        painter.restore()


class OverviewView(QWidget):
    """Table-based overview of all sections and their pipeline status.

    The table is split into *static* columns (slice index, file, dimensions)
    that change only when the section list changes, and *dynamic* columns (AP
    position, flip label, status dots) that track per-step edit/save state.
    ``refresh()`` updates only the dynamic cells unless the section list itself
    changed (add / remove / reorder), in which case it rebuilds the whole table.
    Work is skipped entirely while the view is hidden and flushed on show.
    """

    section_activated = pyqtSignal(int)  # double-click → open in Prep
    section_selected = pyqtSignal(int)  # single click → update properties
    sections_reordered = pyqtSignal()  # slice_index edited → re-sort + persist
    remove_requested = pyqtSignal(list)  # context menu → remove section ids

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._project: Project | None = None
        self._suppress_edits = False  # ignore cellChanged during programmatic fills
        self._pending_refresh = False  # a refresh was requested while hidden
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 10)
        layout.setSpacing(0)

        # Empty state (shown when no project is loaded)
        self._empty = QWidget()
        empty_layout = QVBoxLayout(self._empty)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("No project loaded")
        lbl.setStyleSheet("font-size: 18px; color: #888;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl)
        sub = QLabel("Use  File → Open QuickNII…  or  File → New Project  to get started.")
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
        self._summary.setStyleSheet(_SUMMARY_STYLE)
        layout.addWidget(self._summary)

    def _setup_table(self) -> None:
        t = self._table
        n_cols = _COL_STEPS_START + len(_STEPS)
        t.setColumnCount(n_cols)
        axis_name = self._project.interpolation_axis if self._project is not None else "AP"
        headers = ["Slice index", "File", "Dimensions", f"{axis_name} (mm)", *list(_STEPS)]
        t.setHorizontalHeaderLabels(headers)

        hheader = require(t.horizontalHeader())
        vheader = require(t.verticalHeader())

        hheader.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        # Content columns are fitted once per rebuild (resizeColumnsToContents)
        # and then left Fixed: the user can't drag the dividers, and — unlike
        # live ResizeToContents, which re-measures every row in a column on each
        # cell write (O(rows^2) status refreshes, seconds at a few hundred
        # sections) — cell writes no longer trigger any re-measure.
        for col in [_COL_SERIAL, _COL_DIMS, _COL_AP, *list(range(_COL_STEPS_START, n_cols))]:
            hheader.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)

        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Only the "#" (slice index) cell is editable — see _fill_static_cells.
        # Editing any other cell is blocked by stripping ItemIsEditable in
        # _make_cell.
        t.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        require(t.horizontalHeaderItem(_COL_SERIAL)).setToolTip(
            "Slice index — physical position along the interpolation axis. "
            "Double-click to edit; rows re-sort by index."
        )
        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        vheader.setDefaultSectionSize(38)
        vheader.setVisible(False)
        hheader.setHighlightSections(False)
        t.setStyleSheet(_TABLE_STYLE)

        # Slice index is the only editable column — render it as an input chip
        # (with a hover affordance) so users see it can be changed.
        t.setMouseTracking(True)
        t.setItemDelegateForColumn(_COL_SERIAL, _SliceIndexDelegate(t))

        t.cellDoubleClicked.connect(self._on_double_click)
        t.currentCellChanged.connect(self._on_selection_changed)
        t.cellChanged.connect(self._on_cell_changed)

        t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        t.customContextMenuRequested.connect(self._on_context_menu)

    @staticmethod
    def _make_cell(
        text: str, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        # Non-editable by default; _fill_static_cells re-enables only the
        # slice-index cell.
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _dims_text(section: Section) -> str:
        """Return "W x H" for a section's registration image.

        Read from the section's cached ``resolution_thumbnail_wh`` (populated by
        ``backfill_metadata`` at import and persisted in the project file).
        """
        w, h = section.resolution_thumbnail_wh
        return f"{w} x {h}"

    # -- public API ---------------------------------------------------------

    def load_project(self, project: Project) -> None:
        self._project = project
        axis_name = project.interpolation_axis if project is not None else "AP"
        require(self._table.horizontalHeaderItem(_COL_AP)).setText(f"{axis_name} (mm)")
        # A new project never matches the current rows, so refresh() rebuilds.
        self.refresh()

    def refresh(self) -> None:
        """Update the table — incrementally when only statuses changed.

        Deferred while the view is hidden (many callers fire from Prep / Align /
        Warp); the pending refresh is flushed in ``showEvent``.
        """
        if not self.isVisible():
            self._pending_refresh = True
            return
        self._pending_refresh = False

        p = self._project
        if p is None or not p.sections:
            self._show_empty()
            return
        if self._structure_matches():
            self._refresh_status()
        else:
            self._rebuild()
        self._sync_selection()

    def refresh_row(self, section_index: int) -> None:
        """Update a single row's status cells (e.g. after a flip or align edit)."""
        if self._project is None:
            return
        if not self.isVisible():
            self._pending_refresh = True
            return
        self._suppress_edits = True
        self._fill_status_cells(section_index, self._project.sections[section_index])
        self._suppress_edits = False

    def selected_rows(self) -> list[int]:
        """Return the sorted indices of all currently selected sections."""
        if not self._table.isVisible():
            return []
        rows = {idx.row() for idx in require(self._table.selectionModel()).selectedRows()}
        return sorted(rows)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._pending_refresh:
            self.refresh()

    # -- table building -----------------------------------------------------

    def _show_empty(self) -> None:
        self._empty.setVisible(True)
        self._table.setVisible(False)
        self._summary.setText("  —")

    def _structure_matches(self) -> bool:
        """True when the table's rows still mirror the project's sections.

        A mismatch (different count, or a row whose stored id no longer lines up)
        means a section was added, removed, or reordered, so the table must be
        rebuilt rather than updated in place.
        """
        sections = self._project.sections
        if self._table.rowCount() != len(sections):
            return False
        return all(
            self._section_id_for_row(row) == section.id for row, section in enumerate(sections)
        )

    def _rebuild(self) -> None:
        """Full structural rebuild — static + dynamic cells for every row."""
        p = self._project
        self._empty.setVisible(False)
        self._table.setVisible(True)

        t = self._table
        self._suppress_edits = True
        t.setRowCount(len(p.sections))
        complete, in_progress = 0, 0
        for row, section in enumerate(p.sections):
            self._fill_static_cells(row, section)
            status = self._fill_status_cells(row, section)
            if status == AlignmentStatus.COMPLETE:
                complete += 1
            elif status == AlignmentStatus.IN_PROGRESS:
                in_progress += 1
        self._suppress_edits = False

        self._set_summary(len(p.sections), complete, in_progress)

        # One-shot fit for the Interactive content columns (the Stretch File
        # column is unaffected).  Doing this only on a structural rebuild keeps
        # the frequent status-only refreshes cheap.
        t.resizeColumnsToContents()

    def _sync_selection(self) -> None:
        """Highlight and scroll to the row for the current section.

        Runs after every refresh (not just a structural rebuild) so the
        highlighted row stays in sync when the current section changes
        elsewhere (e.g. selecting a slice in Prep) without any add / remove /
        reorder that would otherwise trigger a rebuild.
        """
        t = self._table
        idx = self._state.section_index
        if 0 <= idx < t.rowCount() and t.currentRow() != idx:
            t.blockSignals(True)
            t.setCurrentCell(idx, 0)
            t.blockSignals(False)
            t.scrollTo(t.model().index(idx, 0))

    def _refresh_status(self) -> None:
        """Update only the dynamic cells (and the summary) — no static churn."""
        p = self._project
        self._suppress_edits = True
        complete, in_progress = 0, 0
        for row, section in enumerate(p.sections):
            status = self._fill_status_cells(row, section)
            if status == AlignmentStatus.COMPLETE:
                complete += 1
            elif status == AlignmentStatus.IN_PROGRESS:
                in_progress += 1
        self._suppress_edits = False
        self._set_summary(len(p.sections), complete, in_progress)

    def _set_summary(self, total: int, complete: int, in_progress: int) -> None:
        self._summary.setText(
            f"  {total} sections  ·  {complete} complete  ·  {in_progress} in progress"
        )

    def _fill_static_cells(self, row: int, section: Section) -> None:
        """Fill the columns that change only when the section list changes."""
        t = self._table

        serial_cell = self._make_cell(str(section.slice_index))
        serial_cell.setFlags(serial_cell.flags() | Qt.ItemFlag.ItemIsEditable)
        serial_cell.setToolTip("Double-click to edit — rows re-sort by slice index")
        # Carry the section id so an edit maps to the right section regardless of
        # how rows are later re-sorted.
        serial_cell.setData(Qt.ItemDataRole.UserRole, section.id)
        t.setItem(row, _COL_SERIAL, serial_cell)

        file_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        t.setItem(
            row, _COL_FILE, self._make_cell(os.path.basename(section.original_path), file_align)
        )
        t.setItem(row, _COL_DIMS, self._make_cell(self._dims_text(section)))

    def _fill_status_cells(self, row: int, section: Section) -> AlignmentStatus:
        """Fill the dynamic columns; return the warp status for the summary tally.

        Prep splits into flip / slice sub-columns: the registry tracks a single
        "prep" dirty flag, but a flushed draft tells us which sub-step actually
        changed so an unsaved edit to an already-saved mask still shows yellow
        (matching the filmstrip dot).  Align / Warp map 1:1 to their step status.
        """
        t = self._table

        pos = section.alignment.position_mm
        t.setItem(row, _COL_AP, self._make_cell(f"{pos:.2f}" if pos is not None else "—"))

        # Prep is one registry step with two independently-shown sub-states.
        # Both derive from the single (section, "prep") draft entry: an unsaved
        # mask edit is parked in the "working" payload, and the last-saved flips
        # live in the baseline — so colour each sub-step on its own.
        mask_dirty = self._state.has_working(section.id, "prep")
        baseline = self._state.get_baseline(section.id, "prep")
        if baseline is not None:
            flip_dirty = (
                baseline.flip_horizontal != section.preprocessing.flip_horizontal
                or baseline.flip_vertical != section.preprocessing.flip_vertical
            )
        else:
            flip_dirty = False

        def prep_status(is_done: bool, sub_dirty: bool) -> AlignmentStatus:
            if sub_dirty:
                return AlignmentStatus.IN_PROGRESS
            return AlignmentStatus.COMPLETE if is_done else AlignmentStatus.NOT_STARTED

        # Flip is a state, not a task: both flipped and un-flipped are valid end
        # states, so it gets a plain H / V / H+V label (regular colour) instead
        # of a traffic-light dot — tinted the same yellow only when unsaved.
        fh = bool(section.preprocessing.flip_horizontal)
        fv = bool(section.preprocessing.flip_vertical)
        flip_text = "H+V" if (fh and fv) else "H" if fh else "V" if fv else "—"
        flip_item = self._make_cell(flip_text)
        if flip_dirty:
            flip_item.setForeground(QColor(_STATUS_COLOR[AlignmentStatus.IN_PROGRESS]))
        t.setItem(row, _COL_STEPS_START, flip_item)

        # The remaining steps are genuine tasks — render them as status dots.
        warp_status = section_step_status(
            section, "warp", dirty=self._state.is_dirty(section.id, "warp")
        )
        dot_statuses = [
            prep_status(bool(section.preprocessing.slice_mask_path), mask_dirty),
            section_step_status(section, "align", dirty=self._state.is_dirty(section.id, "align")),
            warp_status,
        ]
        for i, status in enumerate(dot_statuses):
            item = self._make_cell(_DOT)
            item.setForeground(QColor(_STATUS_COLOR[status]))
            t.setItem(row, _COL_STEPS_START + 1 + i, item)

        return warp_status

    # -- events / interaction ----------------------------------------------

    def _section_id_for_row(self, row: int) -> str | None:
        """Return the section id stored on a row, or None."""
        item = self._table.item(row, _COL_SERIAL)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _on_context_menu(self, pos: QPoint) -> None:
        """Offer 'Remove from project' for the right-clicked / selected rows."""
        if self._project is None or not self._project.sections:
            return
        clicked_row = self._table.rowAt(pos.y())
        if clicked_row < 0:
            return
        # Right-clicking outside the current selection retargets to that row.
        if clicked_row not in self.selected_rows():
            self._table.selectRow(clicked_row)

        ids = [
            sid
            for row in self.selected_rows()
            if (sid := self._section_id_for_row(row)) is not None
        ]
        if not ids:
            return

        menu = QMenu(self._table)
        n = len(ids)
        label = "Remove from project" if n == 1 else f"Remove {n} from project"
        act_remove = menu.addAction(label)
        chosen = menu.exec(require(self._table.viewport()).mapToGlobal(pos))
        if chosen is act_remove:
            self.remove_requested.emit(ids)

    def _on_double_click(self, row: int, col: int) -> None:
        # Double-click on the "#" column opens the inline editor instead of the
        # section; every other column activates (opens in Prep).
        if col == _COL_SERIAL:
            return
        self.section_activated.emit(row)

    def _on_cell_changed(self, row: int, col: int) -> None:
        """Commit an edited slice-index cell: update, re-sort, persist."""
        if self._suppress_edits or col != _COL_SERIAL or self._project is None:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        section_id = item.data(Qt.ItemDataRole.UserRole)
        section = next((s for s in self._project.sections if s.id == section_id), None)
        if section is None:
            return

        text = item.text().strip()
        if not text.isdigit():
            # Reject non-integer input — restore the previous value.
            self._suppress_edits = True
            item.setText(str(section.slice_index))
            self._suppress_edits = False
            return

        new_index = int(text)
        if new_index == section.slice_index:
            return

        # Keep the same physical section selected across the re-sort.
        current = self._state.current_section
        keep_id = current.id if current is not None else None

        section.slice_index = new_index
        self._project.sort_sections()
        self.refresh()

        if keep_id is not None:
            new_pos = next(
                (i for i, s in enumerate(self._project.sections) if s.id == keep_id),
                None,
            )
            if new_pos is not None:
                self._state.set_section(new_pos)

        self.sections_reordered.emit()

    def _on_selection_changed(self, row: int, *_) -> None:
        if row >= 0:
            self.section_selected.emit(row)
