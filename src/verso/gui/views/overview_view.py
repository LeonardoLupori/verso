"""Overview view — table of all sections with progress tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QPoint, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
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

# Every step shows a dot; the colour carries the meaning, matching the
# filmstrip convention (gray = not started, yellow = unsaved edits, green =
# saved).  See verso.engine.model.status.STATUS_COLOR.
_STATUS_SYMBOL = {
    AlignmentStatus.NOT_STARTED: "●",
    AlignmentStatus.IN_PROGRESS: "●",
    AlignmentStatus.COMPLETE: "●",
}

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

    section_activated = pyqtSignal(int)  # double-click → open in Prep
    section_selected = pyqtSignal(int)  # single click → update properties
    sections_reordered = pyqtSignal()  # slice_index edited → re-sort + persist
    remove_requested = pyqtSignal(list)  # context menu → remove section ids

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._project: Project | None = None
        self._dim_loader: _DimensionLoader | None = None
        self._dim_thread: QThread | None = None
        self._suppress_edits = False  # ignore cellChanged during programmatic fills
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
        headers = ["Slice index", "File", "Dimensions", f"{axis_name} (mm)"] + list(_STEPS)
        t.setHorizontalHeaderLabels(headers)

        hheader = require(t.horizontalHeader())
        vheader = require(t.verticalHeader())

        hheader.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        for col in [_COL_SERIAL, _COL_DIMS, _COL_AP] + list(range(_COL_STEPS_START, n_cols)):
            hheader.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Only the "#" (slice index) cell is editable — see _fill_row. Editing
        # any other cell is blocked by stripping ItemIsEditable in _make_cell.
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
        # Non-editable by default; _fill_row re-enables only the slice-index cell.
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
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
        require(self._table.horizontalHeaderItem(_COL_AP)).setText(f"{axis_name} (mm)")
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
        self._suppress_edits = True
        t.setRowCount(len(p.sections))

        complete = 0
        in_progress = 0

        for row, section in enumerate(p.sections):
            self._fill_row(row, section)
            warp_dirty = self._state.is_dirty(section.id, "warp")
            ws = section_step_status(section, "warp", dirty=warp_dirty)
            if ws == AlignmentStatus.COMPLETE:
                complete += 1
            elif ws == AlignmentStatus.IN_PROGRESS:
                in_progress += 1
        self._suppress_edits = False

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

        serial_cell = self._make_cell(str(section.slice_index))
        serial_cell.setFlags(serial_cell.flags() | Qt.ItemFlag.ItemIsEditable)
        serial_cell.setToolTip("Double-click to edit — rows re-sort by slice index")
        # Carry the section id so an edit maps to the right section regardless of
        # how rows are later re-sorted.
        serial_cell.setData(Qt.ItemDataRole.UserRole, section.id)
        t.setItem(row, _COL_SERIAL, serial_cell)
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

        draft = self._state.get_prep_draft(section.id)
        if draft is not None:
            # The flushed draft carries per-sub-step dirty flags, so colour each
            # sub-step independently — editing only the mask yellows just the
            # Slice-mask dot, not Flip.
            mask_dirty = draft.mask_dirty
            flip_dirty = (
                draft.base_flip_h != section.preprocessing.flip_horizontal
                or draft.base_flip_v != section.preprocessing.flip_vertical
            )

            def prep_status(is_done: bool, sub_dirty: bool) -> AlignmentStatus:
                if sub_dirty:
                    return in_progress
                return done if is_done else not_started
        else:
            # No draft — the section is the one currently open in Prep, whose
            # live edits aren't flushed yet, so we only know the aggregate prep
            # flag.  Fall back to it for any not-yet-saved sub-step.
            prep_dirty = self._state.is_dirty(section.id, "prep")
            mask_dirty = flip_dirty = False

            def prep_status(is_done: bool, _sub_dirty: bool) -> AlignmentStatus:
                if is_done:
                    return done
                return in_progress if prep_dirty else not_started

        # Flip is a state, not a task: both flipped and un-flipped are valid end
        # states, so it gets a plain H / V / H+V label (regular colour) instead
        # of a traffic-light dot — tinted the same yellow only when unsaved.
        fh = bool(section.preprocessing.flip_horizontal)
        fv = bool(section.preprocessing.flip_vertical)
        flip_text = "H+V" if (fh and fv) else "H" if fh else "V" if fv else "—"
        flip_item = self._make_cell(flip_text)
        if flip_dirty:
            flip_item.setForeground(QColor(_STATUS_COLOR[in_progress]))
        t.setItem(row, _COL_STEPS_START, flip_item)

        # The remaining steps are genuine tasks — render them as status dots.
        dot_statuses = [
            prep_status(bool(section.preprocessing.slice_mask_path), mask_dirty),
            section_step_status(section, "align", dirty=self._state.is_dirty(section.id, "align")),
            section_step_status(section, "warp", dirty=self._state.is_dirty(section.id, "warp")),
        ]
        for i, status in enumerate(dot_statuses):
            col = _COL_STEPS_START + 1 + i
            item = self._make_cell(_STATUS_SYMBOL[status])
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
        self._suppress_edits = True
        self._fill_row(section_index, section)
        self._suppress_edits = False
        # Single-row refresh: read dims synchronously (~5–20 ms, acceptable).
        try:
            from verso.engine.io.image_io import registration_dimensions

            w, h = registration_dimensions(section)
            self._table.setItem(section_index, _COL_DIMS, self._make_cell(f"{w} x {h}"))
        except Exception:
            pass

    def refresh(self) -> None:
        self._populate()

    def selected_rows(self) -> list[int]:
        """Return the sorted indices of all currently selected sections."""
        if not self._table.isVisible():
            return []
        rows = {idx.row() for idx in require(self._table.selectionModel()).selectedRows()}
        return sorted(rows)

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
        self._populate()

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
