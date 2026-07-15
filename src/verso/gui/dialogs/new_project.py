"""New Project dialog.

Collects:
  - Project name
  - Project file path
  - Atlas selection
  - One or more section image files (TIFF / PNG / JPEG)

On accept, call result() to get the configured Project object.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from verso.engine.io.image_io import (
    SUPPORTED_IMAGE_EXTENSIONS,
    compute_working_scale,
    guess_slice_indices,
    probe_channels,
    thumbnail_filename,
)
from verso.engine.io.project_metadata import AtlasUnavailableError, populate_metadata
from verso.engine.model.alignment import Alignment, AlignmentStatus, WarpState
from verso.engine.model.project import (
    SLICING_ORIENTATION_TO_AXIS,
    AtlasRef,
    ChannelSpec,
    Project,
    Section,
)
from verso.gui.utils import require

# Default per-channel pseudo-color palettes used when seeding Project.channels
# at project creation time.
_FLUORESCENCE_PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),  # Ch 0 — red
    (0, 255, 0),  # Ch 1 — green
    (0, 100, 255),  # Ch 2 — blue / DAPI
    (255, 0, 200),  # Ch 3 — far-red / magenta
    (255, 255, 255),  # Ch 4 — white
    (255, 255, 0),  # Ch 5 — yellow
    (0, 255, 255),  # Ch 6 — cyan
]

_RGB_IDENTITY_PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]


def _default_channel_specs(channel_names: list[str], source_ext: str) -> list[ChannelSpec]:
    """Pick the per-channel color palette based on the source file extension.

    Single-channel inputs default to white. JPG/PNG inputs use the identity
    RGB palette so a regular photo renders correctly. TIFFs (assumed
    scientific) use the fluorescence palette.
    """
    n = len(channel_names)
    if n == 0:
        return []
    if n == 1:
        return [ChannelSpec(name=channel_names[0], color=(255, 255, 255))]

    ext = source_ext.lower().lstrip(".")
    palette = _RGB_IDENTITY_PALETTE if ext in ("jpg", "jpeg", "png") else _FLUORESCENCE_PALETTE

    specs: list[ChannelSpec] = []
    for i, name in enumerate(channel_names):
        color = palette[i] if i < len(palette) else (255, 255, 255)
        specs.append(ChannelSpec(name=name, color=color))
    return specs


_KNOWN_ATLASES = [
    "allen_mouse_25um",
    "allen_mouse_10um",
    "allen_mouse_50um",
    "allen_rat_25um",
    "kim_mouse_25um",
]

_IMAGE_FILTER = (
    "Images (" + " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS) + ");;All files (*)"
)

# Columns of the section-image preview table.
_FILE_COL = 0
_IDX_COL = 1

# Table styling mirrored from OverviewView so the two tables read as a family:
# flat, gridline-free rows, a quiet uppercase header with a soft accent rule.
# Cell padding is trimmed (vs. the overview's 6px 10px) because this dialog
# table has far less horizontal room to spend.
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
    padding: 4px 8px;
    border: none;
}
QTableWidget::item:selected {
    background: #1e5a8a;
    color: #ffffff;
}
QHeaderView::section {
    background: #262626;
    color: #9aa0a6;
    padding: 7px 8px;
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


class _SliceIndexDelegate(QStyledItemDelegate):
    """Renders the slice-index cells as a subtle, editable input chip.

    Matches the affordance used in OverviewView: a faint rounded outline at
    rest that fills and brightens on hover, hinting the value can be edited
    without drawing the eye. Selected rows defer to the default highlight.
    The chip inset is a touch tighter than the overview's to suit this
    dialog's shorter rows.
    """

    _BORDER = QColor("#3f3f3f")
    _BORDER_HOVER = QColor("#3a6d99")
    _FILL_HOVER = QColor("#27414f")
    _TEXT = QColor("#d6d6d6")
    _TEXT_HOVER = QColor("#9fd0f2")

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            return

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        chip = QRectF(option.rect).adjusted(5, 4, -5, -4)
        painter.setPen(QPen(self._BORDER_HOVER if hovered else self._BORDER, 1))
        painter.setBrush(self._FILL_HOVER if hovered else Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(chip, 5, 5)

        painter.setPen(self._TEXT_HOVER if hovered else self._TEXT)
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, index.data() or "")
        painter.restore()


def _slugify_project_name(name: str) -> str:
    """Turn a project name into a filesystem-safe slug for the folder and JSON file.

    Collapses every run of non-word characters to a single underscore and trims
    stray underscores, so only ``[A-Za-z0-9_-]`` survive. No-space slugs keep the
    project path friendly for the MATLAB/scripting entry point. Falls back to
    ``"project"`` when nothing usable remains. (Mirrors ``slugify`` in
    ``engine/io/annotation_io.py`` — kept as a small local duplicate rather than
    a shared import.)
    """
    import re

    slug = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_")
    return slug or "project"


def _natural_name_key(path: str) -> list[object]:
    """Order filenames with embedded numbers numerically (tiebreak for equal index)."""
    import re

    stem = Path(path).stem
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", stem)]


def generate_thumbnails(
    sections: list[Section],
    scale: float,
    parent: QWidget | None = None,
    title: str = "VERSO",
) -> None:
    """Generate working-resolution OME-TIFF thumbnails for ``sections``.

    Every section is downscaled by the project's single ``scale`` factor. Shown
    behind a modal progress dialog; failures fall back to lazy generation on
    first view. Shared by the New Project dialog and the Add-images action.
    """
    from PyQt6.QtWidgets import QApplication, QProgressDialog

    from verso.engine.io.image_io import ensure_working_copy

    n = len(sections)
    progress = QProgressDialog("Generating thumbnails…", "Skip", 0, n, parent)
    progress.setWindowTitle(title)
    progress.setMinimumWidth(300)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    for i, section in enumerate(sections):
        if progress.wasCanceled():
            break
        progress.setLabelText(
            f"Generating thumbnails… ({i + 1}/{n})\n{os.path.basename(section.original_path)}"
        )
        progress.setValue(i)
        QApplication.processEvents()
        # On failure the working copy is generated lazily on first view.
        with contextlib.suppress(Exception):
            ensure_working_copy(section, scale)

    progress.setValue(n)


class NewProjectDialog(QDialog):
    def __init__(
        self, parent: QWidget | None = None, initial_paths: list[str] | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(560)
        self._project: Project | None = None
        self._project_path: Path | None = None
        self._build_ui()
        if initial_paths:
            self._add_paths(initial_paths)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Project info ──────────────────────────────────────────────
        info_box = QGroupBox("Project")
        form = QFormLayout(info_box)

        self._name_edit = QLineEdit("My Experiment")
        form.addRow("Name:", self._name_edit)

        location_row = QWidget()
        h = QHBoxLayout(location_row)
        h.setContentsMargins(0, 0, 0, 0)
        self._location_edit = QLineEdit()
        self._location_edit.setPlaceholderText("Choose a folder to create the project in…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_location)
        h.addWidget(self._location_edit)
        h.addWidget(browse_btn)
        form.addRow("Location:", location_row)

        # Live preview of the self-contained folder VERSO will create from the
        # Name + Location above, so the auto-created subfolder is never a
        # surprise. Updates as either field changes.
        self._path_preview = QLabel()
        self._path_preview.setWordWrap(True)
        self._path_preview.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(self._path_preview)
        self._name_edit.textChanged.connect(self._update_path_preview)
        self._location_edit.textChanged.connect(self._update_path_preview)

        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(_KNOWN_ATLASES)
        self._atlas_combo.setEditable(True)
        form.addRow("Atlas:", self._atlas_combo)

        # Slicing orientation determines which atlas axis the series runs
        # along and is the axis VERSO interpolates anchorings across.
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItem("Coronal", "coronal")
        self._orientation_combo.addItem("Sagittal", "sagittal")
        self._orientation_combo.addItem("Horizontal", "horizontal")
        self._orientation_combo.setCurrentIndex(0)
        form.addRow("Slicing orientation:", self._orientation_combo)

        layout.addWidget(info_box)

        # ── Section images ────────────────────────────────────────────
        sections_box = QGroupBox("Section images")
        sv = QVBoxLayout(sections_box)

        self._file_table = QTableWidget(0, 2)
        self._file_table.setHorizontalHeaderLabels(["File", "Slice index"])
        self._file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._file_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        v_header = require(self._file_table.verticalHeader())
        v_header.setVisible(False)
        v_header.setDefaultSectionSize(30)
        self._file_table.setMinimumHeight(160)
        self._file_table.setAlternatingRowColors(True)
        self._file_table.setShowGrid(False)
        self._file_table.setStyleSheet(_TABLE_STYLE)
        header = require(self._file_table.horizontalHeader())
        header.setHighlightSections(False)
        header.setSectionResizeMode(_FILE_COL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_IDX_COL, QHeaderView.ResizeMode.ResizeToContents)
        # Slice index is the only editable column — render it as an input chip
        # with a hover affordance so users see it can be changed (mirrors the
        # overview table).
        self._file_table.setMouseTracking(True)
        self._file_table.setItemDelegateForColumn(_IDX_COL, _SliceIndexDelegate(self._file_table))
        self._file_table.itemChanged.connect(self._on_index_edited)
        sv.addWidget(self._file_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add images…")
        add_btn.clicked.connect(self._add_images)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        renumber_btn = QPushButton("Auto-number from names")
        renumber_btn.clicked.connect(self._auto_number)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(renumber_btn)
        btn_row.addStretch()
        sv.addLayout(btn_row)

        self._count_label = QLabel("0 images selected")
        self._count_label.setStyleSheet("color: #888; font-size: 11px;")
        sv.addWidget(self._count_label)

        layout.addWidget(sections_box)

        # ── Buttons ───────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_location(self) -> None:
        current = self._location_edit.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Project Location",
            current or str(Path.home()),
        )
        if directory:
            self._location_edit.setText(directory)

    def _update_path_preview(self) -> None:
        """Refresh the preview showing the folder + JSON filename that will be created."""
        location = self._location_edit.text().strip()
        if not location:
            self._path_preview.setText("")
            return
        slug = _slugify_project_name(self._name_edit.text())
        folder = Path(location) / slug
        self._path_preview.setText(f"Creates:  {folder}{os.sep}")

    def _current_entries(self) -> list[tuple[str, int]]:
        """Return ``(path, slice_index)`` for every table row, in row order."""
        entries: list[tuple[str, int]] = []
        for row in range(self._file_table.rowCount()):
            file_item = require(self._file_table.item(row, _FILE_COL))
            idx_item = require(self._file_table.item(row, _IDX_COL))
            path = file_item.data(Qt.ItemDataRole.UserRole)
            entries.append((path, int(idx_item.data(Qt.ItemDataRole.UserRole))))
        return entries

    def _set_entries(self, entries: list[tuple[str, int]]) -> None:
        """Rebuild the table from ``(path, slice_index)`` pairs, sorted by index."""
        ordered = sorted(entries, key=lambda e: (e[1], _natural_name_key(e[0])))
        self._file_table.blockSignals(True)
        self._file_table.setRowCount(len(ordered))
        for row, (path, index) in enumerate(ordered):
            file_item = QTableWidgetItem(os.path.basename(path))
            file_item.setData(Qt.ItemDataRole.UserRole, path)
            file_item.setToolTip(path)
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            idx_item = QTableWidgetItem(str(index))
            idx_item.setData(Qt.ItemDataRole.UserRole, int(index))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            idx_item.setToolTip("Double-click to edit")
            self._file_table.setItem(row, _FILE_COL, file_item)
            self._file_table.setItem(row, _IDX_COL, idx_item)
        self._file_table.blockSignals(False)
        self._update_count()

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Section Images", "", _IMAGE_FILTER)
        if paths:
            self._add_paths(paths)

    def _add_paths(self, paths: list[str]) -> None:
        """Merge ``paths`` into the table, skipping ones already present."""
        existing = {p for p, _ in self._current_entries()}
        merged = [p for p, _ in self._current_entries()]
        for path in paths:
            if path not in existing:
                merged.append(path)
                existing.add(path)
        # Re-guess across the whole set so the heuristic sees every filename.
        self._set_entries(list(zip(merged, guess_slice_indices(merged), strict=False)))

    def _remove_selected(self) -> None:
        selected = {idx.row() for idx in self._file_table.selectedIndexes()}
        if not selected:
            return
        entries = self._current_entries()
        kept = [e for row, e in enumerate(entries) if row not in selected]
        self._set_entries(kept)

    def _auto_number(self) -> None:
        """Re-run the filename heuristic, discarding any manual edits."""
        paths = [p for p, _ in self._current_entries()]
        self._set_entries(list(zip(paths, guess_slice_indices(paths), strict=False)))

    def _on_index_edited(self, item: QTableWidgetItem) -> None:
        """Validate an edited slice-index cell; revert non-integer input."""
        if item.column() != _IDX_COL:
            return
        text = item.text().strip()
        self._file_table.blockSignals(True)
        if text.isdigit():
            item.setData(Qt.ItemDataRole.UserRole, int(text))
            item.setText(text)
        else:
            prev = item.data(Qt.ItemDataRole.UserRole)
            item.setText(str(prev if prev is not None else 1))
        self._file_table.blockSignals(False)

    def _update_count(self) -> None:
        n = self._file_table.rowCount()
        self._count_label.setText(f"{n} image{'s' if n != 1 else ''} selected")

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        location = self._location_edit.text().strip()
        atlas = self._atlas_combo.currentText().strip()
        orientation = self._orientation_combo.currentData() or "coronal"
        interpolation_axis = SLICING_ORIENTATION_TO_AXIS[orientation]

        if not name:
            QMessageBox.warning(self, "Missing field", "Please enter a project name.")
            return
        if not location:
            QMessageBox.warning(self, "Missing field", "Please choose a project location.")
            return
        if self._file_table.rowCount() == 0:
            QMessageBox.warning(self, "No images", "Please add at least one section image.")
            return

        # A project is a self-contained folder named after the project, created
        # inside the location the user picked. The JSON and every subfolder live
        # inside it; the user never names the file directly.
        slug = _slugify_project_name(name)
        folder_path = Path(location) / slug
        if folder_path.exists() and any(folder_path.iterdir()):
            QMessageBox.warning(
                self,
                "Folder already exists",
                f"A non-empty folder named “{slug}” already exists in this location.\n\n"
                "Choose a different name or location so the new project does not "
                "overwrite existing files.",
            )
            return
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "thumbnails").mkdir(exist_ok=True)
        (folder_path / "masks").mkdir(exist_ok=True)
        (folder_path / "exports").mkdir(exist_ok=True)

        project_path = folder_path / f"{slug}_verso.json"

        # Build sections in increasing slice-index order so ``id`` (s001, s002…)
        # follows the physical series; sort_sections keeps the same order later.
        entries = sorted(self._current_entries(), key=lambda e: (e[1], _natural_name_key(e[0])))
        sections: list[Section] = []
        for i, (orig_path, slice_index) in enumerate(entries):
            sections.append(
                Section(
                    id=f"s{i + 1:03d}",
                    slice_index=slice_index,
                    original_path=orig_path,
                    thumbnail_path=str(folder_path / "thumbnails" / thumbnail_filename(orig_path)),
                    alignment=Alignment(status=AlignmentStatus.NOT_STARTED),
                    warp=WarpState(status=AlignmentStatus.NOT_STARTED),
                )
            )

        # Probe channels from the first source image to seed Project.channels
        # with sensible defaults. The user can edit these in the Adjust
        # brightness panel afterwards.
        first_path = Path(sections[0].original_path)
        try:
            channel_names = probe_channels(first_path)
        except Exception:
            channel_names = ["Ch 0"]
        project_channels = _default_channel_specs(channel_names, first_path.suffix)

        # One working scale for the whole batch, derived from the largest image
        # so its longest side fits within THUMBNAIL_MAX_SIDE.
        working_scale = compute_working_scale([s.original_path for s in sections])

        self._project = Project(
            name=name,
            atlas=AtlasRef(name=atlas),
            sections=sections,
            channels=project_channels,
            interpolation_axis=interpolation_axis,
            working_scale=working_scale,
        )
        self._project.sort_sections()
        self._project_path = project_path
        self._project.save(self._project_path)

        # Generate working-resolution thumbnails now so all views load quickly.
        self._generate_thumbnails(self._project.sections, self._project.working_scale)

        # Cache image dimensions and atlas resolution/shape so the saved project
        # is self-contained for pixel <-> atlas voxel mapping. This needs the
        # atlas, which is downloaded from brainglobe on first use; fail loudly if
        # it cannot be fetched rather than writing an unusable project.
        try:
            populate_metadata(self._project, folder_path)
        except AtlasUnavailableError as exc:
            QMessageBox.critical(
                self,
                "Atlas download failed",
                "Could not download the reference atlas. An internet connection is "
                "required the first time an atlas is used.\n\n"
                f"Details: {exc}",
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not create project",
                f"Failed to read image metadata while creating the project.\n\nDetails: {exc}",
            )
            return
        self._project.save(self._project_path)

        self.accept()

    def _generate_thumbnails(self, sections: list[Section], scale: float) -> None:
        """Generate working-resolution OME-TIFF thumbnails for all sections."""
        generate_thumbnails(sections, scale, self, title="New Project")

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result_project(self) -> Project | None:
        return self._project

    def result_project_path(self) -> Path | None:
        return self._project_path
