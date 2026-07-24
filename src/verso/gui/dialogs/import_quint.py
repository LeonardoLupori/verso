"""Import QuickNII / VisuAlign project dialog.

Turns a QuickNII/VisuAlign JSON alignment plus its section images into a real,
saved VERSO project (folder + working thumbnails + cached metadata), mirroring
:class:`~verso.gui.dialogs.new_project.NewProjectDialog`. The images
QuickNII/VisuAlign registered are required and matched to the JSON filenames; the
full-resolution originals are either those same images or a separate matched set.

All parsing / matching / coordinate math lives in the engine
(:mod:`verso.engine.io.quint_import`); this dialog only drives file selection and
calls the shared folder/thumbnail machinery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from verso.engine.anchoring import infer_interpolation_axis
from verso.engine.io.image_io import (
    SUPPORTED_IMAGE_EXTENSIONS,
    image_dimensions,
    probe_channels,
)
from verso.engine.io.project_metadata import AtlasUnavailableError, populate_metadata
from verso.engine.io.quint_import import (
    build_quint_project,
    filenames_are_thumbnails,
    match_originals_by_similarity,
    match_registration_images,
)
from verso.engine.io.quint_io import _BG_ATLAS_SHAPE, _resolve_atlas_name, read_quint_document
from verso.engine.model.project import (
    AXIS_INDEX_TO_NAME,
    AXIS_TO_SLICING_ORIENTATION,
    SLICING_ORIENTATION_TO_AXIS,
    Project,
)
from verso.gui.dialogs.new_project import (
    _KNOWN_ATLASES,
    _TABLE_STYLE,
    _default_channel_specs,
    _slugify_project_name,
    generate_thumbnails,
)
from verso.gui.utils import require

_log = logging.getLogger(__name__)

_IMAGE_FILTER = (
    "Images (" + " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS) + ");;All files (*)"
)

# Columns of the section-matching table.
_COL_NR = 0
_COL_JSON = 1
_COL_REG = 2
_COL_ORIG = 3

_LOCATE_HINT = "⚠  double-click to locate…"
# Full-res column placeholders. "Not chosen yet" is deliberately calm (no warning
# icon, muted colour) since it is the expected initial state; once the user has
# started adding originals, a still-open section gets a gentle amber prompt.
_ORIG_UNSET_HINT = "—"
_ORIG_ASSIGN_HINT = "double-click to assign…"
_AMBER = "#d9a441"
_MUTED = "#888"


class ImportQuintDialog(QDialog):
    """Collects a QuickNII/VisuAlign JSON + its images and builds a saved project."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import QuickNII / VisuAlign project")
        self.setMinimumWidth(680)

        self._project: Project | None = None
        self._project_path: Path | None = None

        # Parsed JSON state (index-aligned with table rows / slices).
        self._json_path: Path | None = None
        self._filenames: list[str] = []
        self._nrs: list[int] = []
        self._reg: dict[int, Path] = {}
        # Full-resolution originals: a pool of user-added files (``_orig_files``),
        # fuzzy-assigned to sections (``_orig``); ``_orig_manual`` marks sections
        # the user pinned by hand so re-assignment leaves them alone.
        self._orig: dict[int, Path] = {}
        self._orig_files: list[Path] = []
        self._orig_manual: set[int] = set()
        self._atlas_known = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Project info ──────────────────────────────────────────────
        info_box = QGroupBox("Project")
        form = QFormLayout(info_box)

        self._name_edit = QLineEdit("Imported Project")
        self._name_edit.textChanged.connect(self._update_path_preview)
        self._name_edit.textChanged.connect(self._update_ok_enabled)
        form.addRow("Name:", self._name_edit)

        location_row = QWidget()
        h = QHBoxLayout(location_row)
        h.setContentsMargins(0, 0, 0, 0)
        self._location_edit = QLineEdit()
        self._location_edit.setPlaceholderText("Choose a folder to create the project in…")
        self._location_edit.textChanged.connect(self._update_path_preview)
        self._location_edit.textChanged.connect(self._update_ok_enabled)
        loc_btn = QPushButton("Browse…")
        loc_btn.setFixedWidth(80)
        loc_btn.clicked.connect(self._browse_location)
        h.addWidget(self._location_edit)
        h.addWidget(loc_btn)
        form.addRow("Location:", location_row)

        self._path_preview = QLabel()
        self._path_preview.setWordWrap(True)
        self._path_preview.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(self._path_preview)

        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(_KNOWN_ATLASES)
        self._atlas_combo.setEditable(True)
        self._atlas_combo.setEnabled(False)
        form.addRow("Atlas:", self._atlas_combo)

        # Slicing orientation is inferred from the alignment's anchoring geometry
        # on load (QuickNII/VisuAlign do not store it), but stays editable so the
        # user can correct a mis-inference — as in New Project.
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItem("Coronal", "coronal")
        self._orientation_combo.addItem("Sagittal", "sagittal")
        self._orientation_combo.addItem("Horizontal", "horizontal")
        form.addRow("Slicing orientation:", self._orientation_combo)

        layout.addWidget(info_box)

        # ── Alignment file ────────────────────────────────────────────
        json_box = QGroupBox("Alignment file")
        jv = QVBoxLayout(json_box)
        json_row = QWidget()
        jh = QHBoxLayout(json_row)
        jh.setContentsMargins(0, 0, 0, 0)
        self._json_edit = QLineEdit()
        self._json_edit.setReadOnly(True)
        self._json_edit.setPlaceholderText(
            "Choose a QuickNII / VisuAlign / DeepSlice .json or QuickNII .xml file…"
        )
        json_btn = QPushButton("Browse…")
        json_btn.setFixedWidth(80)
        json_btn.clicked.connect(self._browse_json)
        jh.addWidget(self._json_edit)
        jh.addWidget(json_btn)
        jv.addWidget(json_row)
        self._atlas_warning = QLabel()
        self._atlas_warning.setWordWrap(True)
        self._atlas_warning.setStyleSheet("color: #d9a441; font-size: 11px;")
        self._atlas_warning.setVisible(False)
        jv.addWidget(self._atlas_warning)
        layout.addWidget(json_box)

        # ── Section images ────────────────────────────────────────────
        images_box = QGroupBox("Section images")
        iv = QVBoxLayout(images_box)

        reg_row = QWidget()
        rh = QHBoxLayout(reg_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(QLabel("Registration images:"))
        self._reg_edit = QLineEdit()
        self._reg_edit.setReadOnly(True)
        self._reg_edit.setPlaceholderText("Folder with the images QuickNII/VisuAlign registered…")
        reg_btn = QPushButton("Browse…")
        reg_btn.setFixedWidth(80)
        reg_btn.clicked.connect(self._browse_registration)
        rh.addWidget(self._reg_edit)
        rh.addWidget(reg_btn)
        iv.addWidget(reg_row)

        self._reuse_check = QCheckBox("Use these images as the full-resolution originals")
        self._reuse_check.setChecked(True)
        self._reuse_check.toggled.connect(self._on_reuse_toggled)
        iv.addWidget(self._reuse_check)

        self._orig_row = QWidget()
        oh = QHBoxLayout(self._orig_row)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.addWidget(QLabel("Full-resolution images:"))
        self._orig_edit = QLineEdit()
        self._orig_edit.setReadOnly(True)
        self._orig_edit.setPlaceholderText("Add one original image file per section…")
        orig_btn = QPushButton("Add images…")
        orig_btn.setFixedWidth(96)
        orig_btn.clicked.connect(self._add_originals)
        oh.addWidget(self._orig_edit)
        oh.addWidget(orig_btn)
        self._orig_row.setVisible(False)
        iv.addWidget(self._orig_row)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["nr", "JSON filename", "Registration", "Full-res"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(_TABLE_STYLE)
        self._table.setMinimumHeight(200)
        require(self._table.verticalHeader()).setVisible(False)
        header = require(self._table.horizontalHeader())
        header.setHighlightSections(False)
        header.setSectionResizeMode(_COL_NR, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_JSON, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_REG, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_ORIG, QHeaderView.ResizeMode.Stretch)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        iv.addWidget(self._table)

        self._status_label = QLabel("Choose an alignment file to begin.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        iv.addWidget(self._status_label)

        layout.addWidget(images_box)

        # ── Buttons ───────────────────────────────────────────────────
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._update_ok_enabled()

    # ------------------------------------------------------------------
    # Slots — file/folder selection
    # ------------------------------------------------------------------

    def _browse_location(self) -> None:
        current = self._location_edit.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self, "Choose Project Location", current or str(Path.home())
        )
        if directory:
            self._location_edit.setText(directory)

    def _update_path_preview(self) -> None:
        location = self._location_edit.text().strip()
        if not location:
            self._path_preview.setText("")
            return
        slug = _slugify_project_name(self._name_edit.text())
        self._path_preview.setText(f"Creates:  {Path(location) / slug}{os.sep}")

    def _browse_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose alignment file",
            "",
            "Alignment files (*.json *.xml);;JSON files (*.json);;"
            "QuickNII XML (*.xml);;All files (*)",
        )
        if path:
            self._load_json(Path(path))

    def _load_json(self, path: Path) -> None:
        try:
            data = read_quint_document(path)
        except Exception as exc:
            _log.exception("Cannot parse alignment file %s", path)
            QMessageBox.critical(
                self, "Cannot read file", f"Could not parse the alignment file:\n\n{exc}"
            )
            return
        raw_sections = data.get("slices")
        if raw_sections is None:
            raw_sections = data.get("sections", [])
        if not raw_sections:
            QMessageBox.warning(
                self, "No sections", "This file contains no slices/sections to import."
            )
            return

        self._json_path = path
        self._json_edit.setText(str(path))
        self._filenames = [str(s.get("filename", "")) for s in raw_sections]
        self._nrs = [int(s.get("nr", i + 1)) for i, s in enumerate(raw_sections)]
        self._reg.clear()
        self._orig.clear()
        self._orig_files.clear()
        self._orig_manual.clear()
        self._reg_edit.clear()
        self._orig_edit.clear()

        # QuickNII/VisuAlign names images by paths relative to the JSON's own
        # folder (e.g. "thumbnails/IMG-thumb.png"), so the registered images
        # almost always sit right beside it. Auto-resolve them from the JSON's
        # directory so the registration column fills itself in.
        auto_reg, _ = match_registration_images(path, path.parent)
        if auto_reg:
            self._reg = dict(auto_reg)
            self._reg_edit.setText(str(path.parent))
            # When those registered images are QUINT working *thumbnails*, they are
            # not the full-resolution originals — untick reuse so the Full-resolution
            # row is revealed and the user is prompted to point at their hi-res
            # images (they can re-tick reuse to import with the thumbnails instead).
            if filenames_are_thumbnails(self._filenames):
                self._reuse_check.setChecked(False)

        name = data.get("name")
        if name:
            self._name_edit.setText(str(name))

        # Recover the slicing orientation from the anchoring geometry and preset
        # the (editable) combo, so the imported project is complete by default.
        axis_index = infer_interpolation_axis([s.get("anchoring") for s in raw_sections])
        orientation = AXIS_TO_SLICING_ORIENTATION[AXIS_INDEX_TO_NAME[axis_index]]
        combo_index = self._orientation_combo.findData(orientation)
        if combo_index >= 0:
            self._orientation_combo.setCurrentIndex(combo_index)

        raw_target = str(data.get("target", ""))
        resolved = _resolve_atlas_name(raw_target) if raw_target else ""
        self._atlas_known = resolved in _BG_ATLAS_SHAPE
        if resolved:
            self._atlas_combo.setCurrentText(resolved)
        # A known target fixes the atlas (the anchoring was authored against it);
        # an unknown one is user-selectable, with a convention warning.
        self._atlas_combo.setEnabled(not self._atlas_known)
        if raw_target and not self._atlas_known:
            self._atlas_warning.setText(
                f"Atlas “{raw_target}” has no known QuickNII convention mapping. "
                "Pick the matching atlas below; anchoring may be offset if it is wrong."
            )
            self._atlas_warning.setVisible(True)
        else:
            self._atlas_warning.setVisible(False)

        self._rebuild_table()
        self._update_ok_enabled()

    def _browse_registration(self) -> None:
        folder = self._pick_image_folder("Choose registration-images folder")
        if not folder or self._json_path is None:
            return
        matched, _unmatched = match_registration_images(self._json_path, folder)
        self._reg = dict(matched)
        self._reg_edit.setText(folder)
        self._rebuild_table()
        self._update_ok_enabled()

    def _add_originals(self) -> None:
        """Add original image files and auto-assign them to sections by name similarity.

        Mirrors New Project's file picker — one original file per section. Added
        files join a pool that is fuzzy-matched to the still-unassigned sections,
        leaving any manual per-row assignments intact. Users fine-tune from the
        table by double-clicking a Full-res cell.
        """
        if self._json_path is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add full-resolution images", "", _IMAGE_FILTER
        )
        if not paths:
            return
        known = {str(p) for p in self._orig_files}
        for p in paths:
            if p not in known:
                self._orig_files.append(Path(p))
                known.add(p)
        self._reassign_originals()
        self._rebuild_table()
        self._update_ok_enabled()

    def _reassign_originals(self) -> None:
        """Fuzzy-assign the pooled original files to sections, one file per section.

        Sections the user pinned by hand (``_orig_manual``) keep their file and
        reserve it; every other section is (re)matched by filename similarity over
        the remaining pool, so the overall assignment stays unique.
        """
        manual = {i: self._orig[i] for i in self._orig_manual if i in self._orig}
        reserved = {str(p) for p in manual.values()}
        free = [p for p in self._orig_files if str(p) not in reserved]
        open_indices = [i for i in range(len(self._filenames)) if i not in manual]
        sub_names = [self._filenames[i] for i in open_indices]
        auto = match_originals_by_similarity(sub_names, free)
        result = dict(manual)
        for sub_i, path in auto.items():
            result[open_indices[sub_i]] = path
        self._orig = result
        self._update_orig_summary()

    def _update_orig_summary(self) -> None:
        n = len(self._filenames)
        if not self._orig_files:
            self._orig_edit.clear()
            return
        self._orig_edit.setText(
            f"{len(self._orig)}/{n} matched from {len(self._orig_files)} file(s)"
        )

    def _pick_image_folder(self, title: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, str(Path.home()))

    def _on_reuse_toggled(self, checked: bool) -> None:
        self._orig_row.setVisible(not checked)
        if checked:
            self._orig.clear()
            self._orig_files.clear()
            self._orig_manual.clear()
            self._orig_edit.clear()
        self._rebuild_table()
        self._update_ok_enabled()

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Manually assign a single image to the slice on *row* (fine-tuning)."""
        if row < 0 or row >= len(self._filenames):
            return
        if col == _COL_REG:
            path, _ = QFileDialog.getOpenFileName(
                self, "Locate registration image", "", _IMAGE_FILTER
            )
            if path:
                self._reg[row] = Path(path)
                self._rebuild_table()
                self._update_ok_enabled()
        elif col == _COL_ORIG and not self._reuse_check.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Locate full-resolution image", "", _IMAGE_FILTER
            )
            if path:
                # Pin this file to this section; reserve it and re-match the rest so
                # the assignment stays unique (one original file per section).
                p = Path(path)
                if p not in self._orig_files:
                    self._orig_files.append(p)
                self._orig_manual.add(row)
                self._orig[row] = p
                self._reassign_originals()
                self._rebuild_table()
                self._update_ok_enabled()

    # ------------------------------------------------------------------
    # Table / validation
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        reuse = self._reuse_check.isChecked()
        n = len(self._filenames)
        self._table.setRowCount(n)
        for i in range(n):
            nr_item = QTableWidgetItem(str(self._nrs[i]))
            nr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            json_item = QTableWidgetItem(Path(self._filenames[i].replace("\\", "/")).name)
            json_item.setToolTip(self._filenames[i])
            reg_item = self._image_item(self._reg.get(i))
            if reuse:
                orig_item = QTableWidgetItem("= registration")
                orig_item.setForeground(Qt.GlobalColor.gray)
            else:
                orig_item = self._orig_item(self._orig.get(i), started=bool(self._orig_files))
            for item in (nr_item, json_item, reg_item, orig_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, _COL_NR, nr_item)
            self._table.setItem(i, _COL_JSON, json_item)
            self._table.setItem(i, _COL_REG, reg_item)
            self._table.setItem(i, _COL_ORIG, orig_item)

    def _image_item(self, path: Path | None) -> QTableWidgetItem:
        if path is None:
            item = QTableWidgetItem(_LOCATE_HINT)
            item.setForeground(Qt.GlobalColor.red)
            return item
        item = QTableWidgetItem(path.name)
        item.setToolTip(str(path))
        return item

    def _orig_item(self, path: Path | None, *, started: bool) -> QTableWidgetItem:
        """Full-res cell: a filename, a calm placeholder, or a gentle assign prompt."""
        if path is not None:
            item = QTableWidgetItem(path.name)
            item.setToolTip(str(path))
            return item
        if not started:
            # No originals chosen yet — the expected initial state, not an error.
            item = QTableWidgetItem(_ORIG_UNSET_HINT)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(Qt.GlobalColor.gray)
            return item
        # Originals are being assigned but this section is still open.
        item = QTableWidgetItem(_ORIG_ASSIGN_HINT)
        item.setForeground(QColor(_AMBER))
        return item

    def _missing_counts(self) -> tuple[int, int]:
        """Return (missing_registration, missing_original) across all slices."""
        n = len(self._filenames)
        missing_reg = sum(1 for i in range(n) if i not in self._reg)
        if self._reuse_check.isChecked():
            missing_orig = 0
        else:
            missing_orig = sum(1 for i in range(n) if i not in self._orig)
        return missing_reg, missing_orig

    def _update_ok_enabled(self) -> None:
        ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is None:
            return
        n = len(self._filenames)
        ready = (
            self._json_path is not None
            and n > 0
            and bool(self._name_edit.text().strip())
            and bool(self._location_edit.text().strip())
        )
        missing_reg, missing_orig = self._missing_counts()
        ready = ready and missing_reg == 0 and missing_orig == 0
        ok_btn.setEnabled(ready)

        if self._json_path is None:
            self._set_status("Choose an alignment file to begin.")
        elif missing_reg:
            # A registered image could not be resolved — a genuine gap to fix.
            self._set_status(
                f"{missing_reg} registration image(s) not found — "
                "double-click a red cell to locate.",
                warn=True,
            )
        elif missing_orig and not self._orig_files:
            # Originals simply not chosen yet: invite the next step, do not warn.
            self._set_status(
                "Next, add the full-resolution images to match to each section — "
                "or tick “Use these images as the full-resolution originals” above."
            )
        elif missing_orig:
            self._set_status(
                f"{missing_orig} section(s) still need a full-resolution image — "
                "double-click to assign.",
                warn=True,
            )
        else:
            self._set_status(f"All {n} sections matched. Ready to import.")

    def _set_status(self, text: str, *, warn: bool = False) -> None:
        """Set the status line; ``warn`` tints it amber, otherwise it stays muted."""
        self._status_label.setStyleSheet(f"color: {_AMBER if warn else _MUTED}; font-size: 11px;")
        self._status_label.setText(text)

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        if self._json_path is None:
            return
        name = self._name_edit.text().strip()
        location = self._location_edit.text().strip()
        if not name or not location:
            QMessageBox.warning(self, "Missing field", "Please set a project name and location.")
            return

        slug = _slugify_project_name(name)
        folder_path = Path(location) / slug
        if folder_path.exists() and any(folder_path.iterdir()):
            QMessageBox.warning(
                self,
                "Folder already exists",
                f"A non-empty folder named “{slug}” already exists in this location.\n\n"
                "Choose a different name or location.",
            )
            return

        reuse = self._reuse_check.isChecked()
        original_paths = None if reuse else dict(self._orig)
        if not self._confirm_aspect_ratios(original_paths):
            return

        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "thumbnails").mkdir(exist_ok=True)
        (folder_path / "masks").mkdir(exist_ok=True)
        (folder_path / "exports").mkdir(exist_ok=True)
        project_path = folder_path / f"{slug}_verso.json"

        orientation = self._orientation_combo.currentData() or "coronal"
        try:
            project = build_quint_project(
                self._json_path,
                folder_path,
                registration_paths=dict(self._reg),
                original_paths=original_paths,
                atlas_name=self._atlas_combo.currentText().strip() or None,
                interpolation_axis=SLICING_ORIENTATION_TO_AXIS.get(orientation),
            )
        except Exception as exc:
            _log.exception("QuickNII import failed to build project from %s", self._json_path)
            QMessageBox.critical(self, "Could not import", f"Failed to build the project:\n\n{exc}")
            return

        self._seed_channels(project)
        project.save(project_path)

        generate_thumbnails(project.sections, project.working_scale, self, title="Import")

        try:
            populate_metadata(project, folder_path)
        except AtlasUnavailableError as exc:
            _log.exception("Atlas unavailable while populating metadata for %s", folder_path)
            QMessageBox.critical(
                self,
                "Atlas download failed",
                "Could not download the reference atlas. An internet connection is required "
                f"the first time an atlas is used.\n\nDetails: {exc}",
            )
            return
        except Exception as exc:
            _log.exception("Cannot read image metadata for %s", folder_path)
            QMessageBox.critical(
                self, "Could not import", f"Failed to read image metadata:\n\n{exc}"
            )
            return
        project.save(project_path)

        self._project = project
        self._project_path = project_path
        self.accept()

    def _seed_channels(self, project: Project) -> None:
        """Seed display channels from the first section's original (like New Project)."""
        if not project.sections:
            return
        first = project.sections[0]
        first_path = Path(first.original_path)
        try:
            channel_names = probe_channels(first_path, first.scene_index)
        except Exception:
            channel_names = ["Ch 0"]
        project.channels = _default_channel_specs(channel_names, first_path.suffix)

    def _confirm_aspect_ratios(self, original_paths: dict[int, Path] | None) -> bool:
        """Warn when separate originals differ in aspect ratio from the JSON dims.

        A large aspect mismatch means the originals are not the same framing as the
        registered images, so the imported anchoring/warp would not line up.
        """
        if not original_paths or self._json_path is None:
            return True
        try:
            data = read_quint_document(self._json_path)
        except Exception:
            return True
        raw = data.get("slices") or data.get("sections", [])
        mismatches: list[str] = []
        for i, orig in original_paths.items():
            if i >= len(raw):
                continue
            wr = int(raw[i].get("width", 0) or 0)
            hr = int(raw[i].get("height", 0) or 0)
            if wr <= 0 or hr <= 0:
                continue
            try:
                wo, ho = image_dimensions(orig)
            except Exception:
                continue
            if wo <= 0 or ho <= 0:
                continue
            if abs((wr / hr) - (wo / ho)) > 0.02 * (wr / hr):
                mismatches.append(Path(orig).name)
        if not mismatches:
            return True
        preview = ", ".join(mismatches[:5]) + (" …" if len(mismatches) > 5 else "")
        reply = QMessageBox.warning(
            self,
            "Aspect ratio mismatch",
            f"{len(mismatches)} full-resolution image(s) have a different aspect ratio than the "
            f"registered images ({preview}). The imported alignment may not line up.\n\n"
            "Import anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result_project(self) -> Project | None:
        return self._project

    def result_project_path(self) -> Path | None:
        return self._project_path
