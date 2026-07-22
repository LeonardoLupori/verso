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

import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt
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

from verso.engine.io.image_io import (
    SUPPORTED_IMAGE_EXTENSIONS,
    image_dimensions,
    probe_channels,
)
from verso.engine.io.project_metadata import AtlasUnavailableError, populate_metadata
from verso.engine.io.quint_import import build_quint_project, match_registration_images
from verso.engine.io.quint_io import _BG_ATLAS_SHAPE, _resolve_atlas_name
from verso.engine.model.project import Project
from verso.gui.dialogs.new_project import (
    _KNOWN_ATLASES,
    _TABLE_STYLE,
    _default_channel_specs,
    _slugify_project_name,
    generate_thumbnails,
)
from verso.gui.utils import require

_IMAGE_FILTER = (
    "Images (" + " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS) + ");;All files (*)"
)

# Columns of the section-matching table.
_COL_NR = 0
_COL_JSON = 1
_COL_REG = 2
_COL_ORIG = 3

_LOCATE_HINT = "⚠  double-click to locate…"


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
        self._orig: dict[int, Path] = {}
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

        layout.addWidget(info_box)

        # ── Alignment file ────────────────────────────────────────────
        json_box = QGroupBox("Alignment file")
        jv = QVBoxLayout(json_box)
        json_row = QWidget()
        jh = QHBoxLayout(json_row)
        jh.setContentsMargins(0, 0, 0, 0)
        self._json_edit = QLineEdit()
        self._json_edit.setReadOnly(True)
        self._json_edit.setPlaceholderText("Choose a QuickNII / VisuAlign / DeepSlice .json file…")
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
        self._orig_edit.setPlaceholderText("Folder with the full-resolution originals…")
        orig_btn = QPushButton("Browse…")
        orig_btn.setFixedWidth(80)
        orig_btn.clicked.connect(self._browse_originals)
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
            self, "Choose alignment file", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._load_json(Path(path))

    def _load_json(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Cannot read file", f"Could not parse the JSON:\n\n{exc}")
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
        self._reg_edit.clear()
        self._orig_edit.clear()

        name = data.get("name")
        if name:
            self._name_edit.setText(str(name))

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

    def _browse_originals(self) -> None:
        folder = self._pick_image_folder("Choose full-resolution images folder")
        if not folder or self._json_path is None:
            return
        matched, _unmatched = match_registration_images(self._json_path, folder)
        self._orig = dict(matched)
        self._orig_edit.setText(folder)
        self._rebuild_table()
        self._update_ok_enabled()

    def _pick_image_folder(self, title: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, str(Path.home()))

    def _on_reuse_toggled(self, checked: bool) -> None:
        self._orig_row.setVisible(not checked)
        if checked:
            self._orig.clear()
            self._orig_edit.clear()
        self._rebuild_table()
        self._update_ok_enabled()

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Manually assign a single image to the slice on *row*."""
        if row < 0 or row >= len(self._filenames):
            return
        if col == _COL_REG:
            target = self._reg
            title = "Locate registration image"
        elif col == _COL_ORIG and not self._reuse_check.isChecked():
            target = self._orig
            title = "Locate full-resolution image"
        else:
            return
        path, _ = QFileDialog.getOpenFileName(self, title, "", _IMAGE_FILTER)
        if path:
            target[row] = Path(path)
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
                orig_item = self._image_item(self._orig.get(i))
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
            self._status_label.setText("Choose an alignment file to begin.")
        elif missing_reg or missing_orig:
            parts = []
            if missing_reg:
                parts.append(f"{missing_reg} registration image(s)")
            if missing_orig:
                parts.append(f"{missing_orig} full-resolution image(s)")
            self._status_label.setText(
                "Unmatched: " + ", ".join(parts) + " — double-click a red cell to locate it."
            )
        else:
            self._status_label.setText(f"All {n} sections matched. Ready to import.")

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

        try:
            project = build_quint_project(
                self._json_path,
                folder_path,
                registration_paths=dict(self._reg),
                original_paths=original_paths,
                atlas_name=self._atlas_combo.currentText().strip() or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not import", f"Failed to build the project:\n\n{exc}")
            return

        self._seed_channels(project)
        project.save(project_path)

        generate_thumbnails(project.sections, project.working_scale, self, title="Import")

        try:
            populate_metadata(project, folder_path)
        except AtlasUnavailableError as exc:
            QMessageBox.critical(
                self,
                "Atlas download failed",
                "Could not download the reference atlas. An internet connection is required "
                f"the first time an atlas is used.\n\nDetails: {exc}",
            )
            return
        except Exception as exc:
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
            data = json.loads(self._json_path.read_text(encoding="utf-8"))
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
