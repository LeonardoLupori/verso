"""New Project dialog.

Collects:
  - Project name
  - Project folder (where project.json and subfolders will be created)
  - Atlas selection
  - One or more section image files (TIFF / PNG / JPEG)

On accept, call result() to get the configured Project object.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.alignment import Alignment, AlignmentStatus, WarpState
from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section

_KNOWN_ATLASES = [
    "allen_mouse_25um",
    "allen_mouse_10um",
    "allen_mouse_50um",
    "allen_rat_25um",
    "kim_mouse_25um",
]

_IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)"


class NewProjectDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(560)
        self._project: Project | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Project info ──────────────────────────────────────────────
        info_box = QGroupBox("Project")
        form = QFormLayout(info_box)

        self._name_edit = QLineEdit("My Experiment")
        form.addRow("Name:", self._name_edit)

        folder_row = QWidget()
        h = QHBoxLayout(folder_row)
        h.setContentsMargins(0, 0, 0, 0)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Choose a folder…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_folder)
        h.addWidget(self._folder_edit)
        h.addWidget(browse_btn)
        form.addRow("Project folder:", folder_row)

        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(_KNOWN_ATLASES)
        self._atlas_combo.setEditable(True)
        form.addRow("Atlas:", self._atlas_combo)

        layout.addWidget(info_box)

        # ── Section images ────────────────────────────────────────────
        sections_box = QGroupBox("Section images")
        sv = QVBoxLayout(sections_box)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_list.setMinimumHeight(160)
        sv.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add images…")
        add_btn.clicked.connect(self._add_images)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        sort_btn = QPushButton("Sort by name")
        sort_btn.clicked.connect(self._sort_by_name)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(sort_btn)
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

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self._folder_edit.setText(folder)

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Section Images", "", _IMAGE_FILTER
        )
        existing = {self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self._file_list.count())}
        for path in paths:
            if path not in existing:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                self._file_list.addItem(item)
        self._update_count()

    def _remove_selected(self) -> None:
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))
        self._update_count()

    def _sort_by_name(self) -> None:
        items = []
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            items.append((item.text(), item.data(Qt.ItemDataRole.UserRole)))
        items.sort(key=lambda x: x[0])
        self._file_list.clear()
        for name, path in items:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self._file_list.addItem(item)

    def _update_count(self) -> None:
        n = self._file_list.count()
        self._count_label.setText(f"{n} image{'s' if n != 1 else ''} selected")

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        folder = self._folder_edit.text().strip()
        atlas = self._atlas_combo.currentText().strip()

        if not name:
            QMessageBox.warning(self, "Missing field", "Please enter a project name.")
            return
        if not folder:
            QMessageBox.warning(self, "Missing field", "Please choose a project folder.")
            return
        if self._file_list.count() == 0:
            QMessageBox.warning(self, "No images", "Please add at least one section image.")
            return

        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "thumbnails").mkdir(exist_ok=True)
        (folder_path / "masks").mkdir(exist_ok=True)
        (folder_path / "alignments").mkdir(exist_ok=True)
        (folder_path / "exports").mkdir(exist_ok=True)

        sections: list[Section] = []
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            orig_path = item.data(Qt.ItemDataRole.UserRole)
            section_id = f"s{i + 1:03d}"
            sections.append(
                Section(
                    id=section_id,
                    serial_number=i + 1,
                    original_path=orig_path,
                    thumbnail_path=str(folder_path / "thumbnails" / f"{section_id}.png"),
                    alignment=Alignment(status=AlignmentStatus.NOT_STARTED),
                    warp=WarpState(status=AlignmentStatus.NOT_STARTED),
                )
            )

        self._project = Project(
            name=name,
            atlas=AtlasRef(name=atlas),
            sections=sections,
        )
        self._project.save(folder_path / "project.json")

        # Generate working-resolution thumbnails now so all views load quickly.
        self._generate_thumbnails(self._project.sections)

        self.accept()

    def _generate_thumbnails(self, sections: list[Section]) -> None:
        """Generate working-resolution PNG thumbnails for all sections."""
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from verso.engine.io.image_io import ensure_working_copy

        n = len(sections)
        progress = QProgressDialog("Generating thumbnails…", "Skip", 0, n, self)
        progress.setWindowTitle("New Project")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, section in enumerate(sections):
            if progress.wasCanceled():
                break
            progress.setLabelText(
                f"Generating thumbnails… ({i + 1}/{n})\n"
                f"{os.path.basename(section.original_path)}"
            )
            progress.setValue(i)
            QApplication.processEvents()
            try:
                ensure_working_copy(section)
            except Exception:
                pass  # will be generated lazily on first view

        progress.setValue(n)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result_project(self) -> Project | None:
        return self._project
