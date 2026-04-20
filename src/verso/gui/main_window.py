"""Main application window.

Layout:
  ┌──────────────────────────────────────────────────────┐
  │  [Overview] [Prep] [Align/Warp]          menubar     │
  ├──────────────────────────────────────────────────────┤
  │                                      │               │
  │     central (QStackedWidget)         │  properties   │
  │     OverviewView / PrepView /        │  (right dock) │
  │     AlignView                        │               │
  ├──────────────────────────────────────┴───────────────┤
  │  filmstrip (bottom dock, hidden in Overview)         │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QWidget,
)

from verso.engine.io.quint_io import load_quicknii, load_visualign
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Project
from verso.gui.dialogs.new_project import NewProjectDialog
from verso.gui.state import AppState
from verso.gui.views.align_view import AlignView
from verso.gui.views.overview_view import OverviewView
from verso.gui.views.prep_view import PrepView
from verso.gui.widgets.filmstrip import Filmstrip
from verso.gui.widgets.properties import PropertiesPanel

_VIEW_OVERVIEW = 0
_VIEW_PREP = 1
_VIEW_ALIGN = 2


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VERSO")
        self.resize(1280, 800)

        self._state = AppState(self)
        self._current_mode = "overview"
        self._reverse_ap_proposal = False

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._connect_signals()

        self._switch_view(_VIEW_OVERVIEW)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Project…", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_open_qn = QAction("Import &QuickNII…", self)
        act_open_qn.triggered.connect(self._open_quicknii)
        file_menu.addAction(act_open_qn)

        act_open_va = QAction("Open &VisuAlign…", self)
        act_open_va.triggered.connect(self._open_visualign)
        file_menu.addAction(act_open_va)

        file_menu.addSeparator()

        act_save = QAction("&Save project", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_export_qn_xml = QAction("Export QuickNII &XML…", self)
        act_export_qn_xml.triggered.connect(self._export_quicknii_xml)
        file_menu.addAction(act_export_qn_xml)

        act_export_qn = QAction("Export &QuickNII JSON…", self)
        act_export_qn.triggered.connect(self._export_quicknii)
        file_menu.addAction(act_export_qn)

        act_export_va = QAction("Export &VisuAlign JSON…", self)
        act_export_va.triggered.connect(self._export_visualign)
        file_menu.addAction(act_export_va)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Views")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setStyleSheet(
            "QToolBar { background: #2a2a2a; border-bottom: 1px solid #444; "
            "spacing: 4px; padding: 4px; }"
        )
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        self._view_buttons: list[QPushButton] = []
        view_specs = [
            ("Overview", _VIEW_OVERVIEW),
            ("Prep", _VIEW_PREP),
            ("Align / Warp", _VIEW_ALIGN),
        ]
        for label, idx in view_specs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setStyleSheet(
                "QPushButton { border-radius: 4px; padding: 2px 14px; color: #ccc;"
                " background: #3a3a3a; border: 1px solid #555; }"
                "QPushButton:checked { background: #1e5a8a; color: #fff; border-color: #1e5a8a; }"
                "QPushButton:hover:!checked { background: #4a4a4a; }"
            )
            btn.clicked.connect(lambda _checked, i=idx: self._switch_view(i))
            self._view_buttons.append(btn)
            tb.addWidget(btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._project_label = QLabel("")
        self._project_label.setStyleSheet("color: #888; font-size: 11px; padding-right: 8px;")
        tb.addWidget(self._project_label)

    def _build_central(self) -> None:
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._overview = OverviewView()
        self._prep = PrepView()
        self._align = AlignView()

        self._stack.addWidget(self._overview)   # 0
        self._stack.addWidget(self._prep)       # 1
        self._stack.addWidget(self._align)      # 2

    def _build_docks(self) -> None:
        # Right: properties panel
        self._props = PropertiesPanel()
        right_dock = QDockWidget("Properties", self)
        right_dock.setWidget(self._props)
        right_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        right_dock.setTitleBarWidget(QWidget())   # hide title bar
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)
        self._right_dock = right_dock

        # Bottom: filmstrip
        self._filmstrip = Filmstrip()
        bottom_dock = QDockWidget("Filmstrip", self)
        bottom_dock.setWidget(self._filmstrip)
        bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        bottom_dock.setTitleBarWidget(QWidget())
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)
        self._bottom_dock = bottom_dock

    def _connect_signals(self) -> None:
        # State → views
        self._state.project_changed.connect(self._on_project_changed)
        self._state.section_changed.connect(self._on_section_changed)
        self._state.atlas_changed.connect(self._on_atlas_loaded)
        self._state.atlas_error.connect(self._on_atlas_error)

        # Overview interactions
        self._overview.section_activated.connect(self._on_section_activated)
        self._overview.section_selected.connect(self._state.set_section)

        # Filmstrip
        self._filmstrip.section_selected.connect(self._state.set_section)

        # Properties
        self._props.flip_h_changed.connect(self._on_flip_h_changed)
        self._props.opacity_changed.connect(self._on_opacity_changed)
        self._props.ap_changed.connect(self._on_ap_changed)

        # AlignView navigator + store/clear + sub-mode
        self._align.anchoring_changed.connect(self._on_anchoring_changed)
        self._align.alignments_updated.connect(self._on_alignments_updated)
        self._align.mode_changed.connect(self._props.set_align_warp_mode)
        self._align.reverse_requested.connect(self._reverse_section_order)
        self._props.cp_style_changed.connect(
            lambda size, shape, color: self._align.set_cp_style(size, shape, color)
        )

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _switch_view(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        modes = ("overview", "prep", "align")
        self._current_mode = modes[index]

        for i, btn in enumerate(self._view_buttons):
            btn.setChecked(i == index)

        # Show filmstrip only outside Overview; enable stored-alignment badges in align view
        self._bottom_dock.setVisible(index != _VIEW_OVERVIEW)
        self._filmstrip.set_align_mode(index == _VIEW_ALIGN)
        self._props.set_mode(self._current_mode)

        # Sync the newly visible view with the current section
        section = self._state.current_section
        if self._current_mode == "prep":
            if self._prep._section is section:
                self._prep.refresh_display()
            else:
                self._prep.load_section(section)
        elif self._current_mode == "align":
            if self._align._section is section:
                self._align.refresh_display()
            else:
                self._align.load_section(section)

        # Refresh properties with current section
        self._refresh_properties()
        self._update_reverse_order_enabled()

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VERSO Project",
            "",
            "VERSO project (project.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                project = Project.load(Path(path))
                self._state.load_project(project)
            except Exception as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Cannot open project", str(exc))

    def _new_project(self) -> None:
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            project = dlg.result_project()
            if project is not None:
                self._state.load_project(project)

    def _open_quicknii(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QuickNII JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_quicknii(Path(path))
            self._state.load_project(project)

    def _open_visualign(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VisuAlign JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_visualign(Path(path))
            self._state.load_project(project)

    def _save_project(self) -> None:
        if self._state.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project JSON", "project.json", "JSON files (*.json)"
        )
        if path:
            self._state.project.save(Path(path))

    # ------------------------------------------------------------------
    # Slots — state changes
    # ------------------------------------------------------------------

    def _on_project_changed(self) -> None:
        project = self._state.project
        if project is None:
            self._project_label.setText("")
            return

        self._reverse_ap_proposal = False

        # Propagate any stored anchorings to neighbours on load
        from verso.engine.registration import interpolate_anchorings
        interpolate_anchorings(project.sections)
        self._sync_ap_mm(project.sections)

        self._project_label.setText(project.name)
        self._overview.load_project(project)
        self._filmstrip.populate(project.sections)
        self._update_reverse_order_enabled()

        if project.atlas:
            self._props.set_atlas_name(project.atlas.name)
            self._props.set_atlas_loading(True)
            self._state.load_atlas(project.atlas.name)

        self._switch_view(_VIEW_OVERVIEW)

    def _on_atlas_loaded(self) -> None:
        atlas = self._state.atlas
        self._align.set_atlas(atlas)
        self._props.set_atlas_loading(False)
        if atlas is not None:
            self._props.set_ap_range(0.0, atlas.ap_extent_mm)
            project = self._state.project
            if project is not None:
                self._initialize_quicknii_anchorings(project.sections)
                self._sync_ap_mm(project.sections)
                self._align.update_overlay()
                self._update_ap_plot()
                self._update_reverse_order_enabled()

    def _on_atlas_error(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        self._props.set_atlas_loading(False)
        QMessageBox.warning(self, "Atlas load failed", message)

    def _on_section_changed(self, index: int) -> None:
        section = self._state.current_section
        self._filmstrip.set_current(index)

        if self._current_mode == "prep":
            self._prep.load_section(section)
        elif self._current_mode == "align":
            self._align.load_section(section)

        self._refresh_properties()
        self._update_ap_plot()

    def _on_section_activated(self, index: int) -> None:
        """Double-click in Overview → switch to Prep."""
        self._state.set_section(index)
        self._switch_view(_VIEW_PREP)
        self._prep.load_section(self._state.current_section)

    def _refresh_properties(self) -> None:
        self._props.update_section(self._state.current_section, self._current_mode)

    # ------------------------------------------------------------------
    # Property change slots
    # ------------------------------------------------------------------

    def _on_flip_h_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        section.preprocessing.flip_horizontal = value
        if self._current_mode == "prep":
            self._prep.refresh_display()
        elif self._current_mode == "align":
            self._align.refresh_display()
        self._overview.refresh_row(self._state.section_index)

    def _on_opacity_changed(self, opacity: float) -> None:
        self._align.canvas.set_overlay_opacity(opacity)

    def _on_ap_changed(self, ap_mm: float) -> None:
        section = self._state.current_section
        atlas = self._state.atlas
        if section is None:
            return
        section.alignment.ap_position_mm = ap_mm
        if atlas is not None:
            from verso.engine.registration import set_ap_position
            anchoring = section.alignment.anchoring
            if not anchoring or all(v == 0.0 for v in anchoring):
                raw_img = self._align._raw_image
                aspect = (raw_img.shape[1] / raw_img.shape[0]) if raw_img is not None else 1.0
                anchoring = atlas.default_anchoring(aspect_ratio=aspect)
            ap_voxel = atlas.ap_mm_to_voxel(ap_mm)
            section.alignment.anchoring = set_ap_position(anchoring, ap_voxel, atlas.ap_axis)
            self._align.update_overlay()
        self._overview.refresh_row(self._state.section_index)
        self._update_ap_plot()
        self._update_reverse_order_enabled()

    def _on_anchoring_changed(self, anchoring: list[float]) -> None:
        """Navigator panel changed the cut origin — sync AP spinbox."""
        atlas = self._state.atlas
        if atlas is not None:
            ap_mm = atlas.ap_voxel_to_mm(anchoring[1])
            self._props.update_ap_from_anchoring(ap_mm)
            section = self._state.current_section
            if section is not None:
                section.alignment.ap_position_mm = ap_mm
        self._overview.refresh_row(self._state.section_index)
        self._update_ap_plot()

    def _on_alignments_updated(self) -> None:
        """Store or Clear was clicked — re-interpolate all non-stored sections."""
        project = self._state.project
        if project is None:
            return
        self._initialize_quicknii_anchorings(project.sections)
        self._sync_ap_mm(project.sections)
        self._align.update_overlay()
        for i in range(len(project.sections)):
            self._overview.refresh_row(i)
        self._filmstrip.refresh_stored()
        self._update_ap_plot()
        self._update_reverse_order_enabled()

    def _reverse_section_order(self) -> None:
        """Reverse the startup AP proposal before any alignment is stored."""
        project = self._state.project
        if project is None or len(project.sections) < 2:
            return

        has_stored_alignment = any(
            section.alignment.status == AlignmentStatus.COMPLETE
            for section in project.sections
        )
        if has_stored_alignment:
            QMessageBox.information(
                self,
                "Cannot reverse proposal",
                "The startup proposal can only be reversed before any alignment is stored.",
            )
            return

        self._reverse_ap_proposal = not self._reverse_ap_proposal
        for section in project.sections:
            section.alignment.anchoring = [0.0] * 9
            section.alignment.ap_position_mm = None
            section.alignment.status = AlignmentStatus.NOT_STARTED
            section.warp.control_points.clear()
            section.warp.status = AlignmentStatus.NOT_STARTED

        self._initialize_quicknii_anchorings(project.sections)
        self._sync_ap_mm(project.sections)

        self._overview.refresh()
        self._on_section_changed(self._state.section_index)
        self._update_ap_plot()
        self._update_reverse_order_enabled()

    def _update_reverse_order_enabled(self) -> None:
        project = self._state.project
        if project is None:
            self._align.set_reverse_enabled(False)
            return
        has_stored_alignment = any(
            section.alignment.status == AlignmentStatus.COMPLETE
            for section in project.sections
        )
        self._align.set_reverse_enabled(
            self._state.atlas is not None
            and len(project.sections) > 1
            and not has_stored_alignment
        )

    def _sync_ap_mm(self, sections: list) -> None:
        """Populate ap_position_mm for every section that has a valid anchoring."""
        atlas = self._state.atlas
        if atlas is None:
            return
        for section in sections:
            if section.alignment.anchoring and any(
                v != 0.0 for v in section.alignment.anchoring
            ):
                section.alignment.ap_position_mm = atlas.ap_voxel_to_mm(
                    section.alignment.anchoring[1]
                )

    def _initialize_quicknii_anchorings(self, sections: list) -> None:
        """Initialize empty section planes with QuickNII-compatible stretch."""
        atlas = self._state.atlas
        if atlas is None:
            return

        from verso.engine.io.image_io import registration_dimensions
        from verso.engine.model.alignment import AlignmentStatus
        from verso.engine.registration import quicknii_coronal_series_anchorings

        usable = []
        for section in sections:
            try:
                w, h = registration_dimensions(section)
            except Exception:
                continue
            if w > 0 and h > 0:
                usable.append((section, w, h))

        if not usable:
            return

        stored_anchorings = [
            section.alignment.anchoring
            if section.alignment.status == AlignmentStatus.COMPLETE
            and section.alignment.anchoring
            and any(v != 0.0 for v in section.alignment.anchoring)
            else None
            for section, _, _ in usable
        ]
        propagated = quicknii_coronal_series_anchorings(
            image_sizes=[(w, h) for _, w, h in usable],
            serial_numbers=[section.serial_number for section, _, _ in usable],
            atlas_shape=atlas.shape,
            stored_anchorings=stored_anchorings,
            reverse_ap=self._reverse_ap_proposal,
        )

        for (section, _, _), anchoring, stored in zip(
            usable, propagated, stored_anchorings
        ):
            if stored is not None:
                continue
            section.alignment.anchoring = anchoring
            if section.alignment.status == AlignmentStatus.NOT_STARTED:
                section.alignment.status = AlignmentStatus.IN_PROGRESS

    def _update_ap_plot(self) -> None:
        project = self._state.project
        if project is None:
            return
        self._props.update_ap_plot(project.sections, self._state.section_index)

    def _export_quicknii_xml(self) -> None:
        if self._state.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QuickNII XML", "quicknii.xml", "XML files (*.xml)"
        )
        if path:
            from verso.engine.io.quint_io import save_quicknii_xml
            atlas_shape = self._state.atlas.shape if self._state.atlas else None
            save_quicknii_xml(self._state.project, Path(path), atlas_shape=atlas_shape)

    def _export_quicknii(self) -> None:
        if self._state.project is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QuickNII JSON", "quicknii.json", "JSON files (*.json)"
        )
        if path:
            from verso.engine.io.quint_io import save_quicknii
            save_quicknii(self._state.project, Path(path))

    def _export_visualign(self) -> None:
        if self._state.project is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export VisuAlign JSON", "visualign.json", "JSON files (*.json)"
        )
        if path:
            from verso.engine.io.quint_io import save_visualign
            save_visualign(self._state.project, Path(path))
