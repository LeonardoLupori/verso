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

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QToolBar,
    QWidget,
)

from verso.engine.io.quint_io import load_quicknii, load_visualign
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import DEFAULT_PROJECT_FILENAME, Project
from verso.gui.dialogs.brightness import BrightnessDialog
from verso.gui.dialogs.new_project import NewProjectDialog
from verso.gui.state import AppState
from verso.gui.views.align_view import AlignView
from verso.gui.views.overview_view import OverviewView
from verso.gui.views.prep_view import PrepView
from verso.gui.views.warp_view import WarpView
from verso.gui.widgets.filmstrip import Filmstrip
from verso.gui.widgets.properties import PropertiesPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel

_VIEW_OVERVIEW = 0
_VIEW_PREP = 1
_VIEW_ALIGN = 2
_VIEW_WARP = 3
_DEEPSLICE_PATH_KEY = "deepslice/env_path"


class _DeepSliceWorker(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        python_executable: str,
        reverse_section_order: bool = False,
    ) -> None:
        super().__init__()
        self._project = project
        self._python_executable = python_executable
        self._reverse_section_order = reverse_section_order

    def run(self) -> None:
        try:
            from verso.engine.deepslice import DeepSliceOptions, run_deepslice_suggestions

            result = run_deepslice_suggestions(
                self._project,
                self._python_executable,
                DeepSliceOptions(
                    species="mouse",
                    reverse_section_order=self._reverse_section_order,
                ),
            )
            self.done.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _BatchMaskWorker(QObject):
    done = pyqtSignal(int, list)

    def __init__(self, sections: list) -> None:
        super().__init__()
        self._sections = sections

    def run(self) -> None:
        errors: list[str] = []
        completed = 0
        from pathlib import Path

        from verso.engine.io.image_io import ensure_working_copy
        from verso.engine.preprocessing import detect_foreground, save_mask

        for section in self._sections:
            try:
                image = ensure_working_copy(section)
                if image is None:
                    errors.append(f"{Path(section.original_path).name}: no readable image")
                    continue
                mask = detect_foreground(image)
                masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
                mask_path = masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"
                save_mask(mask, mask_path)
                section.preprocessing.slice_mask_path = str(mask_path)
                completed += 1
            except Exception as exc:
                errors.append(f"{Path(section.original_path).name}: {exc}")
        self.done.emit(completed, errors)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VERSO")
        self.resize(1280, 800)

        self._state = AppState(self)
        self._current_mode = "overview"
        self._reverse_ap_proposal = False
        self._deepslice_thread: QThread | None = None
        self._deepslice_worker: _DeepSliceWorker | None = None
        self._deepslice_progress: QProgressDialog | None = None
        self._batch_mask_thread: QThread | None = None
        self._batch_mask_worker: _BatchMaskWorker | None = None
        self._batch_mask_progress: QProgressDialog | None = None
        self._brightness_dialog: BrightnessDialog | None = None

        # Coalesce rapid brightness-slider ticks into one redraw per event-loop
        # pass. Without this the GUI thread spends each tick re-compositing the
        # working-resolution image and the slider visibly lags the mouse.
        self._channels_pending: list | None = None
        self._channels_flush_timer = QTimer(self)
        self._channels_flush_timer.setSingleShot(True)
        self._channels_flush_timer.setInterval(0)
        self._channels_flush_timer.timeout.connect(self._flush_channels_changed)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._connect_signals()
        self._build_shortcuts()

        self._switch_view(_VIEW_OVERVIEW)

    def closeEvent(self, event) -> None:
        if self._deepslice_thread and self._deepslice_thread.isRunning():
            QMessageBox.information(
                self,
                "DeepSlice is running",
                "Wait for the DeepSlice proposal run to finish before closing VERSO.",
            )
            event.ignore()
            return
        if self._batch_mask_thread and self._batch_mask_thread.isRunning():
            QMessageBox.information(
                self,
                "Batch masks are running",
                "Wait for batch mask auto-detection to finish before closing VERSO.",
            )
            event.ignore()
            return
        self._save_prep_mask_before_transition()
        self._filmstrip.shutdown()
        self._overview.shutdown()
        super().closeEvent(event)

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

        act_configure_ds = QAction("Configure &DeepSlice Runtime…", self)
        act_configure_ds.triggered.connect(self._configure_deepslice)
        file_menu.addAction(act_configure_ds)

        file_menu.addSeparator()

        act_save = QAction("&Save project", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save project &as…", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

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

        images_menu = mb.addMenu("&Image")
        act_adjust = QAction("Adjust &channels/brightness…", self)
        act_adjust.triggered.connect(self._open_brightness_dialog)
        images_menu.addAction(act_adjust)
        batch_menu = images_menu.addMenu("&Batch processing")
        act_batch_mask = QAction("Autodetect slice mask for &all slices", self)
        act_batch_mask.triggered.connect(self._batch_autodetect_masks)
        batch_menu.addAction(act_batch_mask)

        export_menu = mb.addMenu("&Export")
        act_export_images = QAction("Images with atlas &overlay…", self)
        act_export_images.triggered.connect(self._export_images_with_overlay)
        export_menu.addAction(act_export_images)

        help_menu = mb.addMenu("&Help")
        act_atlas_info = QAction("&Atlas info…", self)
        act_atlas_info.triggered.connect(self._show_atlas_info)
        help_menu.addAction(act_atlas_info)
        act_project_info = QAction("&Project info…", self)
        act_project_info.triggered.connect(self._show_project_info)
        help_menu.addAction(act_project_info)

    def _open_brightness_dialog(self) -> None:
        """Show the floating brightness dialog, constructing it on first use."""
        if self._brightness_dialog is None:
            self._brightness_dialog = BrightnessDialog(self)
            self._brightness_dialog.channels_changed.connect(self._on_channels_changed)
            self._brightness_dialog.channels_committed.connect(
                self._on_channels_committed
            )
        project = self._state.project
        if project is not None:
            self._brightness_dialog.set_channels(project.channels)
        self._brightness_dialog.show()
        self._brightness_dialog.raise_()
        self._brightness_dialog.activateWindow()

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
            ("Align", _VIEW_ALIGN),
            ("Warp", _VIEW_WARP),
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
                "QPushButton:disabled { color: #555; background: #2e2e2e; border-color: #3a3a3a; }"
            )
            btn.clicked.connect(lambda _checked, i=idx: self._switch_view(i))
            if idx != _VIEW_OVERVIEW:
                btn.setEnabled(False)
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
        # Shared canvas + region bar + section/atlas/channels state.  Reparented
        # into whichever of AlignView / WarpView is currently active so zoom,
        # pan, and the channel-layer cache survive mode switches.
        self._panel = SectionCanvasPanel()
        self._align = AlignView(self._panel)
        self._warp = WarpView(self._panel)

        self._stack.addWidget(self._overview)   # 0
        self._stack.addWidget(self._prep)       # 1
        self._stack.addWidget(self._align)      # 2
        self._stack.addWidget(self._warp)       # 3

        # Park the panel inside AlignView's slot immediately.  If we left it as
        # a free-floating child of MainWindow (the default when ``SectionCanvasPanel``
        # is constructed without a layout parent), it renders at (0, 0) of the
        # main window and covers the menubar until the first Align/Warp switch.
        self._align.activate()

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

        # PropertiesPanel no longer pins its width, so its sizeHint inflates
        # the dock past what we want at launch.  Force the initial width here;
        # the user can still drag the splitter to resize.
        self.resizeDocks([right_dock], [260], Qt.Orientation.Horizontal)

    def _on_cp_style_changed(self, size: int, shape: str, color: str) -> None:
        self._warp.set_cp_style(size, shape, color)
        project = self._state.project
        if project is not None:
            project.cp_size = size
            project.cp_shape = shape
            project.cp_color = color

    def _build_shortcuts(self) -> None:
        self._section_shortcuts: list[QShortcut] = []
        for key, delta in (
            (Qt.Key.Key_Left, -1),
            (Qt.Key.Key_Right, 1),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda d=delta: self._step_section(d))
            self._section_shortcuts.append(shortcut)

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
        self._props.flip_v_changed.connect(self._on_flip_v_changed)
        self._props.mask_visibility_changed.connect(self._prep.set_mask_visible)
        self._props.mask_opacity_changed.connect(self._prep.set_mask_opacity)
        self._props.mask_color_changed.connect(self._prep.set_mask_color)
        self._props.mask_negative_changed.connect(self._prep.set_mask_negative)
        self._props.autodetect_requested.connect(self._on_prep_autodetect_requested)
        self._props.clear_mask_requested.connect(self._on_prep_clear_mask_requested)
        # Hemisphere subpanel — non-draw actions.
        self._props.lr_visibility_changed.connect(self._prep.set_lr_visible)
        self._props.lr_set_all_left_requested.connect(self._on_lr_set_all_left)
        self._props.lr_set_all_right_requested.connect(self._on_lr_set_all_right)
        self._props.lr_clear_requested.connect(self._on_lr_clear_requested)
        # Hemisphere — draw-mode lifecycle.
        self._props.lr_draw_mode_toggled.connect(self._on_lr_draw_mode_toggled)
        self._props.lr_apply_requested.connect(self._on_lr_draw_apply)
        self._props.lr_cancel_requested.connect(self._on_lr_draw_cancel)
        self._props.opacity_changed.connect(self._on_opacity_changed)
        self._props.overlay_color_changed.connect(self._panel.set_outline_color)
        self._props.overlay_mode_changed.connect(self._panel.set_overlay_mode)

        # PrepView edits
        self._prep.section_modified.connect(self._on_prep_modified)
        self._prep.mask_negative_changed.connect(self._props.set_mask_negative)
        self._prep.mask_visibility_changed.connect(self._props.set_mask_visible)

        # AlignView navigator + store/clear; WarpView edits.  Both views share
        # the same SectionCanvasPanel, so section-modified signals from either
        # collapse to the same handler.
        self._align.section_modified.connect(self._on_align_modified)
        self._align.anchoring_changed.connect(self._on_anchoring_changed)
        self._align.alignments_updated.connect(self._on_alignments_updated)
        self._align.reverse_requested.connect(self._reverse_section_order)
        self._align.deepslice_requested.connect(self._run_deepslice)
        self._align.default_proposal_requested.connect(self._revert_to_default_proposal)
        self._align.clear_all_alignments_requested.connect(self._clear_all_alignments)
        self._warp.section_modified.connect(self._on_align_modified)
        self._props.cp_style_changed.connect(self._on_cp_style_changed)

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _switch_view(self, index: int) -> None:
        leaving_prep = self._current_mode == "prep" and index != _VIEW_PREP
        if leaving_prep:
            self._save_prep_mask_before_transition()

        # Release panel hooks from whichever Align/Warp view currently owns it.
        if self._current_mode == "align":
            self._align.deactivate()
        elif self._current_mode == "warp":
            self._warp.deactivate()

        self._stack.setCurrentIndex(index)

        modes = ("overview", "prep", "align", "warp")
        self._current_mode = modes[index]

        for i, btn in enumerate(self._view_buttons):
            btn.setChecked(i == index)

        # Show filmstrip outside Overview; enable stored-alignment badges in Align/Warp
        self._bottom_dock.setVisible(index != _VIEW_OVERVIEW)
        if self._current_mode == "overview":
            self._overview.refresh()

        # Sync the newly visible view with the current section
        section = self._state.current_section
        project = self._state.project
        if self._current_mode == "prep":
            if self._prep._section is section:
                self._prep.refresh_display()
            else:
                self._prep.load_section(section)
            # Pick up any brightness edits made while Prep was hidden.
            if project is not None:
                self._prep.set_channels(project.channels)
        elif self._current_mode in ("align", "warp"):
            # Activate the new view (reparents the shared panel + installs hooks),
            # then ensure the section state is current.
            if self._current_mode == "align":
                self._align.activate()
            else:
                self._warp.activate()
            if self._panel.section is not section:
                self._panel.load_section(section)
            if project is not None:
                self._panel.set_channels(project.channels)
            self._props.set_align_warp_mode(self._current_mode)

        # Refresh properties with current section
        self._props.set_mode(self._current_mode)
        self._refresh_properties()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def _open_project(self) -> None:
        self._save_prep_mask_before_transition()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VERSO Project",
            "",
            "VERSO project (*.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                project_path = Path(path)
                project = Project.load(project_path)
                self._state.load_project(project, project_path)
            except Exception as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Cannot open project", str(exc))

    def _new_project(self) -> None:
        self._save_prep_mask_before_transition()
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            project = dlg.result_project()
            if project is not None:
                self._state.load_project(project, dlg.result_project_path())

    def _open_quicknii(self) -> None:
        self._save_prep_mask_before_transition()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QuickNII JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_quicknii(Path(path))
            self._state.load_project(project)

    def _open_visualign(self) -> None:
        self._save_prep_mask_before_transition()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VisuAlign JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_visualign(Path(path))
            self._state.load_project(project)

    def _save_project(self) -> None:
        self._prep.save_current_mask_if_dirty()
        if self._state.project is None:
            return
        if self._state.project_path is None:
            self._save_project_as()
            return
        self._write_project(self._state.project_path)

    def _save_project_as(self) -> None:
        self._prep.save_current_mask_if_dirty()
        if self._state.project is None:
            return
        current_path = self._state.project_path
        suggested = (
            str(current_path) if current_path is not None else DEFAULT_PROJECT_FILENAME
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", suggested, "JSON files (*.json)"
        )
        if path:
            project_path = Path(path)
            if project_path.suffix == "":
                project_path = project_path.with_suffix(".json")
            self._write_project(project_path)
            self._state.set_project_path(project_path)

    def _write_project(self, path: Path) -> None:
        project = self._state.project
        if project is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            project.save(path)
            self.statusBar().showMessage(f"Saved project to {path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot save project", str(exc))

    def _save_prep_mask_before_transition(self) -> None:
        self._prep.save_current_mask_if_dirty()
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)

    # ------------------------------------------------------------------
    # Slots — state changes
    # ------------------------------------------------------------------

    def _set_project_views_enabled(self, enabled: bool) -> None:
        for btn in self._view_buttons[1:]:
            btn.setEnabled(enabled)

    def _on_project_changed(self) -> None:
        project = self._state.project
        if project is None:
            self._project_label.setText("")
            self._set_project_views_enabled(False)
            return

        self._set_project_views_enabled(True)

        self._reverse_ap_proposal = False
        self._align.set_reverse_ap(False)

        # QuickNII interpolation needs atlas dimensions for the no-anchor and
        # one-anchor endpoint controls. If the atlas is still loading,
        # _on_atlas_loaded performs the exact QuickNII propagation.
        if self._state.atlas is not None:
            from verso.engine.registration import interpolate_anchorings

            interpolate_anchorings(
                project.sections,
                atlas_shape=self._state.atlas.shape,
                reverse_ap=self._reverse_ap_proposal,
            )
        self._sync_ap_mm(project.sections)

        self._project_label.setText(project.name)

        # Old project files stored channels per-section rather than at project
        # level, so project.channels may be empty on load.  Probe the first
        # available image to seed defaults so the canvas shows something.
        if not project.channels and project.sections:
            self._seed_channels_from_first_section(project)

        self._overview.load_project(project)
        self._filmstrip.populate(project.sections, project.channels)
        self._prep.set_channels(project.channels)
        self._panel.set_channels(project.channels)
        if self._brightness_dialog is not None:
            self._brightness_dialog.set_channels(project.channels)
        self._props.apply_cp_style(project.cp_size, project.cp_shape, project.cp_color)
        self._warp.set_cp_style(project.cp_size, project.cp_shape, project.cp_color)
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()

        if project.atlas:
            self._state.load_atlas(project.atlas.name)

        self._switch_view(_VIEW_OVERVIEW)

    def _seed_channels_from_first_section(self, project) -> None:
        """Populate project.channels when loading an old project that lacks them."""
        from pathlib import Path

        from verso.engine.io.image_io import probe_channels
        from verso.gui.dialogs.new_project import _default_channel_specs

        for section in project.sections:
            # Prefer the canonical thumbnail (fast metadata read); fall back to original.
            candidates = [
                Path(section.thumbnail_path) if section.thumbnail_path else None,
                Path(section.original_path),
            ]
            for probe_path in candidates:
                if probe_path and probe_path.exists():
                    try:
                        names = probe_channels(probe_path)
                    except Exception:
                        names = ["Ch 0"]
                    project.channels = _default_channel_specs(
                        names, Path(section.original_path).suffix
                    )
                    return

    def _on_atlas_loaded(self) -> None:
        atlas = self._state.atlas
        self._panel.set_atlas(atlas)
        if atlas is not None:
            project = self._state.project
            if project is not None:
                self._initialize_quicknii_anchorings(project.sections)
                self._sync_ap_mm(project.sections)
                self._panel.update_overlay()
                self._update_ap_plot()
                self._update_reverse_order_enabled()
                self._update_deepslice_enabled()

    def _on_atlas_error(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        self._update_deepslice_enabled()
        QMessageBox.warning(self, "Atlas load failed", message)

    def _on_section_changed(self, index: int) -> None:
        section = self._state.current_section
        self._filmstrip.set_current(index)

        if self._current_mode == "prep":
            self._prep.load_section(section)
        elif self._current_mode in ("align", "warp"):
            self._panel.load_section(section)

        self._refresh_properties()
        self._update_ap_plot()
        self._update_deepslice_enabled()

    def _step_section(self, delta: int) -> None:
        if not self._section_shortcuts_enabled():
            return
        project = self._state.project
        if project is None or not project.sections:
            return
        self._state.set_section(self._state.section_index + delta)

    def _section_shortcuts_enabled(self) -> bool:
        focus = self.focusWidget()
        return not isinstance(
            focus,
            (
                QAbstractSlider,
                QAbstractSpinBox,
                QComboBox,
                QLineEdit,
                QPlainTextEdit,
                QTextEdit,
            ),
        )

    def _on_section_activated(self, index: int) -> None:
        """Double-click in Overview → switch to Prep."""
        self._state.set_section(index)
        self._switch_view(_VIEW_PREP)
        self._prep.load_section(self._state.current_section)

    def _refresh_properties(self) -> None:
        self._props.update_section(self._state.current_section, self._current_mode)
        if self._current_mode == "prep":
            # Sync the draw button with PrepView's actual state — covers section
            # navigation, view switches, or anything else that may have torn the
            # editor down behind our back.
            self._props.set_lr_draw_active(self._prep.is_lr_draw_active())
            self._refresh_lr_status()

    # ------------------------------------------------------------------
    # Property change slots
    # ------------------------------------------------------------------

    def _clear_alignment_for_flip(self, section) -> None:
        section.alignment.anchoring = [0.0] * 9
        section.alignment.ap_position_mm = None
        section.alignment.status = AlignmentStatus.NOT_STARTED
        section.alignment.source = None
        section.alignment.stored_anchoring = None
        section.alignment.proposal_anchoring = None
        section.alignment.proposal_confidence = None
        section.alignment.proposal_run_id = None
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED

    def _on_flip_h_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_horizontal:
            return
        if self._prep.cancel_lr_draw_if_active():
            self._props.set_lr_draw_active(False)
        section.preprocessing.flip_horizontal = value
        self._clear_alignment_for_flip(section)
        self._after_flip_refresh()

    def _on_flip_v_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_vertical:
            return
        if self._prep.cancel_lr_draw_if_active():
            self._props.set_lr_draw_active(False)
        section.preprocessing.flip_vertical = value
        self._clear_alignment_for_flip(section)
        self._after_flip_refresh()

    def _after_flip_refresh(self) -> None:
        if self._current_mode == "prep":
            self._prep.refresh_display()
        elif self._current_mode in ("align", "warp"):
            self._panel.refresh_display()
        self._on_alignments_updated()

    def _on_opacity_changed(self, opacity: float) -> None:
        self._panel.canvas.set_overlay_opacity(opacity)

    def _on_channels_changed(self, channels: list) -> None:
        """Live updates while the user drags a brightness slider.

        Coalesces bursts of slider ticks: we stash the latest snapshot and
        flush it on the next event-loop pass, so the GUI thread never queues
        up more than one composite at a time.
        """
        self._channels_pending = channels
        if not self._channels_flush_timer.isActive():
            self._channels_flush_timer.start()

    def _flush_channels_changed(self) -> None:
        channels = self._channels_pending
        if channels is None:
            return
        self._channels_pending = None
        project = self._state.project
        if project is not None:
            project.channels = list(channels)
        # Refresh only the visible view. The other one is re-seeded on view
        # switch via _switch_view; the channel ImageItem stack keeps each raw
        # plane on the GPU, so the re-seed is essentially a LUT swap.
        if self._current_mode == "prep":
            self._prep.set_channels(channels)
        elif self._current_mode in ("align", "warp"):
            self._panel.set_channels(channels)

    def _on_channels_committed(self, channels: list) -> None:
        """Fires after the user releases a slider or makes a discrete edit."""
        project = self._state.project
        if project is not None:
            project.channels = list(channels)

    def _on_prep_autodetect_requested(self) -> None:
        self._prep.autodetect_mask()
        self._overview.refresh_row(self._state.section_index)

    def _on_prep_clear_mask_requested(self) -> None:
        self._prep.clear_mask()
        self._overview.refresh_row(self._state.section_index)

    def _on_lr_set_all_left(self) -> None:
        self._prep.set_lr_all(1)
        self._refresh_lr_status()
        self._overview.refresh_row(self._state.section_index)

    def _on_lr_set_all_right(self) -> None:
        self._prep.set_lr_all(2)
        self._refresh_lr_status()
        self._overview.refresh_row(self._state.section_index)

    def _on_lr_clear_requested(self) -> None:
        self._prep.clear_lr_mask()
        self._refresh_lr_status()
        self._overview.refresh_row(self._state.section_index)

    def _on_lr_draw_mode_toggled(self, active: bool) -> None:
        """Draw-line button toggled by the user."""
        if active:
            self._prep.enter_lr_draw_mode()
        else:
            # User untoggled the button without using Apply/Cancel → treat as Cancel.
            self._prep.exit_lr_draw_mode(apply=False)
        self._props.set_lr_draw_active(active)
        self._refresh_lr_status()

    def _on_lr_draw_apply(self) -> None:
        self._prep.exit_lr_draw_mode(apply=True)
        self._props.set_lr_draw_active(False)
        # section_modified emits inside exit_lr_draw_mode → status + project save
        # are handled by _on_prep_modified.  Refresh overview row eagerly.
        self._overview.refresh_row(self._state.section_index)

    def _on_lr_draw_cancel(self) -> None:
        self._prep.exit_lr_draw_mode(apply=False)
        self._props.set_lr_draw_active(False)
        self._refresh_lr_status()

    def _refresh_lr_status(self) -> None:
        """Push the current PrepView L/R state into the properties panel label."""
        self._props.set_lr_status(self._prep.lr_status_text())

    def _on_prep_modified(self) -> None:
        self._overview.refresh_row(self._state.section_index)
        self._refresh_lr_status()
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)

    def _batch_autodetect_masks(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        if self._batch_mask_thread and self._batch_mask_thread.isRunning():
            return

        reply = QMessageBox.question(
            self,
            "Auto-detect slice masks",
            "Run slice-mask auto-detection for all sections? "
            "Existing slice masks will be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._save_prep_mask_before_transition()
        self._show_batch_mask_progress()

        thread = QThread(self)
        worker = _BatchMaskWorker(list(project.sections))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_batch_masks_done)
        worker.done.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_batch_masks_finished)
        self._batch_mask_thread = thread
        self._batch_mask_worker = worker
        thread.start()

    def _show_batch_mask_progress(self) -> None:
        progress = QProgressDialog(
            "Auto-detecting slice masks...",
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle("Batch masks")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setModal(False)
        progress.show()
        self._batch_mask_progress = progress

    def _hide_batch_mask_progress(self) -> None:
        if self._batch_mask_progress is None:
            return
        self._batch_mask_progress.close()
        self._batch_mask_progress.deleteLater()
        self._batch_mask_progress = None

    def _on_batch_masks_done(self, completed: int, errors: list[str]) -> None:
        self._overview.refresh()
        if self._current_mode == "prep":
            self._prep.load_section(self._state.current_section)
        self._refresh_properties()
        self.statusBar().showMessage(f"Auto-detected {completed} slice masks", 5000)
        if errors:
            preview = "\n".join(errors[:8])
            suffix = "" if len(errors) <= 8 else f"\n...and {len(errors) - 8} more"
            QMessageBox.warning(
                self,
                "Some masks failed",
                f"{len(errors)} sections could not be processed:\n\n{preview}{suffix}",
            )

    def _on_batch_masks_finished(self) -> None:
        self._hide_batch_mask_progress()
        self._batch_mask_thread = None
        self._batch_mask_worker = None

    def _on_anchoring_changed(self, anchoring: list[float]) -> None:
        atlas = self._state.atlas
        if atlas is not None:
            ap_mm = self._anchoring_ap_mm(anchoring)
            self._props.update_ap_from_anchoring(ap_mm)
            section = self._state.current_section
            if section is not None:
                section.alignment.ap_position_mm = ap_mm
                if section.alignment.status != AlignmentStatus.COMPLETE:
                    section.alignment.source = "manual"
        self._overview.refresh_row(self._state.section_index)
        self._update_ap_plot()
        self._refresh_properties()

    def _on_alignments_updated(self) -> None:
        """Store or Clear was clicked — re-interpolate all non-stored sections."""
        project = self._state.project
        if project is None:
            return
        self._initialize_quicknii_anchorings(project.sections)
        self._sync_ap_mm(project.sections)
        self._panel.update_overlay()
        for i in range(len(project.sections)):
            self._overview.refresh_row(i)
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()

    def _on_align_modified(self) -> None:
        self._overview.refresh_row(self._state.section_index)
        self._refresh_properties()

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
        self._align.set_reverse_ap(self._reverse_ap_proposal)
        for section in project.sections:
            section.alignment.anchoring = [0.0] * 9
            section.alignment.ap_position_mm = None
            section.alignment.status = AlignmentStatus.NOT_STARTED
            section.alignment.source = None
            section.alignment.stored_anchoring = None
            section.alignment.proposal_anchoring = None
            section.alignment.proposal_confidence = None
            section.alignment.proposal_run_id = None
            section.warp.control_points.clear()
            section.warp.status = AlignmentStatus.NOT_STARTED

        self._initialize_quicknii_anchorings(project.sections)
        self._sync_ap_mm(project.sections)

        self._overview.refresh()
        self._on_section_changed(self._state.section_index)
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()

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

    def _update_deepslice_enabled(self, running: bool = False) -> None:
        project = self._state.project
        enabled = (
            project is not None
            and bool(project.sections)
            and self._state.atlas is not None
        )
        self._align.set_deepslice_enabled(enabled, running=running)

    def _sync_ap_mm(self, sections: list) -> None:
        """Populate ap_position_mm for every section that has a valid anchoring."""
        atlas = self._state.atlas
        if atlas is None:
            return
        for section in sections:
            if section.alignment.anchoring and any(
                v != 0.0 for v in section.alignment.anchoring
            ):
                section.alignment.ap_position_mm = self._anchoring_ap_mm(
                    section.alignment.anchoring
                )

    def _anchoring_ap_mm(self, anchoring: list[float]) -> float:
        atlas = self._state.atlas
        if atlas is None:
            return 0.0
        center = atlas.cut_center(anchoring)
        return atlas.ap_voxel_to_mm(center[atlas.ap_axis])

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

        from verso.engine.registration import _display_space_anchoring

        display_anchorings = []
        for section, _, _ in usable:
            is_complete = (
                section.alignment.status == AlignmentStatus.COMPLETE
            )
            display_anchorings.append(
                _display_space_anchoring(section) if is_complete else None
            )
        propagated = quicknii_coronal_series_anchorings(
            image_sizes=[(w, h) for _, w, h in usable],
            serial_numbers=[section.serial_number for section, _, _ in usable],
            atlas_shape=atlas.shape,
            stored_anchorings=display_anchorings,
            reverse_ap=self._reverse_ap_proposal,
            center_proposals=True,
        )

        stored_serials = {
            section.serial_number
            for (section, _, _), anch in zip(usable, display_anchorings)
            if anch is not None
        }

        for (section, _, _), anchoring, anch in zip(
            usable, propagated, display_anchorings
        ):
            if anch is not None:
                continue
            # Always sync sections that share a serial with a stored section —
            # they represent the same physical slice and must show the same image.
            is_serial_duplicate = section.serial_number in stored_serials
            has_existing = section.alignment.anchoring and any(
                v != 0.0 for v in section.alignment.anchoring
            )
            if (
                not is_serial_duplicate
                and has_existing
                and section.alignment.status != AlignmentStatus.NOT_STARTED
                and section.alignment.source != "quicknii_default"
            ):
                continue
            section.alignment.anchoring = anchoring
            if section.alignment.status == AlignmentStatus.NOT_STARTED:
                section.alignment.status = AlignmentStatus.IN_PROGRESS
            section.alignment.source = "quicknii_default"

    def _run_deepslice(self) -> None:
        project = self._state.project
        if project is None:
            return
        if self._deepslice_thread and self._deepslice_thread.isRunning():
            return

        executable = self._configured_deepslice_python()
        if executable is None:
            if not self._configure_deepslice():
                return
            executable = self._configured_deepslice_python()
        if executable is None:
            QMessageBox.warning(
                self,
                "DeepSlice not configured",
                "Select a Python environment folder or executable with DeepSlice installed first.",
            )
            return

        self._update_deepslice_enabled(running=True)
        self.statusBar().showMessage("Running DeepSlice suggestions...")
        self._show_deepslice_progress()

        thread = QThread(self)
        worker = _DeepSliceWorker(
            project,
            executable,
            reverse_section_order=self._reverse_ap_proposal,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_deepslice_done)
        worker.error.connect(self._on_deepslice_error)
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_deepslice_finished)
        self._deepslice_thread = thread
        self._deepslice_worker = worker
        thread.start()

    def _show_deepslice_progress(self) -> None:
        progress = QProgressDialog(
            "Running DeepSlice prediction...",
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle("DeepSlice")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setModal(False)
        progress.show()
        self._deepslice_progress = progress

    def _hide_deepslice_progress(self) -> None:
        if self._deepslice_progress is None:
            return
        self._deepslice_progress.close()
        self._deepslice_progress.deleteLater()
        self._deepslice_progress = None

    def _configure_deepslice(self) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle("Select DeepSlice Runtime")
        box.setText("Choose how to locate a Python runtime that has DeepSlice installed.")
        env_btn = box.addButton("Environment folder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Python executable", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(env_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is None or clicked == box.button(QMessageBox.StandardButton.Cancel):
            return False

        settings = QSettings("VERSO", "VERSO")
        current = str(settings.value(_DEEPSLICE_PATH_KEY, ""))
        if clicked == env_btn:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select DeepSlice Runtime",
                current,
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select DeepSlice Runtime",
                current,
                "Python executable (python.exe python);;All files (*)",
            )
        if not path:
            return False

        settings.setValue(_DEEPSLICE_PATH_KEY, path)
        executable = self._python_executable_from_deepslice_path(path)
        if executable is None:
            QMessageBox.warning(
                self,
                "DeepSlice path saved",
                "The path was saved, but VERSO could not find a Python executable there.",
            )
        else:
            self.statusBar().showMessage(f"DeepSlice Python set to {executable}", 5000)
        return True

    def _configured_deepslice_python(self) -> str | None:
        settings = QSettings("VERSO", "VERSO")
        path = settings.value(_DEEPSLICE_PATH_KEY, "")
        if not path:
            return None
        return self._python_executable_from_deepslice_path(str(path))

    def _python_executable_from_deepslice_path(self, path: str) -> str | None:
        p = Path(path)
        if p.is_file():
            return str(p)

        candidates = [
            p / "python.exe",
            p / "Scripts" / "python.exe",
            p / "bin" / "python",
            p / "bin" / "python3",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return None

    def _on_deepslice_done(self, result) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None:
            return
        from verso.engine.deepslice import apply_deepslice_suggestions_with_atlas

        applied = apply_deepslice_suggestions_with_atlas(
            project,
            result,
            atlas.shape if atlas is not None else None,
        )
        if atlas is not None:
            self._sync_ap_mm(project.sections)
        self._overview.refresh()
        self._panel.refresh_display()
        self._refresh_properties()
        self._update_ap_plot()
        self.statusBar().showMessage(f"Applied {applied} DeepSlice suggestions", 5000)

    def _on_deepslice_error(self, message: str) -> None:
        QMessageBox.warning(self, "DeepSlice failed", message)
        self.statusBar().showMessage("DeepSlice failed", 5000)

    def _on_deepslice_finished(self) -> None:
        self._hide_deepslice_progress()
        self._deepslice_thread = None
        self._deepslice_worker = None
        self._update_deepslice_enabled(running=False)

    def _revert_to_default_proposal(self) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None or atlas is None:
            return
        from verso.engine.deepslice import reset_in_progress_to_default_proposals

        changed = reset_in_progress_to_default_proposals(
            project.sections,
            atlas.shape,
            reverse_ap=self._reverse_ap_proposal,
        )
        self._sync_ap_mm(project.sections)
        self._overview.refresh()
        self._on_section_changed(self._state.section_index)
        self._update_ap_plot()
        self.statusBar().showMessage(f"Restored {changed} default proposals", 3000)

    def _clear_all_alignments(self) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None or atlas is None:
            return

        reply = QMessageBox.question(
            self,
            "Clear all alignments",
            "Clear every stored alignment, warp control point, and editable proposal, "
            "then restore VERSO's default AP proposal for all sections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from verso.engine.deepslice import reset_in_progress_to_default_proposals

        changed = reset_in_progress_to_default_proposals(
            project.sections,
            atlas.shape,
            reverse_ap=self._reverse_ap_proposal,
            include_complete=True,
        )
        self._sync_ap_mm(project.sections)
        self._overview.refresh()
        self._on_section_changed(self._state.section_index)
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self.statusBar().showMessage(
            f"Cleared all alignments and restored {changed} default proposals",
            5000,
        )

    def _update_ap_plot(self) -> None:
        project = self._state.project
        if project is None:
            return
        self._props.update_ap_plot(project.sections, self._state.section_index)

    def _maybe_create_pngs(self, export_path: str) -> None:
        """Offer to create PNG copies if any are missing next to the export."""
        project = self._state.project
        if project is None:
            return
        out_dir = Path(export_path).resolve().parent
        from verso.engine.io.quint_io import _export_image_filename
        missing = [
            s for s in project.sections
            if not (out_dir / _export_image_filename(s)).exists()
        ]
        if not missing:
            return
        reply = QMessageBox.question(
            self,
            "PNG images required",
            f"QuickNII and VisuAlign require PNG image files.\n\n"
            f"{len(missing)} of {len(project.sections)} section images are not "
            f"present as PNG in the export folder.\n\n"
            f"Create PNG copies now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from verso.engine.io.quint_io import write_section_pngs
            write_section_pngs(project, out_dir)

    def _export_quicknii_xml(self) -> None:
        self._save_prep_mask_before_transition()
        if self._state.project is None:
            return
        name = self._state.project.name
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QuickNII XML", f"{name}-quicknii.xml", "XML files (*.xml)"
        )
        if path:
            from verso.engine.io.quint_io import save_quicknii_xml
            atlas_shape = self._state.atlas.shape if self._state.atlas else None
            save_quicknii_xml(self._state.project, Path(path), atlas_shape=atlas_shape)
            self._maybe_create_pngs(path)

    def _export_quicknii(self) -> None:
        self._save_prep_mask_before_transition()
        if self._state.project is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        name = self._state.project.name
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QuickNII JSON", f"{name}-quicknii.json", "JSON files (*.json)"
        )
        if path:
            atlas_shape = self._state.atlas.shape if self._state.atlas else None
            from verso.engine.io.quint_io import save_quicknii
            save_quicknii(self._state.project, Path(path), atlas_shape=atlas_shape)
            self._maybe_create_pngs(path)

    def _export_visualign(self) -> None:
        self._save_prep_mask_before_transition()
        if self._state.project is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        name = self._state.project.name
        path, _ = QFileDialog.getSaveFileName(
            self, "Export VisuAlign JSON", f"{name}-visualign.json", "JSON files (*.json)"
        )
        if path:
            atlas_shape = self._state.atlas.shape if self._state.atlas else None
            from verso.engine.io.quint_io import save_visualign
            save_visualign(self._state.project, Path(path), atlas_shape=atlas_shape)
            self._maybe_create_pngs(path)

    def _show_info_dialog(self, title: str, rows: list[tuple[str, str]]) -> None:
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QFrame,
            QVBoxLayout,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(440)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(20, 18, 20, 14)
        outer.setSpacing(14)

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
        outer.addWidget(heading)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #444;")
        outer.addWidget(separator)

        form = QFormLayout()
        form.setSpacing(8)
        form.setHorizontalSpacing(16)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        for label, value in rows:
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("color: #888; font-size: 12px;")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet("color: #ddd; font-size: 12px;")
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(lbl, val)

        outer.addLayout(form)
        outer.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.accept)
        outer.addWidget(buttons)

        dlg.exec()

    def _show_atlas_info(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            project = self._state.project
            name = project.atlas.name if (project and project.atlas) else "(none)"
            self._show_info_dialog("Atlas info", [("Name", name), ("Status", "not yet loaded")])
            return

        ap, dv, lr = atlas._annotation.shape
        n_regions = len(atlas._color_dict) - 1  # exclude background (id 0)
        self._show_info_dialog("Atlas info", [
            ("Name", atlas.atlas_name),
            ("Resolution", f"{atlas.resolution_um:.1f} µm"),
            ("Volume shape", f"AP {ap}  ×  DV {dv}  ×  LR {lr}"),
            ("Brain regions", str(n_regions)),
        ])

    def _show_project_info(self) -> None:
        project = self._state.project
        if project is None:
            self._show_info_dialog("Project info", [("Status", "No project loaded")])
            return

        path_str = str(self._state.project_path) if self._state.project_path else "(not saved)"
        atlas_name = project.atlas.name if project.atlas else "(none)"
        channels = ", ".join(ch.name for ch in project.channels) if project.channels else "(none)"
        self._show_info_dialog("Project info", [
            ("Name", project.name),
            ("File", path_str),
            ("Atlas", atlas_name),
            ("Sections", str(len(project.sections))),
            ("Channels", channels),
        ])

    def _export_images_with_overlay(self) -> None:
        """Open the export dialog and write the requested PNGs to disk."""
        from datetime import datetime

        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtWidgets import QApplication

        from verso.engine.io.export_images import export_section
        from verso.gui.dialogs.export_images import ExportImagesDialog

        self._save_prep_mask_before_transition()

        project = self._state.project
        if project is None or not project.sections:
            QMessageBox.warning(self, "Export", "No project is loaded.")
            return
        atlas = self._state.atlas
        if atlas is None:
            QMessageBox.warning(
                self, "Export", "The atlas is still loading. Try again in a moment."
            )
            return
        if self._state.project_path is None:
            QMessageBox.warning(
                self,
                "Export",
                "Save the project to disk before exporting so VERSO knows where to "
                "write the exports folder.",
            )
            return

        selected_rows = self._overview.selected_rows()
        dlg = ExportImagesDialog(
            n_selected=len(selected_rows),
            n_total=len(project.sections),
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        if dlg.export_all():
            sections = list(project.sections)
        else:
            sections = [project.sections[i] for i in selected_rows]
        if not sections:
            QMessageBox.warning(self, "Export", "No sections selected.")
            return

        options = dlg.options()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = (
            self._state.project_path.parent
            / "exports"
            / f"images_with_overlay_{timestamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog(
            "Exporting images...", "Cancel", 0, len(sections), self
        )
        progress.setWindowTitle("Export images with atlas overlay")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()

        errors: list[str] = []
        for idx, section in enumerate(sections):
            if progress.wasCanceled():
                break
            progress.setLabelText(
                f"Exporting {idx + 1} / {len(sections)}: {Path(section.original_path).name}"
            )
            QApplication.processEvents()
            try:
                export_section(section, project, atlas, options, out_dir)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{Path(section.original_path).name}: {exc}")
            progress.setValue(idx + 1)
            QApplication.processEvents()

        progress.close()

        if errors:
            preview = "\n".join(errors[:8])
            suffix = "" if len(errors) <= 8 else f"\n...and {len(errors) - 8} more"
            QMessageBox.warning(
                self,
                "Export finished with errors",
                f"Wrote some images to:\n{out_dir}\n\nErrors:\n{preview}{suffix}",
            )
        else:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle("Export complete")
            box.setText(f"Wrote {len(sections)} sections to:\n{out_dir}")
            open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Ok)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
