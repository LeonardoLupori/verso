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

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
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
class _DeepSliceWorker(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        reverse_section_order: bool = False,
        bad_section_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._reverse_section_order = reverse_section_order
        self._bad_section_ids = bad_section_ids or []

    def run(self) -> None:
        try:
            from verso.engine.deepslice import DeepSliceOptions, run_deepslice_suggestions

            result = run_deepslice_suggestions(
                self._project,
                DeepSliceOptions(
                    species="mouse",
                    reverse_section_order=self._reverse_section_order,
                    bad_section_ids=self._bad_section_ids,
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
        # Detected masks held in RAM (section.id -> bool array), drained by the
        # main thread into the resident prep-draft store on completion.  Nothing
        # is written to disk until the user saves.
        self.results: dict[str, object] = {}

    def run(self) -> None:
        errors: list[str] = []
        completed = 0
        from pathlib import Path

        from verso.engine.io.image_io import ensure_working_copy
        from verso.engine.preprocessing import detect_foreground

        for section in self._sections:
            try:
                image = ensure_working_copy(section)
                if image is None:
                    errors.append(f"{Path(section.original_path).name}: no readable image")
                    continue
                self.results[section.id] = detect_foreground(image)
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
        self._reverse_axis_proposal = False
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
        if not self._confirm_discard_active_draft():
            event.ignore()
            return
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

        file_menu.addSeparator()

        act_import_settings = QAction("Import &settings from project…", self)
        act_import_settings.triggered.connect(self._import_settings_from_project)
        file_menu.addAction(act_import_settings)

        file_menu.addSeparator()

        act_save = QAction("&Save all", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.setToolTip("Save all unsaved edits across every slice (Ctrl+S)")
        act_save.triggered.connect(self._save_all)
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

        batch_menu = mb.addMenu("&Batch")

        preprocess_menu = batch_menu.addMenu("&Preprocess")
        act_batch_mask = QAction("Autodetect slice mask for &all slices", self)
        act_batch_mask.triggered.connect(self._batch_autodetect_masks)
        preprocess_menu.addAction(act_batch_mask)
        preprocess_menu.addSeparator()
        self._act_clear_all_slice_masks = QAction("Clear all &slice masks…", self)
        self._act_clear_all_slice_masks.setEnabled(False)
        self._act_clear_all_slice_masks.triggered.connect(self._clear_all_slice_masks)
        preprocess_menu.addAction(self._act_clear_all_slice_masks)
        self._act_clear_all_lr_masks = QAction("Clear all &L/R masks…", self)
        self._act_clear_all_lr_masks.setEnabled(False)
        self._act_clear_all_lr_masks.triggered.connect(self._clear_all_lr_masks)
        preprocess_menu.addAction(self._act_clear_all_lr_masks)

        align_menu = batch_menu.addMenu("&Align")
        self._act_deepslice = QAction("Run &DeepSlice", self)
        self._act_deepslice.setEnabled(False)
        self._act_deepslice.triggered.connect(self._run_deepslice)
        align_menu.addAction(self._act_deepslice)

        self._act_default_proposal = QAction("&Default proposal", self)
        self._act_default_proposal.setEnabled(False)
        self._act_default_proposal.triggered.connect(self._revert_to_default_proposal)
        align_menu.addAction(self._act_default_proposal)

        self._act_reverse_proposal = QAction("&Reverse proposal", self)
        self._act_reverse_proposal.setEnabled(False)
        self._act_reverse_proposal.triggered.connect(self._reverse_section_order)
        align_menu.addAction(self._act_reverse_proposal)

        align_menu.addSeparator()
        self._act_clear_all_alignments = QAction("&Clear all alignments…", self)
        self._act_clear_all_alignments.setEnabled(False)
        self._act_clear_all_alignments.triggered.connect(self._clear_all_alignments)
        align_menu.addAction(self._act_clear_all_alignments)

        warp_menu = batch_menu.addMenu("&Warp")
        self._act_clear_all_warps = QAction("&Clear all warps…", self)
        self._act_clear_all_warps.setEnabled(False)
        self._act_clear_all_warps.triggered.connect(self._clear_all_warps)
        warp_menu.addAction(self._act_clear_all_warps)

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
            ("Preprocess", _VIEW_PREP),
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

        self._overview = OverviewView(self._state)
        self._prep = PrepView(self._state)
        # Shared canvas + region bar + section/atlas/channels state.  Reparented
        # into whichever of AlignView / WarpView is currently active so zoom,
        # pan, and the channel-layer cache survive mode switches.
        self._panel = SectionCanvasPanel()
        self._align = AlignView(self._panel, self._state)
        self._warp = WarpView(self._panel, self._state)

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
        self.resizeDocks([right_dock], [270], Qt.Orientation.Horizontal)

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
        self._state.dirty_changed.connect(self._on_dirty_changed)

        # Overview interactions
        self._overview.section_activated.connect(self._on_section_activated)
        self._overview.section_selected.connect(self._state.set_section)

        # Filmstrip
        self._filmstrip.section_selected.connect(self._state.set_section)

        # Properties
        flip = self._props.prep.flip
        flip.flip_h_changed.connect(self._on_flip_h_changed)
        flip.flip_v_changed.connect(self._on_flip_v_changed)
        mask = self._props.prep.mask
        mask.visibility_changed.connect(self._prep.set_mask_visible)
        mask.opacity_changed.connect(self._prep.set_mask_opacity)
        mask.color_changed.connect(self._prep.set_mask_color)
        mask.negative_changed.connect(self._prep.set_mask_negative)
        mask.draw_mode_changed.connect(self._prep.set_draw_mode)
        mask.brush_size_changed.connect(self._prep.set_brush_size)
        self._prep.brush_size_changed.connect(mask.set_brush_size)
        mask.autodetect_requested.connect(self._on_prep_autodetect_requested)
        mask.clear_requested.connect(self._on_prep_clear_mask_requested)
        mask.erode_requested.connect(lambda px: self._prep.apply_morph(px, "erode"))
        mask.expand_requested.connect(lambda px: self._prep.apply_morph(px, "expand"))
        # Hemisphere subpanel — non-draw actions.
        hemi = self._props.prep.hemisphere
        hemi.visibility_changed.connect(self._prep.set_lr_visible)
        hemi.set_all_left_requested.connect(self._on_lr_set_all_left)
        hemi.set_all_right_requested.connect(self._on_lr_set_all_right)
        hemi.clear_requested.connect(self._on_lr_clear_requested)
        # Hemisphere — draw-mode lifecycle.
        hemi.draw_mode_toggled.connect(self._on_lr_draw_mode_toggled)
        hemi.apply_requested.connect(self._on_lr_draw_apply)
        hemi.cancel_requested.connect(self._on_lr_draw_cancel)
        # Hemisphere — appearance.
        hemi.opacity_changed.connect(self._prep.set_lr_opacity)
        hemi.left_color_changed.connect(self._prep.set_lr_left_color)
        hemi.right_color_changed.connect(self._prep.set_lr_right_color)
        # Overlay lives in both Align and Warp pages with independent state.
        for overlay in (self._props.align.overlay, self._props.warp.overlay):
            overlay.opacity_changed.connect(self._on_opacity_changed)
            overlay.color_changed.connect(self._panel.set_outline_color)
            overlay.mode_changed.connect(self._panel.set_overlay_mode)

        # PrepView edits
        self._prep.mask_negative_changed.connect(mask.set_negative)
        self._prep.mask_visibility_changed.connect(mask.set_visible_state)
        self._prep.lr_status_changed.connect(self._refresh_lr_status)

        # AlignView navigator drives the anchoring; alignments_updated fires
        # when the user explicitly saves or clears, triggering re-interpolation.
        self._align.anchoring_changed.connect(self._on_anchoring_changed)
        self._align.alignments_updated.connect(self._on_alignments_updated)
        self._props.warp.cp.style_changed.connect(self._on_cp_style_changed)

        # Save / Clear bars and per-view dirty signals.
        view_bindings = (
            ("prep", self._prep, self._props.prep,
             self._on_prep_save_clicked, self._on_prep_clear_clicked),
            ("align", self._align, self._props.align,
             self._on_align_save_clicked, self._on_align_clear_clicked),
            ("warp", self._warp, self._props.warp,
             self._on_warp_save_clicked, self._on_warp_clear_clicked),
        )
        for step, view, page, on_save, on_clear in view_bindings:
            view.dirty_changed.connect(page.save_bar.set_dirty)
            # Mirror the view's dirty state into the persistent edit registry for
            # the section currently loaded in that view.
            view.dirty_changed.connect(
                lambda dirty, s=step: self._on_view_dirty_changed(s, dirty)
            )
            page.save_bar.save_requested.connect(on_save)
            page.save_bar.clear_requested.connect(on_clear)

        # A prep save/clear that flips the section invalidates its alignment+warp.
        self._prep.alignment_invalidated.connect(self._on_prep_invalidated_alignment)
        # CP add/delete changes the warp dot even when the dirty flag is unchanged
        # (e.g. removing the last CP → gray).
        self._warp.cp_changed.connect(self._refresh_current_step_dot)

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _switch_view(self, index: int) -> None:
        modes = ("overview", "prep", "align", "warp")
        leaving_mode = self._current_mode
        entering_mode = modes[index]
        if leaving_mode != entering_mode:
            self._flush_view_draft(leaving_mode)

        # Release panel hooks from whichever Align/Warp view currently owns it.
        if self._current_mode == "align":
            self._align.deactivate()
        elif self._current_mode == "warp":
            self._warp.deactivate()

        self._stack.setCurrentIndex(index)

        self._current_mode = entering_mode

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

        # Refresh properties with current section
        self._props.set_mode(self._current_mode)
        self._refresh_properties()
        self._refresh_clear_enabled()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        self._refresh_filmstrip_dots()

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def _open_project(self) -> None:
        if not self._confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VERSO Project",
            "",
            "VERSO project (*.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            self.open_project_path(Path(path))

    def open_project_path(self, project_path: Path) -> None:
        try:
            project = Project.load(project_path)
            self._state.load_project(project, project_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))

    def _new_project(self) -> None:
        if not self._confirm_discard_active_draft():
            return
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            project = dlg.result_project()
            if project is not None:
                self._state.load_project(project, dlg.result_project_path())

    def _open_quicknii(self) -> None:
        if not self._confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QuickNII JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_quicknii(Path(path))
            self._state.load_project(project)

    def _open_visualign(self) -> None:
        if not self._confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VisuAlign JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_visualign(Path(path))
            self._state.load_project(project)

    def _import_settings_from_project(self) -> None:
        """Copy channel colors and control-point styling from another project."""
        project = self._state.project
        if project is None:
            QMessageBox.information(
                self,
                "No project loaded",
                "Open or create a project before importing settings.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import settings from project",
            "",
            "VERSO project (*.json);;JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            source = Project.load(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Cannot read project", str(exc))
            return

        from verso.engine.io.project_io import import_project_styling

        import_project_styling(project, source)

        # Refresh widgets that depend on channel display or CP styling.
        self._panel.set_channels(project.channels)
        self._prep.set_channels(project.channels)
        if self._brightness_dialog is not None:
            self._brightness_dialog.set_channels(project.channels)
        self._filmstrip.populate(project.sections, project.channels)
        self._props.warp.cp.apply_style(project.cp_size, project.cp_shape, project.cp_color)
        self._warp.set_cp_style(project.cp_size, project.cp_shape, project.cp_color)

        if self._state.project_path is not None:
            self._write_project(self._state.project_path)

        self.statusBar().showMessage(
            f"Imported settings from {Path(path).name}", 3000
        )

    def _save_all(self) -> bool:
        """Persist every unsaved edit across all slices/steps (Ctrl+S / menu).

        Returns True if the project was saved, False if there's no project or the
        user cancelled a Save-As prompt.
        """
        from verso.engine.drafts import (
            commit_alignment,
            commit_warp,
            persist_prep_draft,
        )

        project = self._state.project
        if project is None:
            return False

        # 1. Persist the active view's in-RAM edits first — this materializes
        #    prep masks held only in the view and seeds a default align plane on
        #    an explicit save — then it clears that section/step from the registry.
        active = self._active_view()
        if active is not None and active.is_dirty():
            active.save()

        # 2. Persist every remaining dirty (section, step).  Snapshot the list up
        #    front since we mutate the registry inside the loop.
        for section, steps in self._state.dirty_sections():
            if "prep" in steps:
                draft = self._state.pop_prep_draft(section.id)
                if draft is not None and persist_prep_draft(section, draft):
                    # Flip invalidated the alignment + warp.
                    self._state.clear_dirty(section.id, "align")
                    self._state.clear_dirty(section.id, "warp")
                self._state.clear_dirty(section.id, "prep")
            # Commit align before warp so warp can reach COMPLETE.
            if "align" in steps and self._state.is_dirty(section.id, "align"):
                commit_alignment(section)
                self._state.clear_dirty(section.id, "align")
            if "warp" in steps and self._state.is_dirty(section.id, "warp"):
                commit_warp(section)
                self._state.clear_dirty(section.id, "warp")

        # 3. Re-interpolate non-stored sections now that all saves are COMPLETE,
        #    and keep position_mm in sync for the AP plot.
        self._initialize_quicknii_anchorings(project.sections)
        self._sync_position_mm(project.sections)

        # 4. Single project.json write + dependent-UI refresh.
        if self._state.project_path is None:
            self._save_project_as()
            if self._state.project_path is None:
                return False  # user cancelled the Save-As dialog
        else:
            self._write_project(self._state.project_path)
        if self._current_mode in ("align", "warp"):
            self._panel.update_overlay()
        self._overview.refresh()
        self._update_ap_plot()
        self._refresh_clear_enabled()
        self._refresh_filmstrip_dots()
        return True

    def _save_project_as(self) -> None:
        self._save_active_view()
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
            self._refresh_clear_enabled()

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

    # ------------------------------------------------------------------
    # Per-view draft save / clear / discard
    # ------------------------------------------------------------------

    def _active_view(self):
        if self._current_mode == "prep":
            return self._prep
        if self._current_mode == "align":
            return self._align
        if self._current_mode == "warp":
            return self._warp
        return None

    def _save_active_view(self) -> bool:
        view = self._active_view()
        if view is None:
            return False
        return view.save()

    def _flush_view_draft(self, mode: str) -> None:
        """Persist the leaving view's in-RAM edits before a slice/view swap.

        Edits are no longer discarded on navigation.  Prep flushes its mask
        arrays into the resident draft store (keyed by section id); Align/Warp
        keep their edits directly on the Section, so nothing to do.
        """
        if mode == "prep":
            self._prep.flush_draft()

    def _on_view_dirty_changed(self, step: str, dirty: bool) -> None:
        """Mirror a view's dirty state into the registry for the current section."""
        section = self._state.current_section
        if section is None:
            return
        if dirty:
            self._state.mark_dirty(section.id, step)
        else:
            self._state.clear_dirty(section.id, step)

    def _on_prep_invalidated_alignment(self) -> None:
        """A prep flip wiped the current section's alignment + warp."""
        section = self._state.current_section
        if section is None:
            return
        self._state.clear_dirty(section.id, "align")
        self._state.clear_dirty(section.id, "warp")

    def _confirm_discard_active_draft(self) -> bool:
        """Prompt when unsaved edits exist anywhere before a disruptive operation.

        Offers Save all / Discard all / Cancel across every dirty section.
        Returns True if the caller may proceed (nothing dirty, or the user chose
        Save all / Discard all); False on Cancel (or a cancelled Save-As).
        """
        if not self._state.any_dirty():
            return True
        n = len(self._state.dirty_sections())
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"You have unsaved edits in {n} section(s). "
            "Save them before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_all()
        if reply == QMessageBox.StandardButton.Discard:
            self._discard_all()
            return True
        return False

    def _discard_all(self) -> None:
        """Drop every unsaved edit by reloading the last-saved project from disk."""
        path = self._state.project_path
        self._state.clear_all_edits()
        if path is None or not path.exists():
            return
        try:
            project = Project.load(path)
        except Exception:
            return
        self._state.load_project(project, path)

    def _after_view_save(self) -> None:
        """Refresh dependent UI after a per-view save/clear and write project."""
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)
        self._overview.refresh()
        self._update_ap_plot()
        self._refresh_clear_enabled()
        self._refresh_filmstrip_dots()

    def _refresh_clear_enabled(self) -> None:
        """Sync each save bar's Clear button to whether the slice has state to wipe."""
        self._props.prep.save_bar.set_clear_enabled(self._prep.has_persisted_state())
        self._props.align.save_bar.set_clear_enabled(self._align.has_persisted_state())
        self._props.warp.save_bar.set_clear_enabled(self._warp.has_persisted_state())

    # ------------------------------------------------------------------
    # Filmstrip status dots
    # ------------------------------------------------------------------

    def _refresh_filmstrip_dots(self) -> None:
        """Recompute all filmstrip status dots for the active view's step."""
        project = self._state.project
        step = self._current_mode
        if project is None or step not in ("prep", "align", "warp"):
            return
        from verso.engine.model.status import section_step_color
        colors = [
            section_step_color(s, step, dirty=self._state.is_dirty(s.id, step))
            for s in project.sections
        ]
        self._filmstrip.set_statuses(colors)

    def _refresh_current_step_dot(self) -> None:
        """Refresh the current section's filmstrip dot for the active step.

        Used when a section's status changes without the dirty flag flipping
        (e.g. removing the last warp control point keeps it dirty but the dot
        must go gray).
        """
        project = self._state.project
        step = self._current_mode
        if project is None or step not in ("prep", "align", "warp"):
            return
        section = self._state.current_section
        if section is None:
            return
        from verso.engine.model.status import section_step_color
        color = section_step_color(
            section, step, dirty=self._state.is_dirty(section.id, step)
        )
        self._filmstrip.set_status_color(self._state.section_index, color)

    def _on_dirty_changed(self, section_id: str, step: str) -> None:
        """Incrementally update one filmstrip dot when a section's dirty flips."""
        project = self._state.project
        if project is None or step != self._current_mode:
            return
        from verso.engine.model.status import section_step_color
        for i, section in enumerate(project.sections):
            if section.id == section_id:
                color = section_step_color(
                    section, step, dirty=self._state.is_dirty(section_id, step)
                )
                self._filmstrip.set_status_color(i, color)
                return

    def _on_prep_save_clicked(self) -> None:
        if self._prep.save():
            self._after_view_save()

    def _on_prep_clear_clicked(self) -> None:
        if self._prep.clear():
            self._after_view_save()

    def _on_align_save_clicked(self) -> None:
        if self._align.save():
            self._after_view_save()

    def _on_align_clear_clicked(self) -> None:
        if self._align.clear():
            self._after_view_save()

    def _on_warp_save_clicked(self) -> None:
        if self._warp.save():
            self._after_view_save()

    def _on_warp_clear_clicked(self) -> None:
        if self._warp.clear():
            self._after_view_save()

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

        self._reverse_axis_proposal = False
        self._align.set_reverse_axis(False)
        self._align.set_interpolation_axis(project.interpolation_axis_index)
        self._props.align.ap_plot.set_axis_name(project.interpolation_axis)

        # QuickNII interpolation needs atlas dimensions for the no-anchor and
        # one-anchor endpoint controls. If the atlas is still loading,
        # _on_atlas_loaded performs the exact QuickNII propagation.
        if self._state.atlas is not None:
            from verso.engine.registration import interpolate_anchorings

            interpolate_anchorings(
                project.sections,
                atlas_shape=self._state.atlas.shape,
                interpolation_axis=project.interpolation_axis_index,
                reverse_axis=self._reverse_axis_proposal,
            )
        self._sync_position_mm(project.sections)

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
        self._props.warp.cp.apply_style(project.cp_size, project.cp_shape, project.cp_color)
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
                self._sync_position_mm(project.sections)
                self._panel.update_overlay()
                self._update_ap_plot()
                self._update_reverse_order_enabled()
                self._update_deepslice_enabled()

    def _on_atlas_error(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        self._update_deepslice_enabled()
        QMessageBox.warning(self, "Atlas load failed", message)

    def _on_section_changed(self, index: int) -> None:
        # Flush (not discard) the leaving section's in-RAM edits so they persist
        # across the swap; prep masks move into the resident draft store.
        self._flush_view_draft(self._current_mode)

        section = self._state.current_section
        self._filmstrip.set_current(index)

        if self._current_mode == "prep":
            self._prep.load_section(section)
        elif self._current_mode in ("align", "warp"):
            self._panel.load_section(section)

        # Keep position_mm in lockstep with the (possibly interpolated) anchoring
        # so the AP-plot white dot is correct without requiring a save first.
        if section is not None:
            self._sync_position_mm([section])

        self._overview.refresh()
        self._refresh_properties()
        self._refresh_clear_enabled()
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
            self._props.prep.hemisphere.set_draw_active(self._prep.is_lr_draw_active())
            self._refresh_lr_status()

    # ------------------------------------------------------------------
    # Property change slots
    # ------------------------------------------------------------------

    def _on_flip_h_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_horizontal:
            return
        if self._prep.cancel_lr_draw_if_active():
            self._props.prep.hemisphere.set_draw_active(False)
        section.preprocessing.flip_horizontal = value
        self._prep.mark_flip_changed()
        self._prep.refresh_display()

    def _on_flip_v_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_vertical:
            return
        if self._prep.cancel_lr_draw_if_active():
            self._props.prep.hemisphere.set_draw_active(False)
        section.preprocessing.flip_vertical = value
        self._prep.mark_flip_changed()
        self._prep.refresh_display()

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

    def _on_prep_clear_mask_requested(self) -> None:
        self._prep.clear_mask()

    def _on_lr_set_all_left(self) -> None:
        self._prep.set_lr_all(1)

    def _on_lr_set_all_right(self) -> None:
        self._prep.set_lr_all(2)

    def _on_lr_clear_requested(self) -> None:
        self._prep.clear_lr_mask()

    def _on_lr_draw_mode_toggled(self, active: bool) -> None:
        """Draw-line button toggled by the user."""
        if active:
            self._prep.enter_lr_draw_mode()
        else:
            # User untoggled the button without using Apply/Cancel → treat as Cancel.
            self._prep.exit_lr_draw_mode(apply=False)
        self._props.prep.hemisphere.set_draw_active(active)
        self._refresh_lr_status()

    def _on_lr_draw_apply(self) -> None:
        self._prep.exit_lr_draw_mode(apply=True)
        self._props.prep.hemisphere.set_draw_active(False)

    def _on_lr_draw_cancel(self) -> None:
        self._prep.exit_lr_draw_mode(apply=False)
        self._props.prep.hemisphere.set_draw_active(False)
        self._refresh_lr_status()

    def _refresh_lr_status(self) -> None:
        """Push the current PrepView L/R state into the properties panel label."""
        self._props.prep.hemisphere.set_status(self._prep.lr_status_text())

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
        if not self._confirm_discard_active_draft():
            return

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
        # Drain the detected masks into the resident prep-draft store as unsaved
        # edits (kept in RAM, shown yellow until the user saves).
        worker = self._batch_mask_worker
        project = self._state.project
        if worker is not None and project is not None:
            from verso.engine.drafts import PrepDraft
            by_id = {s.id: s for s in project.sections}
            for sid, mask in worker.results.items():
                section = by_id.get(sid)
                if section is None:
                    continue
                self._state.set_prep_draft(
                    sid,
                    PrepDraft(
                        slice_mask=mask,
                        mask_dirty=True,
                        base_flip_h=section.preprocessing.flip_horizontal,
                        base_flip_v=section.preprocessing.flip_vertical,
                    ),
                )
                self._state.mark_dirty(sid, "prep")

        self._overview.refresh()
        if self._current_mode == "prep":
            self._prep.load_section(self._state.current_section)
        self._refresh_properties()
        self._refresh_filmstrip_dots()
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
            position_mm = self._anchoring_position_mm(anchoring)
            section = self._state.current_section
            if section is not None:
                section.alignment.position_mm = position_mm
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
        self._sync_position_mm(project.sections)
        self._panel.update_overlay()
        for i in range(len(project.sections)):
            self._overview.refresh_row(i)
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        self._refresh_filmstrip_dots()

    def _reverse_section_order(self) -> None:
        """Reverse the startup proposal direction before any alignment is stored."""
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

        self._reverse_axis_proposal = not self._reverse_axis_proposal
        self._align.set_reverse_axis(self._reverse_axis_proposal)
        for section in project.sections:
            section.alignment.anchoring = [0.0] * 9
            section.alignment.position_mm = None
            section.alignment.status = AlignmentStatus.NOT_STARTED
            section.alignment.source = None
            section.alignment.stored_anchoring = None
            section.alignment.proposal_anchoring = None
            section.alignment.proposal_confidence = None
            section.alignment.proposal_run_id = None
            section.warp.control_points.clear()
            section.warp.status = AlignmentStatus.NOT_STARTED

        self._initialize_quicknii_anchorings(project.sections)
        self._sync_position_mm(project.sections)

        self._overview.refresh()
        self._on_section_changed(self._state.section_index)
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()

    def _update_reverse_order_enabled(self) -> None:
        project = self._state.project
        if project is None:
            self._act_reverse_proposal.setEnabled(False)
            return
        has_stored_alignment = any(
            section.alignment.status == AlignmentStatus.COMPLETE
            for section in project.sections
        )
        self._act_reverse_proposal.setEnabled(
            self._state.atlas is not None
            and len(project.sections) > 1
            and not has_stored_alignment
        )

    def _update_deepslice_enabled(self, running: bool = False) -> None:
        project = self._state.project
        has_sections = project is not None and bool(project.sections)
        atlas_ready = has_sections and self._state.atlas is not None
        # DeepSlice is trained on coronal sections only.
        is_coronal = project is not None and project.interpolation_axis == "AP"
        self._act_deepslice.setEnabled(atlas_ready and is_coronal and not running)
        if project is not None and not is_coronal:
            self._act_deepslice.setToolTip(
                "DeepSlice supports coronal projects only."
            )
        else:
            self._act_deepslice.setToolTip("")
        self._act_deepslice.setText(
            "DeepSlice running…" if running else "Run &DeepSlice"
        )
        self._act_default_proposal.setEnabled(atlas_ready and not running)
        self._act_clear_all_alignments.setEnabled(atlas_ready and not running)
        # Mask + warp wipes only need a project with sections; atlas not required.
        self._act_clear_all_slice_masks.setEnabled(has_sections and not running)
        self._act_clear_all_lr_masks.setEnabled(has_sections and not running)
        self._act_clear_all_warps.setEnabled(has_sections and not running)

    def _interpolation_axis(self) -> int:
        """Return the QuickNII voxel axis index for the current project."""
        project = self._state.project
        if project is None:
            return 1
        return project.interpolation_axis_index

    def _sync_position_mm(self, sections: list) -> None:
        """Populate position_mm for every section that has a valid anchoring."""
        atlas = self._state.atlas
        if atlas is None:
            return
        for section in sections:
            if section.alignment.anchoring and any(
                v != 0.0 for v in section.alignment.anchoring
            ):
                section.alignment.position_mm = self._anchoring_position_mm(
                    section.alignment.anchoring
                )

    def _anchoring_position_mm(self, anchoring: list[float]) -> float:
        atlas = self._state.atlas
        if atlas is None:
            return 0.0
        center = atlas.cut_center(anchoring)
        return atlas.voxel_to_mm(center[self._interpolation_axis()])

    def _initialize_quicknii_anchorings(self, sections: list) -> None:
        """Initialize empty section planes with QuickNII-compatible stretch."""
        atlas = self._state.atlas
        if atlas is None:
            return

        from verso.engine.io.image_io import registration_dimensions
        from verso.engine.model.alignment import AlignmentStatus
        from verso.engine.registration import quicknii_series_anchorings

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
        propagated = quicknii_series_anchorings(
            image_sizes=[(w, h) for _, w, h in usable],
            serial_numbers=[section.serial_number for section, _, _ in usable],
            atlas_shape=atlas.shape,
            interpolation_axis=self._interpolation_axis(),
            stored_anchorings=display_anchorings,
            reverse_axis=self._reverse_axis_proposal,
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

        from verso.gui.dialogs.bad_sections import BadSectionsDialog

        dlg = BadSectionsDialog(
            project.sections, reverse_order=self._reverse_axis_proposal, parent=self
        )
        if dlg.exec() != BadSectionsDialog.DialogCode.Accepted:
            return
        bad_ids = dlg.bad_section_ids()
        self._reverse_axis_proposal = dlg.reverse_section_order()

        self._update_deepslice_enabled(running=True)
        self.statusBar().showMessage("Running DeepSlice suggestions...")
        self._show_deepslice_progress()

        thread = QThread(self)
        worker = _DeepSliceWorker(
            project,
            reverse_section_order=self._reverse_axis_proposal,
            bad_section_ids=bad_ids,
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
            self._sync_position_mm(project.sections)
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
            interpolation_axis=self._interpolation_axis(),
            reverse_axis=self._reverse_axis_proposal,
        )
        self._sync_position_mm(project.sections)
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
            "then restore VERSO's default proposal for all sections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_discard_active_draft():
            return

        from verso.engine.deepslice import reset_in_progress_to_default_proposals

        changed = reset_in_progress_to_default_proposals(
            project.sections,
            atlas.shape,
            interpolation_axis=self._interpolation_axis(),
            reverse_axis=self._reverse_axis_proposal,
            include_complete=True,
        )
        self._sync_position_mm(project.sections)
        self._after_batch_clear()
        self.statusBar().showMessage(
            f"Cleared all alignments and restored {changed} default proposals",
            5000,
        )

    def _clear_all_slice_masks(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        reply = QMessageBox.question(
            self,
            "Clear all slice masks",
            f"Delete the slice mask for all {len(project.sections)} sections? "
            "This removes the PNG files from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_discard_active_draft():
            return
        removed = 0
        for section in project.sections:
            old = section.preprocessing.slice_mask_path
            if old:
                try:
                    Path(old).unlink(missing_ok=True)
                except OSError:
                    pass
                removed += 1
            section.preprocessing.slice_mask_path = None
        self._after_batch_clear()
        self.statusBar().showMessage(
            f"Cleared {removed} slice masks", 5000
        )

    def _clear_all_lr_masks(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        reply = QMessageBox.question(
            self,
            "Clear all L/R masks",
            f"Delete the L/R hemisphere mask for all {len(project.sections)} sections? "
            "This removes the PNG files from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_discard_active_draft():
            return
        removed = 0
        for section in project.sections:
            old = section.preprocessing.lr_mask_path
            if old:
                try:
                    Path(old).unlink(missing_ok=True)
                except OSError:
                    pass
                removed += 1
            section.preprocessing.lr_mask_path = None
            section.preprocessing.lr_line = None
        self._after_batch_clear()
        self.statusBar().showMessage(
            f"Cleared {removed} L/R masks", 5000
        )

    def _clear_all_warps(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        reply = QMessageBox.question(
            self,
            "Clear all warps",
            f"Remove every warp control point from all {len(project.sections)} sections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_discard_active_draft():
            return
        cleared = 0
        for section in project.sections:
            if section.warp.control_points:
                cleared += 1
            section.warp.control_points.clear()
            section.warp.status = AlignmentStatus.NOT_STARTED
        self._after_batch_clear()
        self.statusBar().showMessage(
            f"Cleared warps on {cleared} sections", 5000
        )

    def _after_batch_clear(self) -> None:
        """Refresh dependent UI + write project after a batch wipe."""
        section = self._state.current_section
        if self._current_mode == "prep" and section is not None:
            self._prep.load_section(section)
        elif self._current_mode in ("align", "warp"):
            self._panel.load_section(section)
        self._overview.refresh()
        self._refresh_properties()
        self._refresh_clear_enabled()
        self._update_ap_plot()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)

    def _update_ap_plot(self) -> None:
        project = self._state.project
        if project is None:
            return
        self._props.align.ap_plot.update_plot(project.sections, self._state.section_index)

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
        if not self._confirm_discard_active_draft():
            return
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
        if not self._confirm_discard_active_draft():
            return
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
        if not self._confirm_discard_active_draft():
            return
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

        if not self._confirm_discard_active_draft():
            return

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
