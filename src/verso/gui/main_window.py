"""Main application window."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
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
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QToolBar,
    QWidget,
)

from verso.engine.atlas import orientation_labels
from verso.engine.io.quint_io import load_quicknii, load_visualign
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import DEFAULT_PROJECT_FILENAME, Project
from verso.gui.controllers.export_controller import ExportController
from verso.gui.controllers.job_controller import JobController
from verso.gui.controllers.save_controller import SaveController
from verso.gui.dialogs.brightness import BrightnessDialog
from verso.gui.dialogs.info import show_info_dialog
from verso.gui.dialogs.new_project import NewProjectDialog
from verso.gui.state import AppState
from verso.gui.utils import require
from verso.gui.views.align_view import AlignView
from verso.gui.views.overview_view import OverviewView
from verso.gui.views.prep_view import PrepView
from verso.gui.views.warp_view import WarpView
from verso.gui.widgets.filmstrip import Filmstrip
from verso.gui.widgets.properties import PropertiesPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel

if TYPE_CHECKING:
    pass

_VIEW_OVERVIEW = 0
_VIEW_PREP = 1
_VIEW_ALIGN = 2
_VIEW_WARP = 3


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VERSO")
        self.resize(1280, 800)

        self._state = AppState(self)
        self._current_mode = "overview"
        # Proposal direction shared between QuickNII interpolation (here) and the
        # DeepSlice/reverse batch operations (JobController).
        self._reverse_axis_proposal = False
        self._brightness_dialog: BrightnessDialog | None = None

        # Coalesce rapid brightness-slider ticks into one redraw per event-loop
        # pass. Without this the GUI thread spends each tick re-compositing the
        # working-resolution image and the slider visibly lags the mouse.
        self._channels_pending: list | None = None
        self._channels_flush_timer = QTimer(self)
        self._channels_flush_timer.setSingleShot(True)
        self._channels_flush_timer.setInterval(0)
        self._channels_flush_timer.timeout.connect(self._flush_channels_changed)

        # Controllers own self-contained subsystems; they read widgets/state back
        # through this window, so they must exist before the menu wires actions
        # and before _connect_signals registers the views with SaveController.
        self._export = ExportController(self)
        self._saves = SaveController(self)
        self._jobs = JobController(self)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._connect_signals()
        self._build_shortcuts()
        self._build_status_bar()

        self._switch_view(_VIEW_OVERVIEW)

    def closeEvent(self, event) -> None:
        if self._jobs.warn_if_busy():
            event.ignore()
            return
        if not self.confirm_discard_active_draft():
            event.ignore()
            return
        self._jobs.shutdown()
        self._filmstrip.shutdown()
        self._state.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = require(self.menuBar())

        file_menu = require(mb.addMenu("&File"))

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

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        images_menu = require(mb.addMenu("&Image"))
        act_adjust = QAction("&Channels…", self)
        act_adjust.triggered.connect(self._open_brightness_dialog)
        images_menu.addAction(act_adjust)
        act_reorder = QAction("Reorder slices based on &filename…", self)
        act_reorder.triggered.connect(self._reorder_by_filename)
        images_menu.addAction(act_reorder)
        images_menu.addSeparator()
        act_add_images = QAction("&Add images to project…", self)
        act_add_images.triggered.connect(self._add_images_to_project)
        images_menu.addAction(act_add_images)

        batch_menu = require(mb.addMenu("&Batch"))

        preprocess_menu = require(batch_menu.addMenu("&Preprocess"))
        act_batch_mask = QAction("Autodetect slice mask for &all slices", self)
        act_batch_mask.triggered.connect(self._jobs.batch_autodetect_masks)
        preprocess_menu.addAction(act_batch_mask)
        preprocess_menu.addSeparator()
        self._act_clear_all_slice_masks = QAction("Clear all &slice masks…", self)
        self._act_clear_all_slice_masks.setEnabled(False)
        self._act_clear_all_slice_masks.triggered.connect(self._jobs.clear_all_slice_masks)
        preprocess_menu.addAction(self._act_clear_all_slice_masks)

        align_menu = require(batch_menu.addMenu("&Align"))
        self._act_deepslice = QAction("Run &DeepSlice", self)
        self._act_deepslice.setEnabled(False)
        self._act_deepslice.triggered.connect(self._jobs.run_deepslice)
        align_menu.addAction(self._act_deepslice)

        self._act_default_proposal = QAction("&Default proposal", self)
        self._act_default_proposal.setEnabled(False)
        self._act_default_proposal.triggered.connect(self._jobs.revert_to_default_proposal)
        align_menu.addAction(self._act_default_proposal)

        self._act_reverse_proposal = QAction("&Reverse proposal", self)
        self._act_reverse_proposal.setEnabled(False)
        self._act_reverse_proposal.triggered.connect(self._jobs.reverse_section_order)
        align_menu.addAction(self._act_reverse_proposal)

        align_menu.addSeparator()
        self._act_clear_all_alignments = QAction("&Clear all alignments…", self)
        self._act_clear_all_alignments.setEnabled(False)
        self._act_clear_all_alignments.triggered.connect(self._jobs.clear_all_alignments)
        align_menu.addAction(self._act_clear_all_alignments)

        warp_menu = require(batch_menu.addMenu("&Warp"))
        self._act_batch_auto_cp = QAction("&Auto-generate control points for all slices…", self)
        self._act_batch_auto_cp.setEnabled(False)
        self._act_batch_auto_cp.triggered.connect(self._jobs.batch_auto_generate_warps)
        warp_menu.addAction(self._act_batch_auto_cp)
        warp_menu.addSeparator()
        self._act_clear_manual_cps = QAction("Clear all &manual control points…", self)
        self._act_clear_manual_cps.setEnabled(False)
        self._act_clear_manual_cps.triggered.connect(self._jobs.clear_all_manual_cps)
        warp_menu.addAction(self._act_clear_manual_cps)
        self._act_clear_auto_cps = QAction("Clear all a&utomatic control points…", self)
        self._act_clear_auto_cps.setEnabled(False)
        self._act_clear_auto_cps.triggered.connect(self._jobs.clear_all_auto_cps)
        warp_menu.addAction(self._act_clear_auto_cps)

        export_menu = require(mb.addMenu("&Export"))
        act_export_images = QAction("Export images with atlas &overlay…", self)
        act_export_images.triggered.connect(self._export.export_images_with_overlay)
        export_menu.addAction(act_export_images)

        act_export_stack = QAction("Export aligned section &stack…", self)
        act_export_stack.triggered.connect(self._export.export_aligned_stack)
        export_menu.addAction(act_export_stack)

        export_menu.addSeparator()

        act_export_qn_xml = QAction("Export QuickNII &XML…", self)
        act_export_qn_xml.triggered.connect(self._export.export_quicknii_xml)
        export_menu.addAction(act_export_qn_xml)

        act_export_qn = QAction("Export &QuickNII JSON…", self)
        act_export_qn.triggered.connect(self._export.export_quicknii)
        export_menu.addAction(act_export_qn)

        act_export_va = QAction("Export &VisuAlign JSON…", self)
        act_export_va.triggered.connect(self._export.export_visualign)
        export_menu.addAction(act_export_va)

        help_menu = require(mb.addMenu("&Help"))
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
            self._brightness_dialog.channels_committed.connect(self._on_channels_committed)
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

        self._stack.addWidget(self._overview)  # 0
        self._stack.addWidget(self._prep)  # 1
        self._stack.addWidget(self._align)  # 2
        self._stack.addWidget(self._warp)  # 3

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
        right_dock.setTitleBarWidget(QWidget())  # hide title bar
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

    def _build_status_bar(self) -> None:
        """Create the status bar up front and make it a touch shorter.

        QMainWindow builds the status bar lazily, on the first ``showMessage``
        call — so without this it appears to pop in below the filmstrip the
        first time the user saves, shifting the whole layout.  Instantiating it
        here reserves the space from launch so it is always visible.  The font
        is shrunk a point and the size grip dropped to keep it compact.
        """
        bar = require(self.statusBar())
        self._statusbar = bar
        bar.setSizeGripEnabled(False)
        font = bar.font()
        font.setPointSizeF(max(1.0, font.pointSizeF() - 1.0))
        bar.setFont(font)
        bar.setMaximumHeight(bar.fontMetrics().height() + 4)

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
        self._overview.sections_reordered.connect(self._on_sections_reordered)
        self._overview.remove_requested.connect(self._remove_sections)

        # Filmstrip
        self._filmstrip.section_selected.connect(self._state.set_section)
        self._filmstrip.thumbnail_loaded.connect(self._on_thumbnail_loaded)

        # Properties
        flip = self._props.prep.flip
        flip.flip_h_changed.connect(self._on_flip_h_changed)
        flip.flip_v_changed.connect(self._on_flip_v_changed)
        mask = self._props.prep.mask_box
        mask.visibility_changed.connect(self._prep.set_mask_visible)
        mask.opacity_changed.connect(self._prep.set_mask_opacity)
        mask.color_changed.connect(self._prep.set_mask_color)
        mask.negative_changed.connect(self._prep.set_mask_negative)
        mask.draw_mode_changed.connect(self._prep.set_draw_mode)
        self._prep.draw_mode_changed.connect(mask.set_draw_mode)
        mask.brush_size_changed.connect(self._prep.set_brush_size)
        self._prep.brush_size_changed.connect(mask.set_brush_size)
        mask.autodetect_requested.connect(self._on_prep_autodetect_requested)
        mask.clear_requested.connect(self._on_prep_clear_mask_requested)
        mask.erode_requested.connect(lambda px: self._prep.apply_morph(px, "erode"))
        mask.expand_requested.connect(lambda px: self._prep.apply_morph(px, "expand"))
        # Overlay lives in both Align and Warp pages with independent state.
        for overlay in (self._props.align.overlay, self._props.warp.overlay):
            overlay.opacity_changed.connect(self._on_opacity_changed)
            overlay.color_changed.connect(self._panel.set_outline_color)
            overlay.mode_changed.connect(self._panel.set_overlay_mode)

        # PrepView edits
        self._prep.mask_negative_changed.connect(mask.set_negative)
        self._prep.mask_visibility_changed.connect(mask.set_visible_state)

        # AlignView navigator drives the anchoring; alignments_updated fires
        # when the user explicitly saves or clears, triggering re-interpolation.
        self._align.anchoring_changed.connect(self._on_anchoring_changed)
        self._align.alignments_updated.connect(self._on_alignments_updated)
        self._props.warp.cp.style_changed.connect(self._on_cp_style_changed)
        self._props.warp.cp.autogen_requested.connect(self._jobs.auto_generate_warp_cps)
        self._props.warp.cp.edit_params_requested.connect(self._jobs.edit_elastix_params)

        # Local-changes bars (Save / Clear edits / Reset) and per-view dirty signals.
        # SaveController owns the parameterized save/revert/clear dispatch.
        view_bindings = (
            ("prep", self._prep, self._props.prep),
            ("align", self._align, self._props.align),
            ("warp", self._warp, self._props.warp),
        )
        for step, view, page in view_bindings:
            self._saves.register(step, view, page)
            view.dirty_changed.connect(page.save_bar.set_dirty)
            # Mirror the view's dirty state into the persistent edit registry for
            # the section currently loaded in that view.
            view.dirty_changed.connect(lambda dirty, s=step: self._on_view_dirty_changed(s, dirty))
            page.save_bar.save_requested.connect(lambda s=step: self._saves.on_save(s))
            page.save_bar.revert_requested.connect(lambda s=step: self._saves.on_revert(s))
            page.save_bar.reset_requested.connect(lambda s=step: self._saves.on_clear(s))

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
                # The AP plot is no longer refreshed on every section change
                # (only while Align is visible), so refresh it on entry to pick
                # up any section change made while Align was hidden.
                self._update_slicing_position()
            else:
                self._warp.activate()
                # Spawn + warm the elastix child process now so the first
                # "Auto-generate" click hits a warm optimizer instead of paying
                # the ~15 s cold-start cost interactively.
                self._jobs.warm_elastix_if_supported()
            # Push the working scale before load_section so any thumbnail
            # regeneration uses the project's scale, not the panel default.
            if project is not None:
                self._panel.set_working_scale(project.working_scale)
            if self._panel.section is not section:
                self._panel.load_section(section)
            if project is not None:
                self._panel.set_channels(project.channels)

        # Refresh properties with current section
        self._props.set_mode(self._current_mode)
        self._refresh_properties()
        self._refresh_reset_enabled()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        self._refresh_filmstrip_dots()

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def _open_project(self) -> None:
        if not self.confirm_discard_active_draft():
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
        if not self.confirm_discard_active_draft():
            return
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            project = dlg.result_project()
            if project is not None:
                self._state.load_project(project, dlg.result_project_path())

    def _open_quicknii(self) -> None:
        if not self.confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QuickNII JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            project = load_quicknii(Path(path))
            self._state.load_project(project)

    def _open_visualign(self) -> None:
        if not self.confirm_discard_active_draft():
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
        self._filmstrip.populate(project.sections, project.channels, project.working_scale)
        self._props.warp.cp.apply_style(project.cp_size, project.cp_shape, project.cp_color)
        self._warp.set_cp_style(project.cp_size, project.cp_shape, project.cp_color)

        if self._state.project_path is not None:
            self._write_project(self._state.project_path)

        self._statusbar.showMessage(f"Imported settings from {Path(path).name}", 3000)

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
        #    Save unconditionally (not gated on is_dirty) so Ctrl+S matches the
        #    per-view Save button: an untouched alignment still gets its default
        #    plane committed instead of being silently skipped.
        active = self._active_view()
        if active is not None:
            active.save()

        # 2. Persist every remaining dirty (section, step).  Snapshot the list up
        #    front since we mutate the registry inside the loop.
        for section, steps in self._state.dirty_sections():
            if "prep" in steps:
                draft = self._state.pop_prep_draft(section.id)
                if draft is not None:
                    # Flip invalidation already happened at toggle time, so this
                    # only writes masks — it won't clobber an alignment the user
                    # redid after flipping.
                    persist_prep_draft(section, draft)
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
        self._update_slicing_position()
        self._refresh_reset_enabled()
        self._refresh_filmstrip_dots()
        return True

    def _save_project_as(self) -> None:
        self._save_active_view()
        if self._state.project is None:
            return
        current_path = self._state.project_path
        suggested = str(current_path) if current_path is not None else DEFAULT_PROJECT_FILENAME
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", suggested, "JSON files (*.json)"
        )
        if path:
            project_path = Path(path)
            if project_path.suffix == "":
                project_path = project_path.with_suffix(".json")
            self._write_project(project_path)
            self._state.set_project_path(project_path)
            self._refresh_reset_enabled()

    def _write_project(self, path: Path) -> None:
        project = self._state.project
        if project is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            project.save(path)
            self._statusbar.showMessage(f"Saved project to {path}", 3000)
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
        """A prep Clear/Reset wiped the current section's alignment + warp."""
        section = self._state.current_section
        if section is None:
            return
        self._clear_alignment_view_state(section)
        self._seed_alignment_to_quicknii_default(section)

    def _clear_alignment_view_state(self, section) -> None:
        """Drop registry dirty + stashed baselines for a section whose alignment
        was just wiped, so Align/Warp re-sync to the cleared state on activate."""
        self._state.clear_dirty(section.id, "align")
        self._state.clear_dirty(section.id, "warp")
        self._state.pop_baseline(section.id, "align")
        self._state.pop_baseline(section.id, "warp")

    def _seed_alignment_to_quicknii_default(self, section) -> None:
        """Re-seed a wiped section with the QuickNII interpolated proposal.

        After a flip or prep reset the anchoring is all-zeros. This produces the
        same result as clicking the Align "Reset" button: re-running the QuickNII
        series interpolation so the section gets the best available positional
        guess based on its neighbours. Without a non-zero anchoring every canvas
        drag handler bails out silently.
        """
        project = self._state.project
        if project is None or self._state.atlas is None:
            return
        self._initialize_quicknii_anchorings(project.sections)
        self._sync_position_mm([section])

    def _invalidate_alignment_for_flip(self, section) -> None:
        """Wipe a section's alignment + warp the instant its flip is toggled.

        A horizontal/vertical flip changes the image coordinate frame, so any
        existing registration no longer applies.  Doing this at toggle time (not
        at save time) means a re-alignment performed in the new orientation is
        preserved through the next save instead of being wiped by it.
        """
        from verso.engine.drafts import wipe_alignment_for_flip

        has_alignment = (
            section.alignment.status != AlignmentStatus.NOT_STARTED
            or bool(section.warp.control_points)
            or (section.alignment.anchoring and any(v != 0.0 for v in section.alignment.anchoring))
        )
        if not has_alignment:
            return
        wipe_alignment_for_flip(section)
        self._clear_alignment_view_state(section)
        self._seed_alignment_to_quicknii_default(section)
        self._overview.refresh_row(self._state.section_index)
        self._refresh_filmstrip_dots()

    def confirm_discard_active_draft(self) -> bool:
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
            f"You have unsaved edits in {n} section(s). Save them before continuing?",
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

    def after_view_save(self) -> None:
        """Refresh dependent UI after a per-view save/reset and write project."""
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)
        self._overview.refresh()
        self._update_slicing_position()
        self._refresh_reset_enabled()
        self._refresh_filmstrip_dots()

    def after_view_revert(self) -> None:
        """Refresh dependent UI after a per-view "Clear edits" revert.

        Reverting only drops unsaved edits, so the on-disk project is already
        the last-saved version — no write is needed.
        """
        self._overview.refresh()
        self._update_slicing_position()
        self._refresh_reset_enabled()
        self._refresh_filmstrip_dots()

    def _refresh_reset_enabled(self) -> None:
        """Sync each bar's Reset button to whether the slice has persisted state."""
        self._props.prep.save_bar.set_reset_enabled(self._prep.has_persisted_state())
        self._props.align.save_bar.set_reset_enabled(self._align.has_persisted_state())
        self._props.warp.save_bar.set_reset_enabled(self._warp.has_persisted_state())

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

        color = section_step_color(section, step, dirty=self._state.is_dirty(section.id, step))
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
            self._prep.canvas.set_orientation_labels(None)
            self._panel.canvas.set_orientation_labels(None)
            return

        self._set_project_views_enabled(True)

        self._reverse_axis_proposal = False
        self._align.set_reverse_axis(False)
        self._align.set_interpolation_axis(project.interpolation_axis_index)
        self._props.align.slicing_position.set_axis_name(project.interpolation_axis)

        # Anatomical orientation labels at the canvas edges (Prep + shared
        # Align/Warp canvas), keyed by the project's interpolation axis.
        labels = orientation_labels(project.interpolation_axis)
        self._prep.canvas.set_orientation_labels(labels)
        self._panel.canvas.set_orientation_labels(labels)

        # QuickNII interpolation needs atlas dimensions for the no-anchor and
        # one-anchor endpoint controls. If the atlas is still loading,
        # _on_atlas_loaded performs the exact QuickNII propagation.
        if self._state.atlas is not None:
            from verso.engine.anchoring import interpolate_anchorings

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
        self._filmstrip.populate(project.sections, project.channels, project.working_scale)
        self._prep.set_channels(project.channels)
        self._panel.set_channels(project.channels)
        self._panel.set_working_scale(project.working_scale)
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
                self._update_slicing_position()
                self._update_reverse_order_enabled()
                self._update_deepslice_enabled()

    def _on_atlas_error(self, message: str) -> None:
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

        self._refresh_properties()
        self._refresh_reset_enabled()
        self._update_deepslice_enabled()
        # The overview table is rebuilt on entry to Overview (see _switch_view),
        # and its contents track edits/saves, not which section is current, so a
        # selection change needs no rebuild here.  The AP plot lives only in the
        # Align panel, so refresh it only when that view is visible (entering
        # Align refreshes it too — see _switch_view).
        if self._current_mode == "align":
            self._update_slicing_position()

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

    def _reorder_by_filename(self) -> None:
        """Re-derive every slice index from the image filenames.

        Runs the same heuristic used to seed indices at import
        (:func:`guess_slice_indices`) over the current sections, overwriting any
        manual edits, then re-sorts and recomputes like an overview edit.
        """
        from verso.engine.io.image_io import guess_slice_indices

        project = self._state.project
        if project is None or not project.sections:
            return

        resp = QMessageBox.question(
            self,
            "Reorder slices based on filename",
            "Re-derive every slice index from the image filenames?\n\n"
            "This overwrites any manual slice-index edits.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if resp != QMessageBox.StandardButton.Ok:
            return

        indices = guess_slice_indices([s.original_path for s in project.sections])
        keep_id = (
            self._state.current_section.id if self._state.current_section is not None else None
        )
        for section, index in zip(project.sections, indices):
            section.slice_index = index
        project.sort_sections()

        if keep_id is not None:
            new_pos = next((i for i, s in enumerate(project.sections) if s.id == keep_id), None)
            if new_pos is not None:
                self._state.set_section(new_pos)

        self._on_sections_reordered()

    def _on_sections_reordered(self) -> None:
        """A slice index was edited in Overview: re-interpolate, refresh, persist.

        The Overview view has already mutated ``slice_index``, re-sorted the
        sections, and updated the selection; here we recompute everything that
        depends on the section order and save the project to disk.
        """
        project = self._state.project
        if project is None:
            return

        if self._state.atlas is not None:
            from verso.engine.anchoring import interpolate_anchorings

            interpolate_anchorings(
                project.sections,
                atlas_shape=self._state.atlas.shape,
                interpolation_axis=project.interpolation_axis_index,
                reverse_axis=self._reverse_axis_proposal,
            )
        self._sync_position_mm(project.sections)

        self._filmstrip.populate(project.sections, project.channels, project.working_scale)
        self._filmstrip.set_current(self._state.section_index)
        self._overview.refresh()
        self._update_slicing_position()

        if self._state.project_path is not None:
            self._write_project(self._state.project_path)

    def _add_images_to_project(self) -> None:
        """Add new section images to the current project (Image menu).

        New images are appended after the current series with provisional slice
        indices (``max + 1``…); the user corrects them in the Overview table.
        ``working_scale`` is never recomputed — new thumbnails are generated at
        the project's existing scale so all working-resolution geometry stays valid.
        """
        from verso.engine.sections import make_added_sections
        from verso.gui.dialogs.new_project import _IMAGE_FILTER, generate_thumbnails

        if self._state.project is None:
            QMessageBox.information(self, "No project", "Open or create a project first.")
            return
        if self._state.project_path is None:
            QMessageBox.information(
                self,
                "Save project first",
                "Save the project before adding images so the new thumbnails have a home on disk.",
            )
            return
        if not self.confirm_discard_active_draft():
            return

        # Re-fetch after the confirm gate: a "Discard" reloads the project object.
        project = self._state.project
        project_path = self._state.project_path
        if project is None or project_path is None:
            return

        paths, _ = QFileDialog.getOpenFileNames(self, "Add Section Images", "", _IMAGE_FILTER)
        if not paths:
            return

        thumbnails_dir = project_path.parent / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        new_sections, skipped = make_added_sections(project.sections, paths, thumbnails_dir)

        if skipped:
            names = "\n".join(f"  • {Path(p).name}" for p in skipped)
            QMessageBox.warning(
                self,
                "Some images skipped",
                f"{len(skipped)} image(s) were skipped because they are already in "
                f"the project or share a filename with an existing image:\n\n{names}",
            )
        if not new_sections:
            return

        self._warn_channel_mismatch(new_sections, project)

        keep = self._state.current_section
        keep_id = keep.id if keep is not None else None

        project.sections.extend(new_sections)
        project.sort_sections()
        generate_thumbnails(new_sections, project.working_scale, self, title="Add images")

        if keep_id is not None:
            pos = next((i for i, s in enumerate(project.sections) if s.id == keep_id), None)
            if pos is not None:
                self._state.set_section(pos)
        self._on_sections_reordered()
        self._statusbar.showMessage(f"Added {len(new_sections)} image(s) to the project", 3000)

    def _warn_channel_mismatch(self, new_sections: list, project) -> None:
        """Warn (do not block) if added images differ in channel count."""
        from verso.engine.io.image_io import probe_channels

        expected = len(project.channels)
        if expected == 0:
            return
        mismatched: list[str] = []
        for s in new_sections:
            try:
                n = len(probe_channels(s.original_path))
            except Exception:
                continue
            if n != expected:
                mismatched.append(f"  • {Path(s.original_path).name}: {n} channel(s)")
        if mismatched:
            lines = "\n".join(mismatched)
            QMessageBox.warning(
                self,
                "Channel count differs",
                f"The project expects {expected} channel(s), but some added images "
                f"differ:\n\n{lines}\n\nThey may not display correctly.",
            )

    def _remove_sections(self, section_ids: list[str]) -> None:
        """Remove sections from the project (Overview context menu).

        Surviving ``slice_index`` values are left untouched. Each removed
        section's generated thumbnail and masks are deleted (guarded against
        files still referenced by a surviving section); originals are kept.
        """
        from verso.engine.sections import removed_section_artifacts

        if self._state.project is None or not section_ids:
            return

        ids = set(section_ids)
        surviving = [s for s in self._state.project.sections if s.id not in ids]
        to_remove = [s for s in self._state.project.sections if s.id in ids]
        if not to_remove:
            return
        if not surviving:
            QMessageBox.information(
                self,
                "Cannot remove",
                "A project must keep at least one image. Removing these would empty it.",
            )
            return

        n = len(to_remove)
        resp = QMessageBox.question(
            self,
            "Remove from project",
            f"Remove {n} image{'s' if n != 1 else ''} from the project?\n\n"
            "Their generated thumbnails and masks will be deleted. The original "
            "image files are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        if not self.confirm_discard_active_draft():
            return

        # Re-fetch after the confirm gate: a "Discard" reloads the project object.
        project = self._state.project
        if project is None:
            return
        surviving = [s for s in project.sections if s.id not in ids]
        to_remove = [s for s in project.sections if s.id in ids]
        if not to_remove or not surviving:
            return

        keep = self._state.current_section
        keep_id = keep.id if keep is not None else None
        old_index = self._state.section_index

        for section in to_remove:
            for artifact in removed_section_artifacts(section, surviving):
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    pass
            self._state.forget_section(section.id)

        project.sections = surviving

        if keep_id is not None and any(s.id == keep_id for s in project.sections):
            pos = next(i for i, s in enumerate(project.sections) if s.id == keep_id)
        else:
            pos = min(old_index, len(project.sections) - 1)
        self._state.set_section(pos)

        self._on_sections_reordered()
        # If the index is unchanged but now points at a different section, force a
        # reload of the active view and properties.
        if pos == old_index:
            self._state.section_changed.emit(pos)
        self._statusbar.showMessage(
            f"Removed {n} image{'s' if n != 1 else ''} from the project", 3000
        )

    def _refresh_properties(self) -> None:
        self._props.update_section(self._state.current_section, self._current_mode)
        if self._current_mode == "overview":
            # Reuse the filmstrip's already-loaded tile — no extra I/O.
            self._props.overview.set_preview(
                self._filmstrip.thumbnail_pixmap(self._state.section_index)
            )

    def _on_thumbnail_loaded(self, index: int) -> None:
        """Fill the Overview preview once the current section's tile arrives."""
        if self._current_mode == "overview" and index == self._state.section_index:
            self._props.overview.set_preview(self._filmstrip.thumbnail_pixmap(index))

    # ------------------------------------------------------------------
    # Property change slots
    # ------------------------------------------------------------------

    def _on_flip_h_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_horizontal:
            return
        if not self._confirm_flip(section):
            self._props.prep.flip.set_flip_h(not value)
            return
        section.preprocessing.flip_horizontal = value
        self._prep.mark_flip_changed()
        self._invalidate_alignment_for_flip(section)
        self._prep.refresh_display()

    def _on_flip_v_changed(self, value: bool) -> None:
        section = self._state.current_section
        if section is None:
            return
        if value == section.preprocessing.flip_vertical:
            return
        if not self._confirm_flip(section):
            self._props.prep.flip.set_flip_v(not value)
            return
        section.preprocessing.flip_vertical = value
        self._prep.mark_flip_changed()
        self._invalidate_alignment_for_flip(section)
        self._prep.refresh_display()

    def _confirm_flip(self, section) -> bool:
        """Return True when the flip may proceed.

        Shows a warning dialog when the section has an existing alignment and
        ``dialog_prefs.show_align_deletion`` is True.  If the user ticks
        "Do not show again", the flag is persisted to the project.
        """
        from verso.engine.model.alignment import AlignmentStatus

        has_alignment = (
            section.alignment.status != AlignmentStatus.NOT_STARTED
            or bool(section.warp.control_points)
            or (section.alignment.anchoring and any(v != 0.0 for v in section.alignment.anchoring))
        )
        if not has_alignment:
            return True

        project = self._state.project
        if project is None or not project.dialog_prefs.show_align_deletion:
            return True

        from verso.gui.dialogs.flip_warning import confirm_flip_deletes_alignment

        confirmed, suppress = confirm_flip_deletes_alignment(self)
        if confirmed and suppress:
            project.dialog_prefs.show_align_deletion = False
        return confirmed

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
        self._update_slicing_position()
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
        self._update_slicing_position()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        self._refresh_filmstrip_dots()

    def _update_reverse_order_enabled(self) -> None:
        project = self._state.project
        if project is None:
            self._act_reverse_proposal.setEnabled(False)
            return
        has_stored_alignment = any(
            section.alignment.status == AlignmentStatus.COMPLETE for section in project.sections
        )
        self._act_reverse_proposal.setEnabled(
            self._state.atlas is not None and len(project.sections) > 1 and not has_stored_alignment
        )

    def _update_deepslice_enabled(self, running: bool = False) -> None:
        project = self._state.project
        has_sections = project is not None and bool(project.sections)
        atlas_ready = has_sections and self._state.atlas is not None
        # DeepSlice is trained on coronal sections only.
        is_coronal = project is not None and project.interpolation_axis == "AP"
        self._act_deepslice.setEnabled(atlas_ready and is_coronal and not running)
        if project is not None and not is_coronal:
            self._act_deepslice.setToolTip("DeepSlice supports coronal projects only.")
        else:
            self._act_deepslice.setToolTip("")
        self._act_deepslice.setText("DeepSlice running…" if running else "Run &DeepSlice")
        self._act_default_proposal.setEnabled(atlas_ready and not running)
        self._act_clear_all_alignments.setEnabled(atlas_ready and not running)
        # Mask + warp wipes only need a project with sections; atlas not required.
        self._act_clear_all_slice_masks.setEnabled(has_sections and not running)
        # Clearing manual / automatic control points is only offered when points
        # of that kind actually exist somewhere in the project.
        sections = project.sections if project is not None else []
        has_manual_cps = any(cp for s in sections for cp in s.warp.control_points if not cp.auto)
        has_auto_cps = any(cp for s in sections for cp in s.warp.control_points if cp.auto)
        self._act_clear_manual_cps.setEnabled(has_manual_cps and not running)
        self._act_clear_auto_cps.setEnabled(has_auto_cps and not running)
        # Automatic control points need the atlas loaded and an Allen mouse atlas
        # (the curated anchor points were traced in Allen CCF space).
        auto_cp_ok = atlas_ready and not running and self._jobs.is_auto_cp_atlas()
        auto_cp_busy = self._jobs.auto_cp_busy
        self._act_batch_auto_cp.setEnabled(auto_cp_ok and not auto_cp_busy)
        self._props.warp.cp.set_autogen_enabled(auto_cp_ok and not auto_cp_busy)

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
            if section.alignment.anchoring and any(v != 0.0 for v in section.alignment.anchoring):
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

        from verso.engine.anchoring import quicknii_series_anchorings
        from verso.engine.io.image_io import registration_dimensions
        from verso.engine.model.alignment import AlignmentStatus

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

        from verso.engine.anchoring import _display_space_anchoring

        display_anchorings = []
        for section, _, _ in usable:
            is_complete = section.alignment.status == AlignmentStatus.COMPLETE
            display_anchorings.append(_display_space_anchoring(section) if is_complete else None)
        propagated = quicknii_series_anchorings(
            image_sizes=[(w, h) for _, w, h in usable],
            slice_indices=[section.slice_index for section, _, _ in usable],
            atlas_shape=atlas.shape,
            interpolation_axis=self._interpolation_axis(),
            stored_anchorings=display_anchorings,
            reverse_axis=self._reverse_axis_proposal,
            center_proposals=True,
        )

        stored_indices = {
            section.slice_index
            for (section, _, _), anch in zip(usable, display_anchorings)
            if anch is not None
        }

        for (section, _, _), anchoring, anch in zip(usable, propagated, display_anchorings):
            if anch is not None:
                continue
            # Always sync sections that share a slice index with a stored
            # section — they are the same physical slice and must show the same
            # image.
            is_index_duplicate = section.slice_index in stored_indices
            has_existing = section.alignment.anchoring and any(
                v != 0.0 for v in section.alignment.anchoring
            )
            if (
                not is_index_duplicate
                and has_existing
                and section.alignment.status != AlignmentStatus.NOT_STARTED
                and section.alignment.source != "quicknii_default"
            ):
                continue
            section.alignment.anchoring = anchoring
            if section.alignment.status == AlignmentStatus.NOT_STARTED:
                section.alignment.status = AlignmentStatus.IN_PROGRESS
            section.alignment.source = "quicknii_default"

    def _after_batch_clear(self) -> None:
        """Refresh dependent UI + write project after a batch wipe."""
        section = self._state.current_section
        if self._current_mode == "prep" and section is not None:
            self._prep.load_section(section)
        elif self._current_mode in ("align", "warp"):
            self._panel.load_section(section)
        self._overview.refresh()
        self._refresh_properties()
        self._refresh_reset_enabled()
        self._update_slicing_position()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        if self._state.project is not None and self._state.project_path is not None:
            self._write_project(self._state.project_path)

    def _update_slicing_position(self) -> None:
        project = self._state.project
        if project is None:
            return
        self._props.align.slicing_position.update_plot(project.sections, self._state.section_index)

    def _show_atlas_info(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            project = self._state.project
            name = project.atlas.name if (project and project.atlas) else "(none)"
            show_info_dialog(self, "Atlas info", [("Name", name), ("Status", "not yet loaded")])
            return

        ap, dv, lr = atlas._annotation.shape
        n_regions = len(atlas._color_dict) - 1  # exclude background (id 0)
        show_info_dialog(
            self,
            "Atlas info",
            [
                ("Name", atlas.atlas_name),
                ("Resolution", f"{atlas.resolution_um:.1f} µm"),
                ("Volume shape", f"AP {ap}  ×  DV {dv}  ×  LR {lr}"),
                ("Brain regions", str(n_regions)),
            ],
        )

    def _show_project_info(self) -> None:
        project = self._state.project
        if project is None:
            show_info_dialog(self, "Project info", [("Status", "No project loaded")])
            return

        path_str = str(self._state.project_path) if self._state.project_path else "(not saved)"
        atlas_name = project.atlas.name if project.atlas else "(none)"
        channels = ", ".join(ch.name for ch in project.channels) if project.channels else "(none)"
        show_info_dialog(
            self,
            "Project info",
            [
                ("Name", project.name),
                ("File", path_str),
                ("Atlas", atlas_name),
                ("Sections", str(len(project.sections))),
                ("Channels", channels),
                ("Thumbnail scale", f"{project.working_scale:.2f}×"),
            ],
        )
