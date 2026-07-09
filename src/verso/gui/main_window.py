"""Main application window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
)

from verso.engine.atlas import orientation_labels
from verso.engine.model.alignment import AlignmentStatus
from verso.gui import menus, window_builder
from verso.gui.controllers.export_controller import ExportController
from verso.gui.controllers.job_controller import JobController
from verso.gui.controllers.project_controller import ProjectController
from verso.gui.controllers.save_controller import SaveController
from verso.gui.dialogs.brightness import BrightnessDialog
from verso.gui.dialogs.info import show_info_dialog
from verso.gui.state import AppState
from verso.gui.utils import warn_if_missing_dimensions

if TYPE_CHECKING:
    from PyQt6.QtGui import QAction, QShortcut
    from PyQt6.QtWidgets import (
        QDockWidget,
        QLabel,
        QPushButton,
        QStackedWidget,
        QStatusBar,
    )

    from verso.gui.views.align_view import AlignView
    from verso.gui.views.overview_view import OverviewView
    from verso.gui.views.prep_view import PrepView
    from verso.gui.views.warp_view import WarpView
    from verso.gui.widgets.filmstrip import Filmstrip
    from verso.gui.widgets.filmstrip_status import FilmstripStatusPresenter
    from verso.gui.widgets.properties import PropertiesPanel
    from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel


class MainWindow(QMainWindow):
    # Constructed by the window_builder and menus functions (called from
    # __init__ below), which stash the handles back onto ``self``.
    # Here these are declared so type checkers know these attributes exist wont complain
    _stack: QStackedWidget
    _overview: OverviewView
    _prep: PrepView
    _panel: SectionCanvasPanel
    _align: AlignView
    _warp: WarpView
    _props: PropertiesPanel
    _right_dock: QDockWidget
    _filmstrip: Filmstrip
    _bottom_dock: QDockWidget
    _filmstrip_status: FilmstripStatusPresenter
    _statusbar: QStatusBar
    _section_shortcuts: list[QShortcut]
    _view_buttons: list[QPushButton]
    _project_label: QLabel
    _act_reverse_proposal: QAction
    _act_deepslice: QAction
    _act_default_proposal: QAction
    _act_clear_all_alignments: QAction
    _act_clear_all_slice_masks: QAction
    _act_clear_manual_cps: QAction
    _act_clear_auto_cps: QAction
    _act_batch_auto_cp: QAction

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VERSO")
        self.resize(1000, 600)

        self._state = AppState(self)
        self._current_mode = "overview"
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
        self._project = ProjectController(self)
        self._export = ExportController(self)
        self._saves = SaveController(self)
        self._jobs = JobController(self)

        # Build the UI menus and widgets form the window_builder file
        menus.build_menus(self)
        menus.build_toolbar(self)
        window_builder.build_central(self)
        window_builder.build_docks(self)
        window_builder.connect_signals(self)
        window_builder.build_shortcuts(self)
        window_builder.build_status_bar(self)

        self._switch_view(menus.VIEW_OVERVIEW)

    def closeEvent(self, event) -> None:
        if self._jobs.warn_if_busy():
            event.ignore()
            return
        if not self.confirm_discard_active_draft():
            event.ignore()
            return
        # Stop background work before the window is destroyed: the elastix
        # child process, and the filmstrip/atlas loader QThreads.
        self._jobs.shutdown()
        self._filmstrip.shutdown()
        self._state.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _switch_view(self, index: int) -> None:
        modes = ("overview", "prep", "align", "warp")
        entering_mode = modes[index]

        # Release panel hooks from whichever Align/Warp view currently owns it.
        if self._current_mode == "align":
            self._align.deactivate()
        elif self._current_mode == "warp":
            self._warp.deactivate()

        # Make the selected view visible
        self._stack.setCurrentIndex(index)

        self._current_mode = entering_mode

        for i, btn in enumerate(self._view_buttons):
            btn.setChecked(i == index)

        # Show filmstrip outside Overview; enable stored-alignment badges in Align/Warp
        self._bottom_dock.setVisible(index != menus.VIEW_OVERVIEW)
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
                # Refresh the AP plot
                self._update_slicing_position()
            else:
                self._warp.activate()
                # Spawn + warm the elastix child process now
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
    # Per-view draft save / clear / discard
    # ------------------------------------------------------------------

    def active_view(self):
        """The currently visible canvas view (Prep/Align/Warp), or None in Overview."""
        if self._current_mode == "prep":
            return self._prep
        if self._current_mode == "align":
            return self._align
        if self._current_mode == "warp":
            return self._warp
        return None

    def save_active_view(self) -> bool:
        view = self.active_view()
        if view is None:
            return False
        return view.save()

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
            return self._project.save_all()
        if reply == QMessageBox.StandardButton.Discard:
            self._project.discard_all()
            return True
        return False

    def sync_dependent_ui(self, *, write: bool = False, reload_active: bool = False) -> None:
        """Single refresh point after a *discrete* model change.

        Collapses the former six near-identical refresh clusters (F17). Connected
        to ``AppState.sections_changed`` so controllers can request a full
        dependent-UI refresh by emitting rather than poking window internals, and
        called directly by the in-window save/revert paths. The individual
        ``_refresh_*``/``_update_*`` helpers each no-op when there is no project,
        so running the superset is safe and idempotent.

        Not for per-drag ticks — see ``_on_anchoring_changed``, which stays a
        lightweight targeted refresh.

        Args:
            write: Persist ``project.json`` after refreshing.
            reload_active: Reload the current section into the active canvas view
                (needed after a batch op mutates the visible section in place).
        """
        if reload_active:
            section = self._state.current_section
            if self._current_mode == "prep" and section is not None:
                self._prep.load_section(section)
            elif self._current_mode in ("align", "warp"):
                self._panel.load_section(section)
        if self._current_mode in ("align", "warp"):
            self._panel.update_overlay()
        # A batch op may have flipped the proposal direction (ProjectController
        # owns the flag); keep the Align navigator in step. Idempotent.
        self._align.set_reverse_axis(self._project.reverse_axis_proposal)
        self._overview.refresh()
        self._refresh_properties()
        self._update_slicing_position()
        self._refresh_reset_enabled()
        self._refresh_filmstrip_dots()
        self._update_reverse_order_enabled()
        self._update_deepslice_enabled()
        if write and self._state.project is not None and self._state.project_path is not None:
            self._project.write_project(self._state.project_path)

    def after_view_save(self) -> None:
        """Refresh dependent UI after a per-view save/reset and write project."""
        self.sync_dependent_ui(write=True)

    def after_view_revert(self) -> None:
        """Refresh dependent UI after a per-view "Clear edits" revert.

        Reverting only drops unsaved edits, so the on-disk project is already
        the last-saved version — no write is needed.
        """
        self.sync_dependent_ui()

    def _refresh_reset_enabled(self) -> None:
        """Re-sync every save bar (dirty + Reset) to the current section.

        Delegates to SaveController, which reads dirty state from AppState (the
        single source of truth) and each view's ``has_persisted_state``.  Called
        after the active view has re-synced its baseline (section/view change,
        save, revert) so the reads are current.
        """
        self._saves.refresh_all()

    # ------------------------------------------------------------------
    # Filmstrip status dots
    # ------------------------------------------------------------------

    def _refresh_filmstrip_dots(self) -> None:
        """Recompute all filmstrip status dots for the active view's step."""
        self._filmstrip_status.refresh_all(self._current_mode)

    def _refresh_current_step_dot(self) -> None:
        """Refresh the current section's dot when its status changes without a
        dirty flip (e.g. removing the last warp control point → gray)."""
        self._filmstrip_status.refresh_index(self._state.section_index, self._current_mode)

    def _on_dirty_changed(self, section_id: str, step: str) -> None:
        """Incrementally update one filmstrip dot when a section's dirty flips."""
        if step == self._current_mode:
            self._filmstrip_status.refresh_section(section_id, step)

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

        self._project.reverse_axis_proposal = False
        self._align.set_reverse_axis(False)
        self._align.set_interpolation_axis(project.interpolation_axis_index)
        self._props.align.slicing_position.set_axis_name(project.interpolation_axis)

        # Anatomical orientation labels at the canvas edges (Prep + shared
        # Align/Warp canvas), keyed by the project's interpolation axis.
        labels = orientation_labels(project.interpolation_axis)
        self._prep.canvas.set_orientation_labels(labels)
        self._panel.canvas.set_orientation_labels(labels)

        # Series interpolation needs atlas dimensions for the no-anchor and
        # one-anchor endpoint controls. If the atlas is still loading,
        # _on_atlas_loaded performs the exact series propagation.
        if self._state.atlas is not None and warn_if_missing_dimensions(self, project.sections):
            from verso.engine.anchoring import interpolate_anchorings

            interpolate_anchorings(
                project.sections,
                atlas_shape=self._state.atlas.shape,
                interpolation_axis=project.interpolation_axis_index,
                reverse_axis=self._project.reverse_axis_proposal,
            )
        self._project.sync_position_mm(project.sections)

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

        self._switch_view(menus.VIEW_OVERVIEW)

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
                self._project.initialize_default_anchorings(project.sections)
                self._project.sync_position_mm(project.sections)
                self._panel.update_overlay()
                self._update_slicing_position()
                self._update_reverse_order_enabled()
                self._update_deepslice_enabled()

    def _on_atlas_error(self, message: str) -> None:
        self._update_deepslice_enabled()
        QMessageBox.warning(self, "Atlas load failed", message)

    def _on_section_changed(self, index: int) -> None:
        # Unsaved edits persist across the swap without an explicit flush: Prep's
        # mask lives in the draft store's working payload, Align/Warp edits live
        # on the Section.
        section = self._state.current_section
        self._filmstrip.set_current(index)

        if self._current_mode == "prep":
            self._prep.load_section(section)
        elif self._current_mode in ("align", "warp"):
            self._panel.load_section(section)

        # Keep position_mm in lockstep with the (possibly interpolated) anchoring
        # so the AP-plot white dot is correct without requiring a save first.
        if section is not None:
            self._project.sync_position_mm([section])

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
        self._switch_view(menus.VIEW_PREP)
        self._prep.load_section(self._state.current_section)

    def _on_structure_changed(self) -> None:
        """Rebuild list-dependent UI after the section list changes.

        ProjectController mutates + persists the section list (add/remove/reorder)
        and emits ``structure_changed``; here the window rebuilds the widgets that
        track the list itself — the filmstrip tiles and the overview table.
        """
        project = self._state.project
        if project is None:
            return
        self._filmstrip.populate(project.sections, project.channels, project.working_scale)
        self._filmstrip.set_current(self._state.section_index)
        self._overview.refresh()
        self._update_slicing_position()

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

    def _on_flip_changed(self, value: bool, *, horizontal: bool) -> None:
        """Toggle a flip on the current section (Prep view coordination).

        The domain wipe (alignment/warp no longer apply in the new frame) lives
        in ProjectController; here we drive the Prep view and refresh dots.
        """
        section = self._state.current_section
        if section is None:
            return
        prep = section.preprocessing
        current = prep.flip_horizontal if horizontal else prep.flip_vertical
        if value == current:
            return
        flip_widget = self._props.prep.flip
        if not self._project.confirm_flip(section):
            (flip_widget.set_flip_h if horizontal else flip_widget.set_flip_v)(not value)
            return
        if horizontal:
            prep.flip_horizontal = value
        else:
            prep.flip_vertical = value
        self._prep.mark_flip_changed()
        if self._project.invalidate_alignment_for_flip(section):
            self._overview.refresh_row(self._state.section_index)
            self._refresh_filmstrip_dots()
        self._prep.refresh_display()

    def _on_flip_h_changed(self, value: bool) -> None:
        self._on_flip_changed(value, horizontal=True)

    def _on_flip_v_changed(self, value: bool) -> None:
        self._on_flip_changed(value, horizontal=False)

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
            position_mm = self._project.anchoring_position_mm(anchoring)
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
        self._project.initialize_default_anchorings(project.sections)
        self._project.sync_position_mm(project.sections)
        self.sync_dependent_ui()

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

    def _update_slicing_position(self) -> None:
        project = self._state.project
        if project is None:
            return
        dirty_ids = {s.id for s in project.sections if self._state.is_dirty(s.id, "align")}
        self._props.align.slicing_position.update_plot(
            project.sections, self._state.section_index, dirty_ids
        )

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
