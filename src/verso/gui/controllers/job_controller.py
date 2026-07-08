"""Owns the background-job lifecycle and the batch project operations.

The job *workers* live in :mod:`verso.gui.jobs`; this controller owns their
orchestration — DeepSlice, batch slice-mask detection, and automatic (elastix)
control-point generation — plus the batch clear/reverse operations and the warm
elastix child process. It holds the job references and exposes :meth:`is_busy`
and :meth:`shutdown` for the window's close handling.

Dependent-UI refreshes stay on the window (its coordinator role); this
controller calls back into them through ``self._window``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QDialog, QMessageBox

from verso.engine.elastix import ElastixWorker
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.elastix import ElastixParams
from verso.gui.jobs import AutoCPWorker, BackgroundJob, BatchMaskWorker, DeepSliceWorker
from verso.gui.utils import warn_errors, warn_if_missing_dimensions

if TYPE_CHECKING:
    from verso.gui.main_window import MainWindow


class JobController:
    """Runs DeepSlice / batch-mask / auto-CP jobs and the batch clear operations."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        self._deepslice_job: BackgroundJob[DeepSliceWorker] | None = None
        self._batch_mask_job: BackgroundJob[BatchMaskWorker] | None = None
        self._auto_cp_job: BackgroundJob[AutoCPWorker] | None = None
        self._auto_cp_batch = False
        # Persistent warm child process for elastix registrations (see ElastixWorker).
        self._elastix_worker = ElastixWorker()

    # ------------------------------------------------------------------
    # Lifecycle / close handling
    # ------------------------------------------------------------------

    @property
    def auto_cp_busy(self) -> bool:
        return self._auto_cp_job is not None

    def warn_if_busy(self) -> bool:
        """Show an info box and return True if a job blocks closing the window."""
        if self._deepslice_job and self._deepslice_job.is_running():
            QMessageBox.information(
                self._window,
                "DeepSlice is running",
                "Wait for the DeepSlice proposal run to finish before closing VERSO.",
            )
            return True
        if self._batch_mask_job and self._batch_mask_job.is_running():
            QMessageBox.information(
                self._window,
                "Batch masks are running",
                "Wait for batch mask auto-detection to finish before closing VERSO.",
            )
            return True
        return False

    def warm_elastix_if_supported(self) -> None:
        """Spawn + warm the elastix child process so the first run isn't cold."""
        if self.is_auto_cp_atlas():
            self._elastix_worker.start()

    def shutdown(self) -> None:
        self._elastix_worker.shutdown()

    # ------------------------------------------------------------------
    # Batch slice-mask detection
    # ------------------------------------------------------------------

    def batch_autodetect_masks(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        if self._batch_mask_job and self._batch_mask_job.is_running():
            return

        reply = QMessageBox.question(
            self._window,
            "Auto-detect slice masks",
            "Run slice-mask auto-detection for all sections? "
            "Existing slice masks will be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
            return

        self._batch_mask_job = BackgroundJob(
            self._window,
            BatchMaskWorker(list(project.sections), project.working_scale),
            title="Batch masks",
            message="Auto-detecting slice masks...",
            min_width=300,
        )
        self._batch_mask_job.start(self._on_batch_masks_done, self._on_batch_masks_finished)

    def _on_batch_masks_done(self, completed: int, errors: list[str]) -> None:
        # Drain the detected masks into the resident prep-draft store as unsaved
        # edits (kept in RAM, shown yellow until the user saves).
        worker = self._batch_mask_job.worker if self._batch_mask_job is not None else None
        project = self._state.project
        if worker is not None and project is not None:
            import copy

            by_id = {s.id: s for s in project.sections}
            for sid, mask in worker.results.items():
                section = by_id.get(sid)
                if section is None:
                    continue
                # The detected mask is an unsaved edit: park it in the draft
                # store's "prep" working payload (kept in RAM, shown yellow until
                # the user saves).
                self._state.set_working(sid, "prep", mask)
                # Stash the last-saved baseline before dirtying so "Clear edits"
                # can revert — the mask edit lives in the working payload, so the
                # section's current preprocessing is still the last-saved state.
                # Mirrors the DeepSlice flow; sync_baseline on load is a no-op
                # while dirty.
                self._state.set_baseline(sid, "prep", copy.deepcopy(section.preprocessing))
                self._state.mark_dirty(sid, "prep")

        self._window._overview.refresh()
        if self._window._current_mode == "prep":
            self._window._prep.load_section(self._state.current_section)
        self._window._refresh_properties()
        self._window._refresh_filmstrip_dots()
        self._window._statusbar.showMessage(f"Auto-detected {completed} slice masks", 5000)
        warn_errors(
            self._window,
            "Some masks failed",
            errors,
            f"{len(errors)} sections could not be processed:",
        )

    def _on_batch_masks_finished(self) -> None:
        self._batch_mask_job = None

    # ------------------------------------------------------------------
    # Reverse proposal
    # ------------------------------------------------------------------

    def reverse_section_order(self) -> None:
        """Reverse the startup proposal direction before any alignment is stored."""
        project = self._state.project
        if project is None or len(project.sections) < 2:
            return

        has_stored_alignment = any(
            section.alignment.status == AlignmentStatus.COMPLETE for section in project.sections
        )
        if has_stored_alignment:
            QMessageBox.information(
                self._window,
                "Cannot reverse proposal",
                "The startup proposal can only be reversed before any alignment is stored.",
            )
            return

        from verso.engine.drafts import reset_alignment

        self._window._reverse_axis_proposal = not self._window._reverse_axis_proposal
        self._window._align.set_reverse_axis(self._window._reverse_axis_proposal)
        for section in project.sections:
            reset_alignment(section)

        self._window._initialize_quicknii_anchorings(project.sections)
        self._window._sync_position_mm(project.sections)

        self._window._overview.refresh()
        self._window._on_section_changed(self._state.section_index)
        self._window._update_slicing_position()
        self._window._update_reverse_order_enabled()
        self._window._update_deepslice_enabled()

    # ------------------------------------------------------------------
    # DeepSlice
    # ------------------------------------------------------------------

    def run_deepslice(self) -> None:
        project = self._state.project
        if project is None:
            return
        if self._deepslice_job and self._deepslice_job.is_running():
            return

        from verso.gui.dialogs.bad_sections import BadSectionsDialog

        dlg = BadSectionsDialog(
            project.sections, reverse_order=self._window._reverse_axis_proposal, parent=self._window
        )
        if dlg.exec() != BadSectionsDialog.DialogCode.Accepted:
            return
        bad_ids = dlg.bad_section_ids()
        self._window._reverse_axis_proposal = dlg.reverse_section_order()

        self._window._update_deepslice_enabled(running=True)
        self._window._statusbar.showMessage("Running DeepSlice suggestions...")

        self._deepslice_job = BackgroundJob(
            self._window,
            DeepSliceWorker(
                project,
                reverse_section_order=self._window._reverse_axis_proposal,
                bad_section_ids=bad_ids,
            ),
            title="DeepSlice",
            message="Running DeepSlice prediction...",
            min_width=350,
        )
        self._deepslice_job.start(
            self._on_deepslice_done,
            self._on_deepslice_finished,
            on_error=self._on_deepslice_error,
        )

    def _on_deepslice_done(self, result) -> None:
        import copy

        project = self._state.project
        atlas = self._state.atlas
        if project is None:
            return
        if not warn_if_missing_dimensions(self._window, project.sections):
            return
        from verso.engine.deepslice import apply_deepslice_suggestions_with_atlas

        # Snapshot last-saved alignments before applying so dirtied sections can
        # revert to their genuine baseline via "Clear edits".
        baselines = {s.id: copy.deepcopy(s.alignment) for s in project.sections}

        touched = apply_deepslice_suggestions_with_atlas(
            project,
            result,
            atlas.shape if atlas is not None else None,
            reverse_axis=self._window._reverse_axis_proposal,
        )
        if atlas is not None:
            self._window._sync_position_mm(project.sections)

        # DeepSlice proposals are unsaved edits: flag the sections it touched as
        # dirty so the Overview table and filmstrip show them yellow until the
        # user saves (mirroring the batch-mask flow).
        for section_id in touched:
            self._state.set_baseline(section_id, "align", baselines[section_id])
            self._state.mark_dirty(section_id, "align")

        self._window._overview.refresh()
        # Reload the current view so the align SaveBar reflects the new dirty
        # state; this also re-renders the panel for the active section.
        self._window._on_section_changed(self._state.section_index)
        self._window._refresh_filmstrip_dots()
        self._window._update_slicing_position()
        self._window._statusbar.showMessage(f"Applied {len(touched)} DeepSlice suggestions", 5000)

    def _on_deepslice_error(self, message: str) -> None:
        QMessageBox.warning(self._window, "DeepSlice failed", message)
        self._window._statusbar.showMessage("DeepSlice failed", 5000)

    def _on_deepslice_finished(self) -> None:
        self._deepslice_job = None
        self._window._update_deepslice_enabled(running=False)

    def revert_to_default_proposal(self) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None or atlas is None:
            return
        if not warn_if_missing_dimensions(self._window, project.sections):
            return
        from verso.engine.anchoring import reset_in_progress_to_default_proposals

        changed = reset_in_progress_to_default_proposals(
            project.sections,
            atlas.shape,
            interpolation_axis=self._window._interpolation_axis(),
            reverse_axis=self._window._reverse_axis_proposal,
        )
        self._window._sync_position_mm(project.sections)
        self._window._overview.refresh()
        self._window._on_section_changed(self._state.section_index)
        self._window._update_slicing_position()
        self._window._statusbar.showMessage(f"Restored {changed} default proposals", 3000)

    # ------------------------------------------------------------------
    # Batch clears
    # ------------------------------------------------------------------

    def clear_all_alignments(self) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None or atlas is None:
            return

        reply = QMessageBox.question(
            self._window,
            "Clear all alignments",
            "Clear every stored alignment, warp control point, and editable proposal, "
            "then restore VERSO's default proposal for all sections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
            return
        if not warn_if_missing_dimensions(self._window, project.sections):
            return

        from verso.engine.anchoring import reset_in_progress_to_default_proposals

        changed = reset_in_progress_to_default_proposals(
            project.sections,
            atlas.shape,
            interpolation_axis=self._window._interpolation_axis(),
            reverse_axis=self._window._reverse_axis_proposal,
            include_complete=True,
        )
        self._window._sync_position_mm(project.sections)
        self._window._after_batch_clear()
        self._window._statusbar.showMessage(
            f"Cleared all alignments and restored {changed} default proposals",
            5000,
        )

    def clear_all_slice_masks(self) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        reply = QMessageBox.question(
            self._window,
            "Clear all slice masks",
            f"Delete the slice mask for all {len(project.sections)} sections? "
            "This removes the PNG files from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
            return
        removed = 0
        for section in project.sections:
            old = section.preprocessing.slice_mask_path
            if old:
                with contextlib.suppress(OSError):
                    Path(old).unlink(missing_ok=True)
                removed += 1
            section.preprocessing.slice_mask_path = None
        self._window._after_batch_clear()
        self._window._statusbar.showMessage(f"Cleared {removed} slice masks", 5000)

    def clear_all_manual_cps(self) -> None:
        """Batch: drop every hand-placed control point, keeping auto ones."""
        self._clear_all_cps(auto=False)

    def clear_all_auto_cps(self) -> None:
        """Batch: drop every auto-generated control point, keeping manual ones."""
        self._clear_all_cps(auto=True)

    def _clear_all_cps(self, *, auto: bool) -> None:
        project = self._state.project
        if project is None or not project.sections:
            return
        kind = "automatic" if auto else "manual"
        reply = QMessageBox.question(
            self._window,
            f"Clear all {kind} control points",
            f"Remove every {kind} warp control point from all {len(project.sections)} sections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
            return
        cleared = 0
        for section in project.sections:
            cps = section.warp.control_points
            kept = [cp for cp in cps if cp.auto != auto]
            if len(kept) != len(cps):
                cleared += 1
            section.warp.control_points = kept
            if not kept:
                section.warp.status = AlignmentStatus.NOT_STARTED
            elif any(not cp.auto for cp in kept):
                section.warp.status = AlignmentStatus.COMPLETE
            else:
                section.warp.status = AlignmentStatus.IN_PROGRESS
        self._window._after_batch_clear()
        self._window._statusbar.showMessage(
            f"Cleared {kind} control points on {cleared} sections", 5000
        )

    # ------------------------------------------------------------------
    # Automatic (elastix) control points
    # ------------------------------------------------------------------

    def is_auto_cp_atlas(self) -> bool:
        from verso.engine.elastix import is_supported_atlas

        project = self._state.project
        return project is not None and is_supported_atlas(project.atlas.name)

    def _resolve_elastix_params(self) -> ElastixParams:
        project = self._state.project
        if project is not None and project.elastix_params is not None:
            return project.elastix_params
        return ElastixParams()

    def edit_elastix_params(self) -> None:
        from verso.gui.dialogs.elastix_params import ElastixParamsDialog

        project = self._state.project
        if project is None:
            return
        dialog = ElastixParamsDialog(self._resolve_elastix_params(), self._window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            project.elastix_params = dialog.get_params()
            if self._state.project_path is not None:
                self._window._write_project(self._state.project_path)
            self._window._statusbar.showMessage("Saved automatic registration parameters", 4000)

    def auto_generate_warp_cps(self) -> None:
        """Generate control points for the current section (Warp view button)."""
        project = self._state.project
        section = self._state.current_section
        atlas = self._state.atlas
        if project is None or section is None or atlas is None:
            return
        if not self.is_auto_cp_atlas():
            QMessageBox.information(
                self._window,
                "Atlas not supported",
                "Automatic control points are only available for Allen mouse atlases.",
            )
            return
        if not section.alignment.is_anchored:
            QMessageBox.information(
                self._window,
                "No alignment",
                "Align this section before generating control points.",
            )
            return
        self._auto_cp_batch = False
        self._run_auto_cp([section], "Generating control points…", self._on_single_auto_cp_done)

    def batch_auto_generate_warps(self) -> None:
        """Generate control points for every aligned section (Batch menu)."""
        project = self._state.project
        atlas = self._state.atlas
        if project is None or not project.sections or atlas is None:
            return
        if not self.is_auto_cp_atlas():
            QMessageBox.information(
                self._window,
                "Atlas not supported",
                "Automatic control points are only available for Allen mouse atlases.",
            )
            return
        aligned = [s for s in project.sections if s.alignment.is_anchored]
        if not aligned:
            QMessageBox.information(
                self._window,
                "No aligned sections",
                "Align sections before generating control points.",
            )
            return
        reply = QMessageBox.question(
            self._window,
            "Auto-generate control points",
            f"Generate control points for {len(aligned)} aligned section(s)? "
            "Existing automatic control points will be replaced; manual points are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
            return
        self._auto_cp_batch = True
        self._run_auto_cp(
            aligned,
            "Generating control points for all slices…",
            self._on_batch_auto_cp_done,
        )

    def _run_auto_cp(self, sections: list, message: str, on_done) -> None:
        project = self._state.project
        atlas = self._state.atlas
        if project is None or atlas is None:
            return
        self._auto_cp_job = BackgroundJob(
            self._window,
            AutoCPWorker(
                self._elastix_worker,
                list(sections),
                atlas,
                project.working_scale,
                self._resolve_elastix_params(),
            ),
            title="Automatic control points",
            message=message,
            modal=True,
            min_width=350,
        )
        self._window._update_deepslice_enabled()  # disable triggers while running
        self._auto_cp_job.start(on_done, self._on_auto_cp_finished)

    def _on_single_auto_cp_done(self, completed: int, errors: list[str]) -> None:
        worker = self._auto_cp_job.worker if self._auto_cp_job is not None else None
        section = self._state.current_section
        n = 0
        if worker is not None and section is not None:
            cps = worker.results.get(section.id)
            if cps is not None:
                n = len(cps)
                self._window._warp.apply_auto_control_points(cps)
        self._window._refresh_current_step_dot()
        self._window._statusbar.showMessage(f"Generated {n} control points", 5000)
        warn_errors(
            self._window,
            "Some sections failed",
            errors,
            f"{len(errors)} sections could not be processed:",
        )

    def _on_batch_auto_cp_done(self, completed: int, errors: list[str]) -> None:
        worker = self._auto_cp_job.worker if self._auto_cp_job is not None else None
        project = self._state.project
        total = 0
        if worker is not None and project is not None:
            by_id = {s.id: s for s in project.sections}
            for sid, cps in worker.results.items():
                section = by_id.get(sid)
                if section is None:
                    continue
                manual = [cp for cp in section.warp.control_points if not cp.auto]
                section.warp.control_points = manual + list(cps)
                if not section.warp.control_points:
                    section.warp.status = AlignmentStatus.NOT_STARTED
                elif manual:
                    section.warp.status = AlignmentStatus.COMPLETE
                else:
                    # Auto-only warp: a proposal awaiting review → yellow.
                    section.warp.status = AlignmentStatus.IN_PROGRESS
                total += len(cps)
        self._window._after_batch_clear()  # refresh dependent UI + persist project
        self._window._statusbar.showMessage(
            f"Generated {total} control points across {completed} sections", 6000
        )
        warn_errors(
            self._window,
            "Some sections failed",
            errors,
            f"{len(errors)} sections could not be processed:",
        )

    def _on_auto_cp_finished(self) -> None:
        self._auto_cp_job = None
        self._window._update_deepslice_enabled()
