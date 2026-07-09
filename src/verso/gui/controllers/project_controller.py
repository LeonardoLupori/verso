"""Owns project persistence and QuickNII series-anchoring orchestration.

Extracted from ``MainWindow`` so the plane-proposal math (default anchorings,
``position_mm`` sync) and the ``project.json`` write live in one cohesive place
that both the window and the background-job controller can call. All *computation*
already lives in :mod:`verso.engine.anchoring`; this controller only orchestrates
it against the current :class:`AppState` and reports progress through the state's
``status_message`` signal rather than poking the status bar.

Dependent-UI refreshes are *not* done here: callers mutate the model, then emit
``AppState.sections_changed`` (via :meth:`AppState.notify_sections_changed`) so the
window re-renders. The window reference is kept only to parent modal dialogs.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from verso.engine.io.quint_io import load_quicknii, load_visualign
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import DEFAULT_PROJECT_FILENAME, Project
from verso.gui.dialogs.new_project import NewProjectDialog
from verso.gui.utils import warn_if_missing_dimensions

if TYPE_CHECKING:
    from verso.engine.model.project import Section
    from verso.gui.main_window import MainWindow


class ProjectController:
    """Series-anchoring proposals + project persistence for the current project."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        # Proposal direction shared between series interpolation (here) and the
        # DeepSlice / reverse batch operations (JobController). Reset to False on
        # every project load by MainWindow._on_project_changed.
        self.reverse_axis_proposal = False

    # ------------------------------------------------------------------
    # Series anchoring proposals (thin orchestration over engine.anchoring)
    # ------------------------------------------------------------------

    @property
    def interpolation_axis(self) -> int:
        """The anchoring voxel axis index for the current project."""
        project = self._state.project
        return 1 if project is None else project.interpolation_axis_index

    def anchoring_position_mm(self, anchoring: list[float]) -> float:
        """Millimetre position of a plane along the interpolation axis."""
        atlas = self._state.atlas
        if atlas is None:
            return 0.0
        center = atlas.cut_center(anchoring)
        return atlas.voxel_to_mm(center[self.interpolation_axis])

    def sync_position_mm(self, sections: list[Section]) -> None:
        """Populate ``position_mm`` for every section that has a valid anchoring."""
        atlas = self._state.atlas
        if atlas is None:
            return
        for section in sections:
            if section.alignment.is_anchored:
                section.alignment.position_mm = self.anchoring_position_mm(
                    section.alignment.current_anchoring
                )

    def initialize_default_anchorings(self, sections: list[Section]) -> None:
        """Seed empty section planes with default proposals (engine does the math)."""
        atlas = self._state.atlas
        if atlas is None:
            return
        if not warn_if_missing_dimensions(self._window, sections):
            return

        from verso.engine.anchoring import initialize_default_anchorings

        initialize_default_anchorings(
            sections,
            atlas_shape=atlas.shape,
            interpolation_axis=self.interpolation_axis,
            reverse_axis=self.reverse_axis_proposal,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_project(self, path: Path | None = None) -> None:
        """Write ``project.json`` to ``path`` (default: the current project path)."""
        project = self._state.project
        path = path or self._state.project_path
        if project is None or path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            project.save(path)
            self._state.show_status(f"Saved project to {path}")
        except Exception as exc:
            QMessageBox.critical(self._window, "Cannot save project", str(exc))

    # ------------------------------------------------------------------
    # Opening / creating projects (each guarded by the unsaved-edits prompt)
    # ------------------------------------------------------------------

    def open_project(self) -> None:
        if not self._window.confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open VERSO Project",
            "",
            "VERSO project (*.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            self.open_project_path(Path(path))

    def open_project_path(self, project_path: Path) -> None:
        """Load a project from ``project_path`` (public; also used by drag-drop)."""
        try:
            project = Project.load(project_path)
            self._state.load_project(project, project_path)
        except Exception as exc:
            QMessageBox.critical(self._window, "Cannot open project", str(exc))

    def new_project(self) -> None:
        self._open_new_project_dialog()

    def on_images_dropped(self, paths: list[str]) -> None:
        """Drag-and-drop onto the empty overview → New Project, pre-filled."""
        self._open_new_project_dialog(paths)

    def _open_new_project_dialog(self, initial_paths: list[str] | None = None) -> None:
        if not self._window.confirm_discard_active_draft():
            return
        dlg = NewProjectDialog(self._window, initial_paths=initial_paths)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            project = dlg.result_project()
            if project is not None:
                self._state.load_project(project, dlg.result_project_path())

    def open_quicknii(self) -> None:
        self._open_foreign("Open QuickNII JSON", load_quicknii)

    def open_visualign(self) -> None:
        self._open_foreign("Open VisuAlign JSON", load_visualign)

    def _open_foreign(self, title: str, loader) -> None:
        """Import a QuickNII/VisuAlign JSON as a new (path-less) project."""
        if not self._window.confirm_discard_active_draft():
            return
        path, _ = QFileDialog.getOpenFileName(
            self._window, title, "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._state.load_project(loader(Path(path)))

    # ------------------------------------------------------------------
    # Saving / discarding
    # ------------------------------------------------------------------

    def save_all(self) -> bool:
        """Persist every unsaved edit across all slices/steps (Ctrl+S / menu).

        Returns True if the project was saved, False if there's no project or the
        user cancelled a Save-As prompt.
        """
        from verso.engine.drafts import commit_alignment, commit_prep_draft, commit_warp

        project = self._state.project
        if project is None:
            return False

        # 1. Persist the active view's in-RAM edits first — this materializes
        #    prep masks held only in the view and seeds a default align plane on
        #    an explicit save — then it clears that section/step from the registry.
        #    Save unconditionally (not gated on is_dirty) so Ctrl+S matches the
        #    per-view Save button: an untouched alignment still gets its default
        #    plane committed instead of being silently skipped.
        active = self._window.active_view()
        if active is not None:
            active.save()

        # 2. Persist every remaining dirty (section, step).  Snapshot the list up
        #    front since we mutate the registry inside the loop.
        for section, steps in self._state.dirty_sections():
            if "prep" in steps:
                # Flip invalidation already happened at toggle time, so this only
                # writes the mask (None when only flips changed) — it won't
                # clobber an alignment the user redid after flipping.
                mask = self._state.pop_working(section.id, "prep")
                commit_prep_draft(section, mask)
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
        self.initialize_default_anchorings(project.sections)
        self.sync_position_mm(project.sections)

        # 4. Single project.json write + dependent-UI refresh.
        if self._state.project_path is None:
            self.save_project_as()
            if self._state.project_path is None:
                return False  # user cancelled the Save-As dialog
        else:
            self.write_project(self._state.project_path)
        self._window.sync_dependent_ui()
        return True

    def save_project_as(self) -> None:
        self._window.save_active_view()
        if self._state.project is None:
            return
        current_path = self._state.project_path
        suggested = str(current_path) if current_path is not None else DEFAULT_PROJECT_FILENAME
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Project As", suggested, "JSON files (*.json)"
        )
        if path:
            project_path = Path(path)
            if project_path.suffix == "":
                project_path = project_path.with_suffix(".json")
            self.write_project(project_path)
            self._state.set_project_path(project_path)
            self._window.sync_dependent_ui()

    def discard_all(self) -> None:
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

    # ------------------------------------------------------------------
    # Flip invalidation (the alignment is tied to the image coordinate frame)
    # ------------------------------------------------------------------

    def confirm_flip(self, section) -> bool:
        """Return True when a flip may proceed.

        Shows a warning dialog when the section has an alignment that is
        genuinely saved or has unsaved edits, and
        ``dialog_prefs.show_align_deletion`` is True.  If the user ticks
        "Do not show again", the flag is persisted to the project.

        ``section.alignment.status``/``anchoring`` alone aren't enough:
        ``interpolate_anchorings`` seeds every not-yet-aligned section with an
        IN_PROGRESS default guess on project load, which isn't something the
        user did and isn't what flipping would actually destroy.  Instead,
        check the same "saved" signal the Align save-bar uses
        (``stored_anchoring``, only ever set by an explicit commit) plus the
        per-step dirty flag for an in-progress unsaved edit.
        """
        from verso.engine.anchoring import is_anchored

        has_alignment = (
            is_anchored(section.alignment.stored_anchoring)
            or bool(section.warp.control_points)
            or self._state.is_dirty(section.id, "align")
            or self._state.is_dirty(section.id, "warp")
        )
        if not has_alignment:
            return True

        project = self._state.project
        if project is None or not project.dialog_prefs.show_align_deletion:
            return True

        from verso.gui.dialogs.flip_warning import confirm_flip_deletes_alignment

        confirmed, suppress = confirm_flip_deletes_alignment(self._window)
        if confirmed and suppress:
            project.dialog_prefs.show_align_deletion = False
        return confirmed

    def invalidate_alignment_for_flip(self, section) -> bool:
        """Wipe a section's alignment + warp the instant its flip is toggled.

        A horizontal/vertical flip changes the image coordinate frame, so any
        existing registration no longer applies.  Doing this at toggle time (not
        at save time) means a re-alignment performed in the new orientation is
        preserved through the next save instead of being wiped by it. Returns
        True if anything was wiped (so the caller can refresh the relevant UI).
        """
        from verso.engine.drafts import reset_alignment

        has_alignment = (
            section.alignment.status != AlignmentStatus.NOT_STARTED
            or bool(section.warp.control_points)
            or section.alignment.is_anchored
        )
        if not has_alignment:
            return False
        reset_alignment(section)
        self.clear_alignment_view_state(section)
        self.seed_alignment_to_default_proposal(section)
        return True

    def clear_alignment_view_state(self, section) -> None:
        """Drop registry dirty + stashed baselines for a section whose alignment
        was just wiped, so Align/Warp re-sync to the cleared state on activate."""
        self._state.clear_dirty(section.id, "align")
        self._state.clear_dirty(section.id, "warp")
        self._state.pop_baseline(section.id, "align")
        self._state.pop_baseline(section.id, "warp")

    def seed_alignment_to_default_proposal(self, section) -> None:
        """Re-seed a wiped section with the default interpolated proposal.

        After a flip or prep reset the anchoring is all-zeros. This produces the
        same result as clicking the Align "Reset" button: re-running the series
        interpolation so the section gets the best available positional guess
        based on its neighbours. Without a non-zero anchoring every canvas drag
        handler bails out silently.
        """
        project = self._state.project
        if project is None or self._state.atlas is None:
            return
        self.initialize_default_anchorings(project.sections)
        self.sync_position_mm([section])

    def on_prep_invalidated_alignment(self) -> None:
        """A prep Clear/Reset wiped the current section's alignment + warp."""
        section = self._state.current_section
        if section is None:
            return
        self.clear_alignment_view_state(section)
        self.seed_alignment_to_default_proposal(section)

    # ------------------------------------------------------------------
    # Section list edits (add / remove / reorder)
    # ------------------------------------------------------------------

    def on_sections_reordered(self) -> None:
        """Recompute everything that depends on section order, then persist.

        Called after a slice-index edit in Overview and after add/remove/reorder.
        The model recompute (series interpolation + position_mm) happens here; the
        list-dependent widgets are rebuilt by the window's ``structure_changed``
        slot, which this emits before writing the project once.
        """
        project = self._state.project
        if project is None:
            return

        if self._state.atlas is not None and warn_if_missing_dimensions(
            self._window, project.sections
        ):
            from verso.engine.anchoring import interpolate_anchorings

            interpolate_anchorings(
                project.sections,
                atlas_shape=self._state.atlas.shape,
                interpolation_axis=project.interpolation_axis_index,
                reverse_axis=self.reverse_axis_proposal,
            )
        self.sync_position_mm(project.sections)
        self._state.structure_changed.emit()
        self.write_project()

    def reorder_by_filename(self) -> None:
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
            self._window,
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
        for section, index in zip(project.sections, indices, strict=False):
            section.slice_index = index
        project.sort_sections()

        if keep_id is not None:
            new_pos = next((i for i, s in enumerate(project.sections) if s.id == keep_id), None)
            if new_pos is not None:
                self._state.set_section(new_pos)

        self.on_sections_reordered()

    def add_images(self) -> None:
        """Add new section images to the current project (Image menu).

        New images are appended after the current series with provisional slice
        indices (``max + 1``…); the user corrects them in the Overview table.
        ``working_scale`` is never recomputed — new thumbnails are generated at
        the project's existing scale so all working-resolution geometry stays valid.
        """
        from verso.engine.sections import make_added_sections
        from verso.gui.dialogs.new_project import _IMAGE_FILTER, generate_thumbnails

        if self._state.project is None:
            QMessageBox.information(self._window, "No project", "Open or create a project first.")
            return
        if self._state.project_path is None:
            QMessageBox.information(
                self._window,
                "Save project first",
                "Save the project before adding images so the new thumbnails have a home on disk.",
            )
            return
        if not self._window.confirm_discard_active_draft():
            return

        # Re-fetch after the confirm gate: a "Discard" reloads the project object.
        project = self._state.project
        project_path = self._state.project_path
        if project is None or project_path is None:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self._window, "Add Section Images", "", _IMAGE_FILTER
        )
        if not paths:
            return

        thumbnails_dir = project_path.parent / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        new_sections, skipped = make_added_sections(project.sections, paths, thumbnails_dir)

        if skipped:
            names = "\n".join(f"  • {Path(p).name}" for p in skipped)
            QMessageBox.warning(
                self._window,
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
        generate_thumbnails(new_sections, project.working_scale, self._window, title="Add images")

        if keep_id is not None:
            pos = next((i for i, s in enumerate(project.sections) if s.id == keep_id), None)
            if pos is not None:
                self._state.set_section(pos)
        self.on_sections_reordered()
        self._state.show_status(f"Added {len(new_sections)} image(s) to the project")

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
                self._window,
                "Channel count differs",
                f"The project expects {expected} channel(s), but some added images "
                f"differ:\n\n{lines}\n\nThey may not display correctly.",
            )

    def remove_sections(self, section_ids: list[str]) -> None:
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
                self._window,
                "Cannot remove",
                "A project must keep at least one image. Removing these would empty it.",
            )
            return

        n = len(to_remove)
        resp = QMessageBox.question(
            self._window,
            "Remove from project",
            f"Remove {n} image{'s' if n != 1 else ''} from the project?\n\n"
            "Their generated thumbnails and masks will be deleted. The original "
            "image files are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        if not self._window.confirm_discard_active_draft():
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
                with contextlib.suppress(OSError):
                    artifact.unlink(missing_ok=True)
            self._state.forget_section(section.id)

        project.sections = surviving

        if keep_id is not None and any(s.id == keep_id for s in project.sections):
            pos = next(i for i, s in enumerate(project.sections) if s.id == keep_id)
        else:
            pos = min(old_index, len(project.sections) - 1)
        self._state.set_section(pos)

        self.on_sections_reordered()
        # If the index is unchanged but now points at a different section, force a
        # reload of the active view and properties.
        if pos == old_index:
            self._state.section_changed.emit(pos)
        self._state.show_status(f"Removed {n} image{'s' if n != 1 else ''} from the project")
