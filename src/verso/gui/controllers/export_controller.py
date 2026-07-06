"""Owns the five export flows and their shared pre-flight/completion glue.

The heavy lifting lives in ``engine/io/export_images``, ``engine/io/export_stack``
and ``engine/io/quint_io``; this controller only drives the dialogs, progress
reporting and section selection. It reaches the loaded project/atlas and the
draft-discard prompt through the owning :class:`MainWindow`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from verso.gui.utils import warn_errors

if TYPE_CHECKING:
    from collections.abc import Callable

    from verso.gui.main_window import MainWindow


class ExportController:
    """Drives QuickNII/VisuAlign, images-with-overlay and aligned-stack exports."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state

    # ------------------------------------------------------------------
    # QuickNII / VisuAlign
    # ------------------------------------------------------------------

    def export_quicknii_xml(self) -> None:
        from verso.engine.io.quint_io import save_quicknii_xml

        self._export_quint(
            "Export QuickNII XML", "quicknii.xml", "XML files (*.xml)", save_quicknii_xml
        )

    def export_quicknii(self) -> None:
        from verso.engine.io.quint_io import save_quicknii

        self._export_quint(
            "Export QuickNII JSON", "quicknii.json", "JSON files (*.json)", save_quicknii
        )

    def export_visualign(self) -> None:
        from verso.engine.io.quint_io import save_visualign

        self._export_quint(
            "Export VisuAlign JSON", "visualign.json", "JSON files (*.json)", save_visualign
        )

    def _export_quint(
        self, title: str, default_suffix: str, file_filter: str, save_fn: Callable
    ) -> None:
        """Shared QuickNII/VisuAlign exporter: confirm, pick a path, write, offer PNGs.

        ``save_fn`` is one of the ``quint_io`` writers; they all share the
        ``(project, path, atlas_shape=...)`` signature.
        """
        if not self._window.confirm_discard_active_draft():
            return
        project = self._state.project
        if project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window, title, f"{project.name}-{default_suffix}", file_filter
        )
        if not path:
            return
        atlas_shape = self._state.atlas.shape if self._state.atlas else None
        save_fn(project, Path(path), atlas_shape=atlas_shape)
        self._maybe_create_pngs(path)

    def _maybe_create_pngs(self, export_path: str) -> None:
        """Offer to create PNG copies if any are missing next to the export."""
        project = self._state.project
        if project is None:
            return
        out_dir = Path(export_path).resolve().parent
        from verso.engine.io.quint_io import _export_image_filename

        missing = [
            s for s in project.sections if not (out_dir / _export_image_filename(s)).exists()
        ]
        if not missing:
            return
        reply = QMessageBox.question(
            self._window,
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

    # ------------------------------------------------------------------
    # Images with overlay / aligned stack
    # ------------------------------------------------------------------

    def export_images_with_overlay(self) -> None:
        """Open the export dialog and write the requested PNGs to disk."""
        from datetime import datetime

        from PyQt6.QtWidgets import QApplication

        from verso.engine.io.export_images import export_section
        from verso.gui.dialogs.export_images import ExportImagesDialog

        preflight = self._export_preflight()
        if preflight is None:
            return
        project, atlas, project_path = preflight

        dlg = ExportImagesDialog(
            n_selected=len(self._window._overview.selected_rows()),
            n_total=len(project.sections),
            parent=self._window,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        sections = self._select_export_sections(dlg, project)
        if sections is None:
            return

        options = dlg.options()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = project_path.parent / "exports" / f"images_with_overlay_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Exporting images...", "Cancel", 0, len(sections), self._window)
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
            except Exception as exc:
                errors.append(f"{Path(section.original_path).name}: {exc}")
            progress.setValue(idx + 1)
            QApplication.processEvents()

        progress.close()

        if errors:
            warn_errors(
                self._window,
                "Export finished with errors",
                errors,
                f"Wrote some images to:\n{out_dir}\n\nErrors:",
            )
        else:
            self._show_export_done(out_dir, f"Wrote {len(sections)} sections to:\n{out_dir}")

    def export_aligned_stack(self) -> None:
        """Open the aligned-stack dialog and write the un-warped TIFF stack."""
        from datetime import datetime

        from PyQt6.QtWidgets import QApplication

        from verso.engine.io.export_stack import (
            export_section_aligned,
            finalize_aligned_pages,
            write_aligned_stack,
        )
        from verso.gui.dialogs.export_stack import ExportStackDialog

        preflight = self._export_preflight()
        if preflight is None:
            return
        project, atlas, project_path = preflight

        dlg = ExportStackDialog(
            n_selected=len(self._window._overview.selected_rows()),
            n_total=len(project.sections),
            parent=self._window,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        sections = self._select_export_sections(dlg, project)
        if sections is None:
            return

        options = dlg.options()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = project_path.parent / "exports"
        out_path = out_dir / f"aligned_stack_{timestamp}.ome.tif"

        progress = QProgressDialog(
            "Resampling sections...", "Cancel", 0, len(sections), self._window
        )
        progress.setWindowTitle("Export aligned section stack")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()

        entries: list = []
        skipped: list[str] = []
        errors: list[str] = []
        canceled = False
        for idx, section in enumerate(sections):
            if progress.wasCanceled():
                canceled = True
                break
            progress.setLabelText(
                f"Resampling {idx + 1} / {len(sections)}: {Path(section.original_path).name}"
            )
            QApplication.processEvents()
            try:
                result = export_section_aligned(
                    section,
                    project,
                    atlas,
                    options.scale,
                    apply_slice_mask=options.background is not None,
                )
                if result is None:
                    skipped.append(Path(section.original_path).name)
                else:
                    page, valid = result
                    entries.append((section.slice_index, page, valid))
            except Exception as exc:
                errors.append(f"{Path(section.original_path).name}: {exc}")
            progress.setValue(idx + 1)
            QApplication.processEvents()

        if canceled:
            progress.close()
            return

        pages = finalize_aligned_pages(entries, options) if entries else []
        if pages:
            try:
                channel_names = [c.name for c in project.channels] or [
                    f"Ch {i}" for i in range(pages[0].shape[2])
                ]
                write_aligned_stack(pages, channel_names, out_path)
            except Exception as exc:
                errors.append(f"writing stack: {exc}")

        progress.close()

        notes: list[str] = []
        if skipped:
            notes.append(f"Skipped {len(skipped)} section(s) without alignment.")
        if errors:
            warn_errors(
                self._window,
                "Export finished with errors",
                errors,
                "\n".join(notes) + "\n\nErrors:",
            )
            return
        if not pages:
            QMessageBox.warning(
                self._window,
                "Export",
                "No sections had a usable alignment, so no stack was written.",
            )
            return

        msg = f"Wrote a {len(pages)}-section stack to:\n{out_path}"
        if notes:
            msg += "\n\n" + "\n".join(notes)
        self._show_export_done(out_dir, msg)

    # ------------------------------------------------------------------
    # Shared pre-flight / selection / completion
    # ------------------------------------------------------------------

    def _export_preflight(self):
        """Confirm drafts and validate project/atlas/path before an export.

        Returns ``(project, atlas, project_path)`` when the export may proceed,
        or ``None`` (after showing the relevant warning) when it cannot.
        """
        if not self._window.confirm_discard_active_draft():
            return None
        project = self._state.project
        if project is None or not project.sections:
            QMessageBox.warning(self._window, "Export", "No project is loaded.")
            return None
        atlas = self._state.atlas
        if atlas is None:
            QMessageBox.warning(
                self._window, "Export", "The atlas is still loading. Try again in a moment."
            )
            return None
        if self._state.project_path is None:
            QMessageBox.warning(
                self._window,
                "Export",
                "Save the project to disk before exporting so VERSO knows where to "
                "write the exports folder.",
            )
            return None
        return project, atlas, self._state.project_path

    def _select_export_sections(self, dlg, project) -> list | None:
        """Map an export dialog's all/selected choice to a list of sections."""
        if dlg.export_all():
            sections = list(project.sections)
        else:
            sections = [project.sections[i] for i in self._window._overview.selected_rows()]
        if not sections:
            QMessageBox.warning(self._window, "Export", "No sections selected.")
            return None
        return sections

    def _show_export_done(self, out_dir: Path, message: str) -> None:
        """Completion box with an "Open folder" shortcut to the export dir."""
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Export complete")
        box.setText(message)
        open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
