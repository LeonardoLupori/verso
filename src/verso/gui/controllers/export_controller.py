"""Owns the five export flows and their shared pre-flight/completion glue.

The heavy lifting lives in ``engine/io/export_images``, ``engine/io/export_stack``
and ``engine/io/quint_io``; this controller only drives the dialogs, progress
reporting and section selection. It reaches the loaded project/atlas and the
draft-discard prompt through the owning :class:`MainWindow`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from verso.gui.utils import warn_errors

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from verso.gui.main_window import MainWindow


class ExportController:
    """Drives QuickNII/VisuAlign, images-with-overlay and aligned-stack exports."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        self._quant_job = None  # type: ignore[var-annotated]  # BackgroundJob[QuantifyWorker]

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
        _log.info("%s: wrote %d section(s) to %s", title, len(project.sections), path)
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

            progress = self._make_progress(
                "Creating PNG copies…", "PNG export", len(project.sections)
            )
            try:
                write_section_pngs(project, out_dir, on_progress=self._progress_tick(progress))
            finally:
                progress.close()

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
        _log.info("Exporting %d section(s) with overlay → %s", len(sections), out_dir)

        progress = self._make_progress(
            "Exporting images...",
            "Export images with atlas overlay",
            len(sections),
            cancellable=True,
        )
        tick = self._progress_tick(progress)

        errors: list[str] = []
        for idx, section in enumerate(sections):
            if progress.wasCanceled():
                break
            tick(idx, len(sections), Path(section.original_path).name)
            try:
                export_section(section, project, atlas, options, out_dir)
            except Exception as exc:
                _log.exception("Overlay export failed for %s", section.original_path)
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
            _log.info("Overlay export finished: %d section(s) → %s", len(sections), out_dir)
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

        progress = self._make_progress(
            "Resampling sections...",
            "Export aligned section stack",
            len(sections),
            cancellable=True,
        )
        tick = self._progress_tick(progress)

        entries: list = []
        skipped: list[str] = []
        errors: list[str] = []
        canceled = False
        for idx, section in enumerate(sections):
            if progress.wasCanceled():
                canceled = True
                break
            tick(idx, len(sections), Path(section.original_path).name)
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
                _log.exception("Aligned-stack resample failed for %s", section.original_path)
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
                _log.exception("Cannot write aligned stack to %s", out_path)
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
    # Shared progress dialogs
    # ------------------------------------------------------------------

    def _make_progress(
        self, message: str, title: str, total: int, *, cancellable: bool = False
    ) -> QProgressDialog:
        """A modal 0..``total`` dialog, shown immediately."""
        from PyQt6.QtWidgets import QApplication

        progress = QProgressDialog(message, "Cancel" if cancellable else "", 0, total, self._window)
        progress.setWindowTitle(title)
        if not cancellable:
            progress.setCancelButton(None)
        progress.setMinimumWidth(320)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()
        return progress

    @staticmethod
    def _progress_tick(progress: QProgressDialog) -> Callable[[int, int, str], None]:
        """A ``(done, total, name)`` callback that advances ``progress``.

        The label names only the section being worked on — the bar itself shows
        how far along the run is. The work runs on the UI thread here, so each
        tick pumps the event loop to keep the dialog painting.
        """
        from PyQt6.QtWidgets import QApplication

        message = progress.labelText()

        def tick(done: int, total: int, name: str) -> None:
            progress.setValue(min(done, total))
            if name:
                progress.setLabelText(f"{message}\n{name}")
            QApplication.processEvents()

        return tick

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

    # ------------------------------------------------------------------
    # Quantification (Export ▸ Quantify)
    # ------------------------------------------------------------------

    def quantify_intensity(self) -> None:
        """Open the intensity-quantification dialog and run it in the background."""
        self._quantify("intensity")

    def quantify_dots(self) -> None:
        self._quantify("dots")

    def quantify_area(self) -> None:
        self._quantify("area")

    def _quantify(self, kind: str) -> None:
        from verso.engine import quantify_area, quantify_dots, quantify_intensity
        from verso.gui.dialogs.quantify_dialog import QuantifyDialog

        preflight = self._export_preflight()
        if preflight is None:
            return
        project, atlas, project_path = preflight
        project_dir = project_path.parent

        titles = None
        if kind in ("area", "dots"):
            titles = self._annotation_titles(project_dir, kind)
            if not titles:
                label = "area annotations" if kind == "area" else "point series"
                QMessageBox.information(
                    self._window,
                    "Quantify",
                    f"This project has no {label} to quantify.\n\n"
                    "Create one in the Annotate view first.",
                )
                return

        dlg = QuantifyDialog(
            kind,
            channel_names=[c.name for c in project.channels],
            annotation_titles=titles,
            parent=self._window,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        problem = dlg.validate()
        if problem:
            QMessageBox.warning(self._window, "Quantify", problem)
            return

        out_base = project_dir / "exports"
        options = dlg.quant_options()
        options.out_dir = str(out_base)

        # ``on_progress`` is the worker's progress signal: the engine calls it
        # once per section so the dialog can name the section being quantified.
        if kind == "intensity":

            def run_fn(on_progress):
                return quantify_intensity(
                    project,
                    project_dir=project_dir,
                    atlas=atlas,
                    options=options,
                    on_progress=on_progress,
                )
        elif kind == "area":
            annotation = dlg.annotation()

            def run_fn(on_progress):
                return quantify_area(
                    project,
                    annotation,
                    project_dir=project_dir,
                    atlas=atlas,
                    options=options,
                    on_progress=on_progress,
                )
        else:  # dots
            annotation = dlg.annotation()
            intensity_channels = dlg.intensity_channels()
            diameter = dlg.dot_diameter()

            def run_fn(on_progress):
                return quantify_dots(
                    project,
                    annotation,
                    intensity_channels=intensity_channels,
                    dot_diameter_px=diameter,
                    project_dir=project_dir,
                    atlas=atlas,
                    options=options,
                    on_progress=on_progress,
                )

        self._launch_quantify(run_fn, out_base, kind)

    def _launch_quantify(self, run_fn, out_base: Path, kind: str) -> None:
        from verso.gui.jobs import BackgroundJob, QuantifyWorker

        _log.info("Starting %s quantification → %s", kind, out_base)
        self._quant_job = BackgroundJob(
            self._window,
            QuantifyWorker(run_fn),
            title="Quantification",
            message=f"Running {kind} quantification…",
            # BackgroundJob wires QuantifyWorker.progress into the dialog: the bar
            # advances per section and the label names the one being quantified.
            modal=True,
            min_width=320,
        )
        self._quant_job.start(
            lambda _result: self._show_export_done(
                out_base,
                f"{kind.capitalize()} quantification complete.\n\n"
                f"CSV files were written under:\n{out_base}",
            ),
            self._on_quantify_finished,
            self._on_quantify_error,
        )

    def _on_quantify_finished(self) -> None:
        self._quant_job = None

    def _on_quantify_error(self, message: str) -> None:
        _log.error("Quantification could not run: %s", message)
        QMessageBox.warning(self._window, "Quantification could not run", message)

    @staticmethod
    def _annotation_titles(project_dir: Path, kind: str) -> list[str]:
        """Titles of annotations of ``kind`` (``"area"``/``"dots"``) without loading points."""
        import json

        from verso.engine.io.annotation_io import annotations_dir
        from verso.engine.model.annotation import AREA, POINT_SERIES

        want = AREA if kind == "area" else POINT_SERIES
        root = annotations_dir(project_dir)
        titles: list[str] = []
        if not root.exists():
            return titles
        for child in sorted(root.iterdir()):
            meta_path = child / "annotation.json"
            if not (child.is_dir() and meta_path.exists()):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(meta.get("type", POINT_SERIES)) == want and meta.get("title"):
                titles.append(str(meta["title"]))
        return titles

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
