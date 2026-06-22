"""Shared application state passed between MainWindow and views."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume
    from verso.engine.drafts import PrepDraft
    from verso.engine.model.project import Project, Section


class _AtlasLoader(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, atlas_name: str) -> None:
        super().__init__()
        self._name = atlas_name

    def run(self) -> None:
        try:
            from verso.engine.atlas import AtlasVolume

            self.done.emit(AtlasVolume(self._name))
        except Exception as exc:
            self.error.emit(str(exc))


class AppState(QObject):
    """Observable container for the currently loaded project and selection."""

    project_changed = pyqtSignal()
    section_changed = pyqtSignal(int)
    atlas_changed = pyqtSignal()  # emitted when atlas finishes loading
    atlas_error = pyqtSignal(str)
    dirty_changed = pyqtSignal(str, str)  # (section_id, step) — edit registry change

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._project_path: Path | None = None
        self._section_index: int = 0
        self._atlas: AtlasVolume | None = None
        self._atlas_thread: QThread | None = None
        # Persistent unsaved-edit bookkeeping, surviving slice/view navigation.
        # _dirty: which (section.id, step) pairs have unsaved edits.
        # _baselines: last-saved view-state snapshot for each dirty (id, step),
        #             so "Clear edits" can revert to it even after navigation.
        # _prep_drafts: resident slice/L-R mask edits, keyed by section.id.
        self._dirty: dict[tuple[str, str], bool] = {}
        self._baselines: dict[tuple[str, str], object] = {}
        self._prep_drafts: dict[str, PrepDraft] = {}

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    @property
    def project(self) -> Project | None:
        return self._project

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    def load_project(self, project: Project, path: Path | None = None) -> None:
        self._project = project
        self._project_path = path
        self._section_index = 0
        self._dirty.clear()
        self._baselines.clear()
        self._prep_drafts.clear()
        self.project_changed.emit()
        self.section_changed.emit(0)

    def set_project_path(self, path: Path | None) -> None:
        self._project_path = path

    # ------------------------------------------------------------------
    # Edit registry (persistent unsaved-edit tracking)
    # ------------------------------------------------------------------

    def mark_dirty(self, section_id: str, step: str) -> None:
        """Flag a section/step as having unsaved edits."""
        if self._dirty.get((section_id, step)):
            return
        self._dirty[(section_id, step)] = True
        self.dirty_changed.emit(section_id, step)

    def clear_dirty(self, section_id: str, step: str) -> None:
        """Clear the unsaved-edit flag for a section/step (e.g. after save)."""
        if self._dirty.pop((section_id, step), None) is None:
            return
        self.dirty_changed.emit(section_id, step)

    def is_dirty(self, section_id: str, step: str) -> bool:
        return bool(self._dirty.get((section_id, step)))

    def any_dirty(self) -> bool:
        return bool(self._dirty)

    def clear_all_edits(self) -> None:
        """Forget every unsaved edit (registry + resident drafts) without saving."""
        self._dirty.clear()
        self._baselines.clear()
        self._prep_drafts.clear()

    def forget_section(self, section_id: str) -> None:
        """Drop all edit bookkeeping for a section being removed from the project.

        Pops every ``_dirty``, ``_baselines`` and ``_prep_drafts`` entry keyed by
        ``section_id`` so a removed section leaves no orphaned state behind.
        """
        for key in [k for k in self._dirty if k[0] == section_id]:
            self.clear_dirty(*key)
        for key in [k for k in self._baselines if k[0] == section_id]:
            self._baselines.pop(key, None)
        self._prep_drafts.pop(section_id, None)

    def dirty_sections(self) -> list[tuple[Section, set[str]]]:
        """Return (section, {dirty steps}) for every section with unsaved edits."""
        if self._project is None:
            return []
        by_id = {s.id: s for s in self._project.sections}
        grouped: dict[str, set[str]] = {}
        for (section_id, step), flag in self._dirty.items():
            if flag and section_id in by_id:
                grouped.setdefault(section_id, set()).add(step)
        return [(by_id[sid], steps) for sid, steps in grouped.items()]

    # ------------------------------------------------------------------
    # Last-saved baselines (for per-view "Clear edits" reverts)
    # ------------------------------------------------------------------

    def set_baseline(self, section_id: str, step: str, snapshot: object) -> None:
        """Stash the last-saved view-state for a section/step.

        Stores only if absent so the first stash — taken at the clean→dirty
        transition — captures the genuine last-saved value, never a later
        mid-edit state.
        """
        self._baselines.setdefault((section_id, step), snapshot)

    def get_baseline(self, section_id: str, step: str) -> object | None:
        return self._baselines.get((section_id, step))

    def pop_baseline(self, section_id: str, step: str) -> object | None:
        return self._baselines.pop((section_id, step), None)

    # ------------------------------------------------------------------
    # Prep mask drafts (resident in RAM until saved)
    # ------------------------------------------------------------------

    def get_prep_draft(self, section_id: str) -> PrepDraft | None:
        return self._prep_drafts.get(section_id)

    def set_prep_draft(self, section_id: str, draft: PrepDraft) -> None:
        self._prep_drafts[section_id] = draft

    def pop_prep_draft(self, section_id: str) -> PrepDraft | None:
        return self._prep_drafts.pop(section_id, None)

    def has_prep_draft(self, section_id: str) -> bool:
        return section_id in self._prep_drafts

    # ------------------------------------------------------------------
    # Section selection
    # ------------------------------------------------------------------

    @property
    def section_index(self) -> int:
        return self._section_index

    @property
    def current_section(self) -> Section | None:
        if self._project is None or not self._project.sections:
            return None
        return self._project.sections[self._section_index]

    def set_section(self, index: int) -> None:
        if self._project is None:
            return
        index = max(0, min(index, len(self._project.sections) - 1))
        if index != self._section_index:
            self._section_index = index
            self.section_changed.emit(index)

    # ------------------------------------------------------------------
    # Atlas
    # ------------------------------------------------------------------

    @property
    def atlas(self) -> AtlasVolume | None:
        return self._atlas

    def load_atlas(self, atlas_name: str) -> None:
        """Load atlas in a background thread so the UI stays responsive."""
        # Cancel any in-progress load
        if self._atlas_thread and self._atlas_thread.isRunning():
            self._atlas_thread.quit()
            self._atlas_thread.wait()

        self._atlas = None
        thread = QThread(self)
        loader = _AtlasLoader(atlas_name)
        loader.moveToThread(thread)

        thread.started.connect(loader.run)
        loader.done.connect(self._on_atlas_loaded)
        loader.error.connect(self.atlas_error)
        loader.done.connect(thread.quit)
        loader.error.connect(thread.quit)
        thread.finished.connect(loader.deleteLater)

        self._atlas_thread = thread
        self._loader = loader  # keep reference alive
        thread.start()

    def _on_atlas_loaded(self, atlas: AtlasVolume) -> None:
        self._atlas = atlas
        self.atlas_changed.emit()

    def shutdown(self) -> None:
        """Stop the background atlas loader. Must be called before destruction."""
        if self._atlas_thread is not None:
            try:
                if self._atlas_thread.isRunning():
                    self._atlas_thread.quit()
                    self._atlas_thread.wait()
            except RuntimeError:
                pass  # C++ object already deleted — thread has already finished
        self._atlas_thread = None
