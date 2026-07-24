"""Shared application state passed between MainWindow and views."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from verso.gui.draft_store import DraftStore

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume
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
            _log.exception("Atlas load failed for %r", self._name)
            self.error.emit(str(exc))


class AppState(QObject):
    """Observable container for the currently loaded project and selection."""

    project_changed = pyqtSignal()
    section_changed = pyqtSignal(int)
    atlas_changed = pyqtSignal()  # emitted when atlas finishes loading
    atlas_error = pyqtSignal(str)
    dirty_changed = pyqtSignal(str, str)  # (section_id, step) — edit registry change
    # "Series content/status changed across one or more sections; re-render
    # dependent UI." Distinct from section_changed(int), which is selection.
    # Controllers emit this after mutating the model instead of poking the window.
    sections_changed = pyqtSignal()
    status_message = pyqtSignal(str)  # transient status-bar text
    # The section *list* changed (add/remove/reorder), so list-dependent UI
    # (filmstrip tiles, overview table) must be rebuilt — heavier than the
    # status-only sections_changed, which only recolours existing dots/rows.
    structure_changed = pyqtSignal()
    # A DeepSlice run started (True) / ended (False) — drives the menu action's
    # "running…" label and disabled state without a controller→window poke.
    deepslice_running_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._project_path: Path | None = None
        self._section_index: int = 0
        self._atlas: AtlasVolume | None = None
        self._atlas_thread: QThread | None = None
        self._loader: _AtlasLoader | None = None
        # Persistent unsaved-edit bookkeeping (dirty flags, last-saved baselines,
        # resident prep-mask drafts) lives in one store keyed by (id, step). Its
        # dirty_changed is forwarded as this object's own signal.
        self._drafts = DraftStore(self)
        self._drafts.dirty_changed.connect(self.dirty_changed)

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
        _log.info(
            "Project loaded: %s (%d section(s), atlas=%s, working_scale=%.4f)",
            path,
            len(project.sections),
            project.atlas.name,
            project.working_scale,
        )
        self._project = project
        self._project_path = path
        self._section_index = 0
        self._drafts.clear_all()
        self.project_changed.emit()
        self.section_changed.emit(0)

    def set_project_path(self, path: Path | None) -> None:
        self._project_path = path

    # ------------------------------------------------------------------
    # UI notifications (let controllers signal the window without a back-ref)
    # ------------------------------------------------------------------

    def notify_sections_changed(self) -> None:
        """Signal that series content/status changed; dependent UI should refresh."""
        self.sections_changed.emit()

    def show_status(self, message: str) -> None:
        """Post transient text to the status bar."""
        self.status_message.emit(message)

    # ------------------------------------------------------------------
    # Edit registry (persistent unsaved-edit tracking) — delegates to DraftStore
    # ------------------------------------------------------------------

    def mark_dirty(self, section_id: str, step: str) -> None:
        """Flag a section/step as having unsaved edits."""
        self._drafts.mark_dirty(section_id, step)

    def clear_dirty(self, section_id: str, step: str) -> None:
        """Clear the unsaved-edit flag for a section/step (e.g. after save)."""
        self._drafts.clear_dirty(section_id, step)

    def is_dirty(self, section_id: str, step: str) -> bool:
        return self._drafts.is_dirty(section_id, step)

    def any_dirty(self) -> bool:
        return self._drafts.any_dirty()

    def clear_all_edits(self) -> None:
        """Forget every unsaved edit (registry + resident drafts) without saving."""
        self._drafts.clear_all()

    def forget_section(self, section_id: str) -> None:
        """Drop all edit bookkeeping for a section being removed from the project."""
        self._drafts.forget_section(section_id)

    def dirty_sections(self) -> list[tuple[Section, set[str]]]:
        """Return (section, {dirty steps}) for every section with unsaved edits."""
        if self._project is None:
            return []
        by_id = {s.id: s for s in self._project.sections}
        grouped: dict[str, set[str]] = {}
        for section_id, step in self._drafts.dirty_keys():
            if section_id in by_id:
                grouped.setdefault(section_id, set()).add(step)
        return [(by_id[sid], steps) for sid, steps in grouped.items()]

    # ------------------------------------------------------------------
    # Last-saved baselines (for per-view "Clear edits" reverts)
    # ------------------------------------------------------------------

    def set_baseline(self, section_id: str, step: str, snapshot: object) -> None:
        """Stash the last-saved view-state, first stash wins (see DraftStore.set_saved)."""
        self._drafts.set_saved(section_id, step, snapshot)

    def sync_baseline(self, section_id: str, step: str, snapshot: object) -> None:
        """Refresh the baseline from a *clean* section (see DraftStore.sync_saved)."""
        self._drafts.sync_saved(section_id, step, snapshot)

    def get_baseline(self, section_id: str, step: str) -> object | None:
        return self._drafts.get_saved(section_id, step)

    def pop_baseline(self, section_id: str, step: str) -> object | None:
        return self._drafts.pop_saved(section_id, step)

    # ------------------------------------------------------------------
    # Working payloads (resident in RAM until saved) — Prep's slice mask
    # ------------------------------------------------------------------

    def get_working(self, section_id: str, step: str) -> object | None:
        return self._drafts.get_working(section_id, step)

    def set_working(self, section_id: str, step: str, payload: object) -> None:
        self._drafts.set_working(section_id, step, payload)

    def pop_working(self, section_id: str, step: str) -> object | None:
        return self._drafts.pop_working(section_id, step)

    def has_working(self, section_id: str, step: str) -> bool:
        return self._drafts.has_working(section_id, step)

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
