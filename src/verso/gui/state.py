"""Shared application state passed between MainWindow and views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal

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
            self.error.emit(str(exc))


class AppState(QObject):
    """Observable container for the currently loaded project and selection."""

    project_changed = pyqtSignal()
    section_changed = pyqtSignal(int)
    atlas_changed = pyqtSignal()        # emitted when atlas finishes loading
    atlas_error = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project: "Project | None" = None
        self._section_index: int = 0
        self._atlas: "AtlasVolume | None" = None
        self._atlas_thread: QThread | None = None

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    @property
    def project(self) -> "Project | None":
        return self._project

    def load_project(self, project: "Project") -> None:
        self._project = project
        self._section_index = 0
        self.project_changed.emit()
        self.section_changed.emit(0)

    # ------------------------------------------------------------------
    # Section selection
    # ------------------------------------------------------------------

    @property
    def section_index(self) -> int:
        return self._section_index

    @property
    def current_section(self) -> "Section | None":
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
    def atlas(self) -> "AtlasVolume | None":
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
        self._loader = loader   # keep reference alive
        thread.start()

    def _on_atlas_loaded(self, atlas: "AtlasVolume") -> None:
        self._atlas = atlas
        self.atlas_changed.emit()
