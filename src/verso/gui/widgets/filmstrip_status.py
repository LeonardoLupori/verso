"""Single home for filmstrip status-dot colours.

The colour of a section's dot is ``section_step_color(section, step, dirty=…)``.
Before this, three MainWindow methods recomputed it at three scopes (all sections
/ current / one by id). This presenter owns that one computation; the window keeps
only thin delegators that supply the active ``step`` (its ``_current_mode``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verso.engine.model.status import section_step_color

if TYPE_CHECKING:
    from verso.gui.state import AppState
    from verso.gui.widgets.filmstrip import Filmstrip

_STEPS = ("prep", "align", "warp")


class FilmstripStatusPresenter:
    """Pushes per-section status-dot colours to the filmstrip for a given step."""

    def __init__(self, state: AppState, filmstrip: Filmstrip) -> None:
        self._state = state
        self._filmstrip = filmstrip

    def refresh_all(self, step: str) -> None:
        """Recompute every dot for ``step``.

        Steps without per-section status (e.g. Annotate, whose annotations are a
        project-global resource) clear the dots so a previous view's colours do
        not linger.
        """
        project = self._state.project
        if project is None:
            return
        if step not in _STEPS:
            self._filmstrip.set_statuses([None] * len(project.sections))
            return
        self._filmstrip.set_statuses([self._color(s, step) for s in project.sections])

    def refresh_index(self, index: int, step: str) -> None:
        """Refresh a single section's dot by row index."""
        project = self._state.project
        if project is None or step not in _STEPS:
            return
        if 0 <= index < len(project.sections):
            self._filmstrip.set_status_color(index, self._color(project.sections[index], step))

    def refresh_section(self, section_id: str, step: str) -> None:
        """Refresh a single section's dot by id (e.g. on a dirty-flag flip)."""
        project = self._state.project
        if project is None or step not in _STEPS:
            return
        for i, section in enumerate(project.sections):
            if section.id == section_id:
                self._filmstrip.set_status_color(i, self._color(section, step))
                return

    def _color(self, section, step: str):
        return section_step_color(section, step, dirty=self._state.is_dirty(section.id, step))
