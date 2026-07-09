"""The per-view Save / Clear edits / Reset controller.

Prep, Align and Warp each expose the same ``save()/revert()/clear()`` contract
and each properties page carries a ``save_bar``. Rather than nine near-identical
handlers on the window, the three views register here by step name and the
save-bar signals dispatch through a single parameterized path.

This controller is also the one place that turns AppState's dirty state — the
single source of truth — into save-bar button states, so the save bars are a
pure view of the registry:

- ``_on_dirty_changed`` updates one bar when the current section's dirty flag
  flips during editing (driven by ``AppState.dirty_changed``);
- ``refresh_all`` re-syncs all three bars (dirty + reset-enabled) whenever the
  current section or active view changes (the window calls it after loading the
  view so ``has_persisted_state`` reads the freshly-synced baseline).

The dependent-UI refresh (project write, overview, filmstrip) stays on the
window in its coordinator role.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verso.gui.main_window import MainWindow
    from verso.gui.views.base_canvas_view import BaseCanvasView
    from verso.gui.widgets.properties.sections.save_bar import SaveBarBox


class SaveController:
    """Routes save-bar buttons and mirrors AppState dirty state onto the bars."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        # step -> (view, its save bar); filled by register() as views are wired.
        # Only the save bar is kept, not the whole properties page: repainting a
        # page after revert/clear is a dependent-UI refresh the window owns.
        self._views: dict[str, tuple[BaseCanvasView, SaveBarBox]] = {}
        # Drive the save bars from the single source of truth: an edit that flips
        # a section's dirty flag emits AppState.dirty_changed, which enables or
        # disables the matching bar's Save / Clear buttons.
        self._state.dirty_changed.connect(self._on_dirty_changed)

    def register(self, step: str, view: BaseCanvasView, save_bar: SaveBarBox) -> None:
        """Associate a step name ("prep"/"align"/"warp") with its view and save bar."""
        self._views[step] = (view, save_bar)

    def refresh_all(self) -> None:
        """Re-sync every save bar's dirty + reset-enabled to the current section.

        Called by the window whenever the current section or active view changes,
        after the view has re-synced its baseline into AppState.
        """
        section = self._state.current_section
        for step, (view, save_bar) in self._views.items():
            dirty = section is not None and self._state.is_dirty(section.id, step)
            save_bar.set_dirty(dirty)
            save_bar.set_reset_enabled(view.has_persisted_state())

    def _on_dirty_changed(self, section_id: str, step: str) -> None:
        """Reflect one section/step's dirty flag onto its save bar while editing."""
        section = self._state.current_section
        if section is None or section.id != section_id:
            return
        entry = self._views.get(step)
        if entry is None:
            return
        entry[1].set_dirty(self._state.is_dirty(section_id, step))

    def on_save(self, step: str) -> None:
        view, _ = self._views[step]
        if view.save():
            self._window.after_view_save()

    def on_revert(self, step: str) -> None:
        view, _ = self._views[step]
        if view.revert():
            self._window.after_view_revert()

    def on_clear(self, step: str) -> None:
        view, _ = self._views[step]
        if view.clear():
            self._window.after_view_clear()
