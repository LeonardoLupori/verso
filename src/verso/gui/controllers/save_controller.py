"""Collapses the per-view Save / Clear edits / Reset handling into one path.

Prep, Align and Warp each expose the same ``save()/revert()/clear()`` contract
and each properties page carries a ``save_bar``. Rather than nine near-identical
handlers on the window, the three views register here by step name and the
save-bar signals dispatch through a single parameterized path. The dependent-UI
refresh itself stays on the window (its coordinator role); this controller only
routes the button clicks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verso.gui.main_window import MainWindow


class SaveController:
    """Routes each view's save-bar buttons to that view's save/revert/clear."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        # step -> (view, properties page); filled by register() as views are wired.
        self._views: dict[str, tuple[object, object]] = {}

    def register(self, step: str, view, page) -> None:
        """Associate a step name ("prep"/"align"/"warp") with its view and page."""
        self._views[step] = (view, page)

    def on_save(self, step: str) -> None:
        view, _ = self._views[step]
        if view.save():
            self._window.after_view_save()

    def on_revert(self, step: str) -> None:
        view, page = self._views[step]
        if view.revert():
            self._refresh_page(step, page)
            self._window.after_view_revert()

    def on_clear(self, step: str) -> None:
        view, page = self._views[step]
        if view.clear():
            self._refresh_page(step, page)
            # Clear writes to disk (it wipes persisted state), so use the save path.
            self._window.after_view_save()

    def _refresh_page(self, step: str, page) -> None:
        """Re-render the properties page after a revert/clear.

        Only Prep repaints its page from the section (mask/flip controls); the
        Align/Warp pages have nothing section-derived to refresh.
        """
        if step == "prep":
            page.update_section(self._state.current_section)
