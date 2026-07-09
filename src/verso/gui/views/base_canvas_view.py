"""Shared base for the editable canvas views (Prep / Align / Warp).

Owns the parts that were copy-pasted across the three views: the shallow undo
stack, the dirty-flag plumbing into AppState, and the save / revert / clear
skeletons.  Each view supplies the small view-specific pieces through the hook
methods below, so the "unsaved edits that survive navigation and revert to
last-saved" contract lives in exactly one place.  (The read-only Overview table
is not a canvas view and does not derive from this.)

Two snapshot-ish concepts meet here; keep them distinct:

* The **undo stack** is transient per-view session history holding
  ``_capture_edit()`` values (an anchoring list, a control-point list, or a mask
  array).  It is reset on save/revert/clear and on section load, and never
  leaves the view.  (Unrelated to ``registration._SectionSnapshot``.)
* The **last-saved baseline** lives in the
  :class:`~verso.gui.draft_store.DraftStore` and survives navigation;
  ``_saved_copy()`` / ``_apply_saved()`` manage it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from verso.engine.model.project import Section
    from verso.gui.state import AppState


class BaseCanvasView(QWidget):
    """Base class for a canvas view whose edits are unsaved drafts.

    Subclasses set :attr:`STEP` (``"prep"`` / ``"align"`` / ``"warp"``) and
    implement the ``_``-prefixed hooks.  Everything else — the undo stack, the
    ``is_dirty`` / ``_set_dirty`` bridge to AppState, and the save/revert/clear
    flow — is provided here.
    """

    #: The draft step this view owns; used as the DraftStore key's second field.
    STEP: str = ""
    #: Shallow undo depth — a handful of steps is plenty for canvas editing.
    _UNDO_LIMIT: int = 10

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._undo_stack: list[Any] = []

    # ------------------------------------------------------------------
    # Dirty flag (single source of truth is AppState / DraftStore)
    # ------------------------------------------------------------------

    def is_dirty(self) -> bool:
        section = self._current_section()
        return section is not None and self._state.is_dirty(section.id, self.STEP)

    def _set_dirty(self, dirty: bool) -> None:
        """Flip the current section's dirty flag for this step in AppState.

        ``mark_dirty`` / ``clear_dirty`` are idempotent and emit
        ``dirty_changed`` only on a real transition, which drives the save bar
        and filmstrip dot.
        """
        section = self._current_section()
        if section is None:
            return
        if dirty:
            self._state.mark_dirty(section.id, self.STEP)
        else:
            self._state.clear_dirty(section.id, self.STEP)

    def _saved_state(self) -> Any | None:
        """The last-saved baseline for the current section/step, if any."""
        section = self._current_section()
        if section is None:
            return None
        return self._state.get_baseline(section.id, self.STEP)

    # ------------------------------------------------------------------
    # Undo stack (transient per-view session history)
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Restore the previous edit from the undo history (Ctrl+Z)."""
        section = self._current_section()
        if section is None or not self._undo_stack:
            return
        self._end_edit_gesture()
        previous = self._undo_stack.pop()
        self._restore(previous)
        self._set_dirty(not self._matches_saved(previous))
        self._after_undo_restore()

    def _push_undo(self) -> None:
        """Snapshot the current edit before a mutating change."""
        if self._current_section() is None:
            return
        self._undo_stack.append(self._capture_edit())
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)

    def _reset_undo(self) -> None:
        """Clear the undo history (whenever the baseline is re-snapshotted)."""
        self._end_edit_gesture()
        self._undo_stack.clear()

    # ------------------------------------------------------------------
    # Save / revert / clear (identical flow; view-specific pieces are hooks)
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """Commit the current draft to the section, marking it saved."""
        section = self._current_section()
        if section is None:
            return False
        if not self._commit():
            return False
        self._reset_undo()
        self._set_dirty(False)
        self._state.sync_baseline(section.id, self.STEP, self._saved_copy())
        self._after_save()
        return True

    def revert(self) -> bool:
        """Discard unsaved edits, restoring the last-saved baseline."""
        section = self._current_section()
        baseline = self._saved_state()
        if section is None or baseline is None:
            return False
        self._apply_saved(baseline)
        self._reset_undo()
        self._set_dirty(False)
        self._state.sync_baseline(section.id, self.STEP, self._saved_copy())
        self._after_revert()
        return True

    def clear(self) -> bool:
        """Wipe this step's state back to default (and persist that wipe)."""
        section = self._current_section()
        if section is None:
            return False
        self._wipe()
        self._reset_undo()
        self._set_dirty(False)
        self._state.sync_baseline(section.id, self.STEP, self._saved_copy())
        self._after_clear()
        return True

    # ------------------------------------------------------------------
    # Required hooks — each view implements these
    # ------------------------------------------------------------------

    def _current_section(self) -> Section | None:
        """The section this view is currently editing."""
        raise NotImplementedError

    def _capture_edit(self) -> Any:
        """Snapshot the current editable value for the undo stack."""
        raise NotImplementedError

    def _restore(self, snapshot: Any) -> None:
        """Apply an undo snapshot back onto the section and refresh the display."""
        raise NotImplementedError

    def _matches_saved(self, snapshot: Any) -> bool:
        """Whether an undo snapshot equals the last-saved baseline (→ clean)."""
        raise NotImplementedError

    def _saved_copy(self) -> Any:
        """Deep copy of the current model piece to store as the new baseline."""
        raise NotImplementedError

    def _apply_saved(self, baseline: Any) -> None:
        """Restore ``baseline`` into the section and refresh the display (revert)."""
        raise NotImplementedError

    def _commit(self) -> bool:
        """Persist the draft to the section/disk; return False to abort the save."""
        raise NotImplementedError

    def _wipe(self) -> None:
        """Reset this step's state to default (clear)."""
        raise NotImplementedError

    def has_persisted_state(self) -> bool:
        """Whether Clear has saved/non-default state to wipe (enables Reset).

        Public because :class:`~verso.gui.controllers.save_controller.SaveController`
        reads it to drive each save bar's Reset button.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks — default no-ops
    # ------------------------------------------------------------------

    def _end_edit_gesture(self) -> None:
        """Close any in-progress drag gesture before an undo / reset."""

    def _after_undo_restore(self) -> None:
        """Emit view signals after an undo restores a snapshot."""

    def _after_save(self) -> None:
        """Emit view signals after a successful save."""

    def _after_revert(self) -> None:
        """Emit view signals after a successful revert."""

    def _after_clear(self) -> None:
        """Emit view signals after a successful clear."""
