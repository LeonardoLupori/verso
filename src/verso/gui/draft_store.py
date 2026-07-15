"""Single per-``(section_id, step)`` store for unsaved draft edits.

Replaces the three parallel dicts AppState used to carry (a dirty set, a
last-saved baseline map, and a resident prep-mask map) with **one**
:class:`SectionDraft` entry per ``(section_id, step)``.  Each entry holds the
last-saved snapshot (revert target), an optional working-edit payload, and the
dirty flag, so the whole "unsaved edits that survive navigation and revert to
last-saved" concept lives in a single, unit-testable place.

Pure GUI-side state — it only needs ``QObject`` for the :attr:`dirty_changed`
signal and never imports a view or the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class SectionDraft:
    """Unsaved-edit bookkeeping for one ``(section, step)``.

    Attributes:
        last_saved: Deep-copied snapshot of the last-saved model piece
            (preprocessing / alignment / warp), used to revert "Clear edits".
            ``None`` until a baseline has been captured.
        working: Optional resident working-edit payload.  Only Prep uses it, to
            hold the in-progress slice-mask draft; Align/Warp keep their live
            edit on the section itself and leave this ``None``.
        dirty: Whether this ``(section, step)`` has unsaved edits.
    """

    last_saved: object | None = None
    working: object | None = None
    dirty: bool = False


class DraftStore(QObject):
    """One :class:`SectionDraft` per ``(section_id, step)``.

    Owns the dirty flags, last-saved baselines, and working payloads that must
    survive slice/view navigation.  Emits :attr:`dirty_changed` on every real
    clean<->dirty transition (idempotent otherwise).
    """

    dirty_changed = pyqtSignal(str, str)  # (section_id, step)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._entries: dict[tuple[str, str], SectionDraft] = {}

    def _entry(self, section_id: str, step: str) -> SectionDraft:
        """Return the entry for ``(id, step)``, creating an empty one if absent."""
        key = (section_id, step)
        entry = self._entries.get(key)
        if entry is None:
            entry = SectionDraft()
            self._entries[key] = entry
        return entry

    def _prune(self, section_id: str, step: str) -> None:
        """Drop a fully-empty entry so the store doesn't accrete dead keys."""
        key = (section_id, step)
        entry = self._entries.get(key)
        if (
            entry is not None
            and not entry.dirty
            and entry.last_saved is None
            and entry.working is None
        ):
            del self._entries[key]

    # ------------------------------------------------------------------
    # Dirty flags
    # ------------------------------------------------------------------

    def mark_dirty(self, section_id: str, step: str) -> None:
        """Flag a ``(section, step)`` as having unsaved edits."""
        entry = self._entry(section_id, step)
        if entry.dirty:
            return
        entry.dirty = True
        self.dirty_changed.emit(section_id, step)

    def clear_dirty(self, section_id: str, step: str) -> None:
        """Clear the unsaved-edit flag for a ``(section, step)`` (e.g. after save)."""
        entry = self._entries.get((section_id, step))
        if entry is None or not entry.dirty:
            return
        entry.dirty = False
        self.dirty_changed.emit(section_id, step)

    def is_dirty(self, section_id: str, step: str) -> bool:
        entry = self._entries.get((section_id, step))
        return entry is not None and entry.dirty

    def any_dirty(self) -> bool:
        return any(entry.dirty for entry in self._entries.values())

    def dirty_keys(self) -> list[tuple[str, str]]:
        """All ``(section_id, step)`` pairs currently dirty."""
        return [key for key, entry in self._entries.items() if entry.dirty]

    # ------------------------------------------------------------------
    # Last-saved baselines (revert targets)
    # ------------------------------------------------------------------

    def set_saved(self, section_id: str, step: str, snapshot: object) -> None:
        """Stash the last-saved snapshot only if none is present yet.

        Mirrors the former ``dict.setdefault`` baseline: the first stash — taken
        at the clean->dirty transition — captures the genuine last-saved value,
        never a later mid-edit state.  Used by batch flows (e.g. DeepSlice) that
        dirty a section *after* mutating it.
        """
        entry = self._entry(section_id, step)
        if entry.last_saved is None:
            entry.last_saved = snapshot

    def sync_saved(self, section_id: str, step: str, snapshot: object) -> None:
        """Refresh the last-saved snapshot from a *clean* section only.

        Views call this when they load/activate a section: while **clean** the
        section is at its last-saved state, so the baseline is (re)set; while
        **dirty** the snapshot stashed at the clean->dirty transition is kept so
        "Clear edits" reverts correctly even after navigating away and back.
        """
        entry = self._entries.get((section_id, step))
        if entry is not None and entry.dirty:
            return
        self._entry(section_id, step).last_saved = snapshot

    def get_saved(self, section_id: str, step: str) -> object | None:
        entry = self._entries.get((section_id, step))
        return entry.last_saved if entry is not None else None

    def pop_saved(self, section_id: str, step: str) -> object | None:
        entry = self._entries.get((section_id, step))
        if entry is None:
            return None
        snapshot = entry.last_saved
        entry.last_saved = None
        self._prune(section_id, step)
        return snapshot

    # ------------------------------------------------------------------
    # Working payloads (resident in RAM until saved) — Prep only
    # ------------------------------------------------------------------

    def get_working(self, section_id: str, step: str) -> object | None:
        entry = self._entries.get((section_id, step))
        return entry.working if entry is not None else None

    def set_working(self, section_id: str, step: str, payload: object) -> None:
        self._entry(section_id, step).working = payload

    def pop_working(self, section_id: str, step: str) -> object | None:
        entry = self._entries.get((section_id, step))
        if entry is None:
            return None
        payload = entry.working
        entry.working = None
        self._prune(section_id, step)
        return payload

    def has_working(self, section_id: str, step: str) -> bool:
        entry = self._entries.get((section_id, step))
        return entry is not None and entry.working is not None

    # ------------------------------------------------------------------
    # Bulk
    # ------------------------------------------------------------------

    def forget_section(self, section_id: str) -> None:
        """Drop every entry for a section, emitting for any that were dirty."""
        for key in [k for k in self._entries if k[0] == section_id]:
            entry = self._entries.pop(key)
            if entry.dirty:
                self.dirty_changed.emit(*key)

    def clear_all(self) -> None:
        """Forget every entry without emitting (callers refresh the UI wholesale)."""
        self._entries.clear()
