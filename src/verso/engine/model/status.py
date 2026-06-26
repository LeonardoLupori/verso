"""Per-step section status used by the Overview table and filmstrip dots.

A "step" is one of ``"prep"``, ``"align"``, ``"warp"``.  Status maps to a
traffic-light colour:

- gray  — no saved state and no unsaved edits for this step
- yellow — has unsaved edits (the section is dirty for this step), or a warp made
  up only of auto-generated control points that has not yet been saved/accepted
  (an elastix proposal awaiting review)
- green — saved/persisted state exists and no unsaved edits

The ``dirty`` flag is supplied by the caller (the GUI's edit registry) so this
module stays UI-agnostic and importable from the pure engine layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verso.engine.model.alignment import AlignmentStatus

if TYPE_CHECKING:
    from verso.engine.model.project import Section

# Single source of truth for the traffic-light colours, shared by the Overview
# table and the filmstrip status dots.
STATUS_COLOR: dict[AlignmentStatus, str] = {
    AlignmentStatus.NOT_STARTED: "#888888",  # gray
    AlignmentStatus.IN_PROGRESS: "#E6A817",  # yellow
    AlignmentStatus.COMPLETE: "#4CAF50",  # green
}

STEPS = ("prep", "align", "warp")


def section_step_status(section: Section, step: str, *, dirty: bool) -> AlignmentStatus:
    """Return the traffic-light status of *section* for *step*.

    Args:
        section: the section to inspect.
        step: one of ``"prep"``, ``"align"``, ``"warp"``.
        dirty: whether the section has unsaved edits for this step (from the
            GUI edit registry).  When True the status is always IN_PROGRESS
            (yellow), regardless of any saved state.

    Returns:
        ``AlignmentStatus`` — IN_PROGRESS when dirty, otherwise COMPLETE when a
        saved state exists for the step, else NOT_STARTED.
    """
    if step == "warp":
        # Warp is special: with no control points the step is empty (gray) even
        # mid-edit — e.g. right after the last CP was removed — so this check
        # takes precedence over the dirty flag.
        cps = section.warp.control_points
        if not cps:
            return AlignmentStatus.NOT_STARTED
        if dirty:
            return AlignmentStatus.IN_PROGRESS
        # A purely auto-generated (elastix) warp is a proposal awaiting review:
        # it stays yellow until the user accepts it by saving (which promotes
        # ``warp.status`` to COMPLETE) or adds a manual control point. A warp
        # with any hand-placed point is user-owned and goes green when saved.
        if all(cp.auto for cp in cps) and section.warp.status != AlignmentStatus.COMPLETE:
            return AlignmentStatus.IN_PROGRESS
        return AlignmentStatus.COMPLETE

    if dirty:
        return AlignmentStatus.IN_PROGRESS

    if step == "prep":
        pp = section.preprocessing
        done = bool(pp.slice_mask_path or pp.flip_horizontal or pp.flip_vertical)
        return AlignmentStatus.COMPLETE if done else AlignmentStatus.NOT_STARTED

    if step == "align":
        return (
            AlignmentStatus.COMPLETE
            if section.alignment.status == AlignmentStatus.COMPLETE
            else AlignmentStatus.NOT_STARTED
        )

    return AlignmentStatus.NOT_STARTED


def section_step_color(section: Section, step: str, *, dirty: bool) -> str:
    """Convenience: the hex colour for :func:`section_step_status`."""
    return STATUS_COLOR[section_step_status(section, step, dirty=dirty)]


__all__ = ["STATUS_COLOR", "STEPS", "section_step_status", "section_step_color"]
