"""Pure-engine helpers for adding and removing sections in a project.

These back the GUI's "Add images to project" and "Remove from project" actions.
They never touch :attr:`Project.working_scale` — it is frozen after import because
every working-resolution coordinate (control points, masks, anchorings) is
defined relative to it. Recomputing it would silently invalidate all existing
geometry.

No Qt imports here, so this stays usable from scripts and tests.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from verso.engine.drafts import slice_mask_path_for
from verso.engine.io.image_io import thumbnail_filename
from verso.engine.model.alignment import Alignment, AlignmentStatus, WarpState
from verso.engine.model.project import Section


def _natural_name_key(path: str) -> list[object]:
    """Order filenames with embedded numbers numerically (``s2`` before ``s10``)."""
    stem = Path(path).stem
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", stem)]


def _norm(path: str | Path) -> str:
    """Normalize a path for case-insensitive, separator-agnostic comparison."""
    return os.path.normcase(os.path.normpath(str(path)))


def next_section_ids(existing_ids: list[str], count: int) -> list[str]:
    """Return ``count`` fresh ``sNNN`` ids not present in ``existing_ids``.

    Ids continue the ``s001``/``s002`` scheme used at import. The next number is
    one past the largest numeric ``sNNN`` suffix already in use; any candidate
    that nonetheless collides (e.g. non-standard ids) is skipped.

    Args:
        existing_ids: Ids already used by the project's sections.
        count: How many new ids to generate.

    Returns:
        A list of ``count`` unique ids, none of which appear in ``existing_ids``.
    """
    used = set(existing_ids)
    max_n = 0
    for sid in existing_ids:
        m = re.fullmatch(r"s(\d+)", sid)
        if m:
            max_n = max(max_n, int(m.group(1)))

    ids: list[str] = []
    n = max_n
    while len(ids) < count:
        n += 1
        candidate = f"s{n:03d}"
        if candidate not in used:
            ids.append(candidate)
            used.add(candidate)
    return ids


def make_added_sections(
    existing_sections: list[Section],
    new_paths: list[str | Path],
    thumbnails_dir: Path,
) -> tuple[list[Section], list[str]]:
    """Build :class:`Section` objects for images added to an existing project.

    New sections are appended *after* the current series: their provisional
    ``slice_index`` runs from ``max(existing slice_index) + 1`` upward, assigned
    in natural-filename order. The user is expected to correct these in the
    Overview table. ``working_scale`` is deliberately not consulted or changed.

    A path is skipped (and reported) when it duplicates an existing section's
    original image, or when its canonical thumbnail filename would collide with
    an existing section's or an already-accepted new section's thumbnail (two
    sources sharing a filename stem). Skipping protects existing thumbnails from
    being clobbered and keeps per-section artifact paths unique.

    Args:
        existing_sections: The project's current sections.
        new_paths: Candidate original-image paths to add.
        thumbnails_dir: The project's ``thumbnails/`` directory.

    Returns:
        ``(new_sections, skipped_paths)`` — the sections to append (in
        slice-index order) and the input paths that were skipped.
    """
    existing_originals = {_norm(s.original_path) for s in existing_sections}
    used_thumbs = {_norm(s.thumbnail_path) for s in existing_sections}

    kept: list[str] = []
    skipped: list[str] = []
    for path in new_paths:
        path = str(path)
        thumb = _norm(thumbnails_dir / thumbnail_filename(path))
        if _norm(path) in existing_originals or thumb in used_thumbs:
            skipped.append(path)
            continue
        kept.append(path)
        used_thumbs.add(thumb)

    if not kept:
        return [], skipped

    kept.sort(key=_natural_name_key)
    base_index = max((s.slice_index for s in existing_sections), default=0)
    new_ids = next_section_ids([s.id for s in existing_sections], len(kept))

    sections: list[Section] = []
    for i, path in enumerate(kept):
        sections.append(
            Section(
                id=new_ids[i],
                slice_index=base_index + 1 + i,
                original_path=path,
                thumbnail_path=str(thumbnails_dir / thumbnail_filename(path)),
                alignment=Alignment(status=AlignmentStatus.NOT_STARTED),
                warp=WarpState(status=AlignmentStatus.NOT_STARTED),
            )
        )
    return sections, skipped


def removed_section_artifacts(section: Section, surviving_sections: list[Section]) -> list[Path]:
    """Return generated files safe to delete when ``section`` is removed.

    Covers the section's working-resolution thumbnail and slice mask.
    A path is excluded when any surviving section maps to the same file (possible
    when two sources share a filename stem), so a removal never deletes a file
    another section still relies on. Original images are never included.

    Args:
        section: The section being removed.
        surviving_sections: Sections that remain after the removal.

    Returns:
        Canonical artifact paths that are safe to ``unlink``.
    """
    surviving_paths: set[str] = set()
    for s in surviving_sections:
        surviving_paths.add(_norm(s.thumbnail_path))
        surviving_paths.add(_norm(slice_mask_path_for(s)))

    candidates = [
        Path(section.thumbnail_path),
        slice_mask_path_for(section),
    ]
    return [p for p in candidates if _norm(p) not in surviving_paths]
