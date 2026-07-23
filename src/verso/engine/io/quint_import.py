"""Build a full VERSO project from a QuickNII / VisuAlign JSON alignment.

The parsing / coordinate-convention layer lives in
:mod:`verso.engine.io.quint_io` (:func:`~verso.engine.io.quint_io.load_quicknii`
/ :func:`~verso.engine.io.quint_io.load_visualign`). Those produce an in-memory,
path-less project: anchorings and warp control points are populated, but every
section points at the bare ``filename`` string from the JSON, carries no cached
dimensions, and has no working-resolution thumbnail. This module closes that gap
so an imported QuickNII/VisuAlign alignment becomes a real, saveable VERSO
project (folder creation and thumbnail generation stay with the caller — the GUI
runs them behind a progress dialog, exactly like New Project).

Two image roles, both resolved by the caller *before* building:

* **Registration images** — the images QuickNII/VisuAlign actually registered,
  matched to the JSON ``filename`` entries (see :func:`match_registration_images`).
  They define the pixel space of the imported control points, i.e. the JSON
  ``width`` / ``height``.
* **Full-resolution originals** — what ``Section.original_path`` points at and
  what full-resolution export reads. Either a separately matched set, or the
  registration images reused as originals.

Coordinate handling. Imported anchoring is normalised ``[0, 1]`` and therefore
resolution-independent (left untouched). Imported control points are in
registration-image pixel space ``(Wr, Hr)``; VERSO stores control points in
working-resolution pixels ``(Ww, Hw) = round(original × working_scale)``.
:func:`build_quint_project` rescales every control point by
``(Ww / Wr, Hw / Hr)`` so the warp lands in VERSO's working grid unchanged.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

from verso.engine.io.image_io import (
    compute_working_scale,
    image_dimensions,
    thumbnail_filename,
)
from verso.engine.io.quint_io import load_visualign, read_quint_document
from verso.engine.model.alignment import ControlPoint
from verso.engine.model.project import AXIS_NAME_TO_INDEX, AtlasRef, Project

_log = logging.getLogger(__name__)

# Below this filename-similarity ratio an original is treated as unrelated to a
# section rather than auto-assigned to it — the user assigns those by hand.
_MIN_ORIGINAL_SIMILARITY = 0.2


def _read_slice_entries(json_path: Path) -> list[dict]:
    """Return the raw per-slice dicts from a QuickNII/VisuAlign JSON or XML, in file order."""
    data = read_quint_document(Path(json_path))
    raw = data.get("slices")
    if raw is None:
        raw = data.get("sections", [])
    return list(raw)


def filenames_are_thumbnails(filenames: list[str]) -> bool:
    """True when the JSON image names are QUINT working thumbnails, not originals.

    QuickNII/VisuAlign export working images as ``{stem}-thumb.png`` inside a
    ``thumbnails/`` folder and reference them by that relative path. When most of
    the names follow that convention the full-resolution originals live elsewhere,
    so an importer should ask for them separately instead of reusing the
    thumbnails. Bare names (plain QuickNII) or absolute paths (DeepSlice) are not
    thumbnails and return ``False``.
    """
    if not filenames:
        return False
    hits = 0
    for f in filenames:
        norm = f.replace("\\", "/").lower()
        parent_dirs = norm.split("/")[:-1]
        if "thumbnails" in parent_dirs or Path(norm).stem.endswith("-thumb"):
            hits += 1
    return hits >= len(filenames) / 2


def _match_keys(filename: str) -> list[str]:
    """Lowercased basename match keys for a JSON ``filename``.

    The JSON name may be a bare name (``IMG_0001.png``), a relative export path
    (``thumbnails/IMG_0001-thumb.png``), or an absolute path (DeepSlice). Match on
    the basename, its stem, and — because VERSO exports thumbnails as
    ``{stem}-thumb.png`` — a ``-thumb``-stripped stem so an exported project can
    be matched back to its source images.
    """
    name = Path(filename.replace("\\", "/")).name
    stem = Path(name).stem
    keys = [name.lower(), stem.lower()]
    if stem.lower().endswith("-thumb"):
        keys.append(stem[: -len("-thumb")].lower())
    return keys


def _index_folder(folder: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Index the image files in *folder* and its immediate subfolders.

    Returns ``(by_name, by_stem)`` maps, both lowercased. First occurrence wins so
    the shallowest match is preferred.
    """
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    search_dirs = [folder]
    try:
        search_dirs += sorted(d for d in folder.iterdir() if d.is_dir())
    except OSError:
        _log.warning("Could not list subfolders of %s", folder, exc_info=True)
    for directory in search_dirs:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.is_file():
                continue
            by_name.setdefault(path.name.lower(), path)
            by_stem.setdefault(path.stem.lower(), path)
    return by_name, by_stem


def _normalize_for_match(name: str) -> str:
    """Basename stem of *name*, lowercased and with any ``-thumb`` suffix removed.

    Puts a working-thumbnail name (``AL1A_002-thumb.png``) and its full-resolution
    original (``AL1A_002.tif``) into the same normalised space so filename
    similarity reflects the underlying section, not the export decoration.
    """
    stem = Path(str(name).replace("\\", "/")).stem.lower()
    if stem.endswith("-thumb"):
        stem = stem[: -len("-thumb")]
    return stem


def match_originals_by_similarity(
    reference_names: list[str],
    candidates: list[Path],
) -> dict[int, Path]:
    """Assign each section its most filename-similar original, one file per section.

    Used to attach full-resolution originals to sections whose reference name is a
    prestored thumbnail: the originals are picked as individual files (as in New
    Project) and may not share the thumbnails' exact names, so matching is by
    filename similarity rather than exact basename.

    Args:
        reference_names: Per-section reference filename (the JSON ``filename`` /
            thumbnail name), index-aligned with the sections.
        candidates: Available full-resolution image files to assign from.

    Returns:
        ``{section_index: path}`` for the sections that found a candidate above a
        small similarity floor. Assignment is greedy by descending similarity with
        each candidate used at most once, so the mapping is a partial one-to-one
        (unique) matching; sections left out are for the user to assign manually.
    """
    if not reference_names or not candidates:
        return {}
    refs = [_normalize_for_match(n) for n in reference_names]
    cands = [_normalize_for_match(p.name) for p in candidates]

    scored: list[tuple[float, int, int]] = []
    matcher = difflib.SequenceMatcher()
    for j, cn in enumerate(cands):
        matcher.set_seq2(cn)
        for i, rn in enumerate(refs):
            matcher.set_seq1(rn)
            scored.append((matcher.ratio(), i, j))
    # Highest similarity first; stable so ties fall to lower section/candidate index.
    scored.sort(key=lambda t: t[0], reverse=True)

    matched: dict[int, Path] = {}
    used_candidates: set[int] = set()
    for score, i, j in scored:
        if score < _MIN_ORIGINAL_SIMILARITY:
            break
        if i in matched or j in used_candidates:
            continue
        matched[i] = candidates[j]
        used_candidates.add(j)
    return matched


def match_registration_images(
    json_path: str | Path,
    folder: str | Path,
) -> tuple[dict[int, Path], list[tuple[int, str]]]:
    """Resolve each JSON slice's ``filename`` to a real image file in *folder*.

    Matching is by basename, case-insensitive and extension-tolerant (a ``.png``
    named in the JSON may be a ``.tif`` on disk), searching *folder* and one level
    of subfolders. Used to resolve the registered images (the prestored
    thumbnails); full-resolution originals are attached separately by
    :func:`match_originals_by_similarity`.

    Args:
        json_path: Path to the QuickNII/VisuAlign ``*.json`` file.
        folder: Directory to search for the section images.

    Returns:
        ``(matched, unmatched)`` where ``matched`` maps a 0-based slice index to
        the resolved :class:`~pathlib.Path`, and ``unmatched`` is a list of
        ``(index, filename)`` for slices with no match (for manual assignment).
    """
    raw_sections = _read_slice_entries(Path(json_path))
    by_name, by_stem = _index_folder(Path(folder))

    matched: dict[int, Path] = {}
    unmatched: list[tuple[int, str]] = []
    for i, raw in enumerate(raw_sections):
        filename = str(raw.get("filename", ""))
        hit: Path | None = None
        for key in _match_keys(filename):
            hit = by_name.get(key) or by_stem.get(key)
            if hit is not None:
                break
        if hit is not None:
            matched[i] = hit
        else:
            unmatched.append((i, filename))
    return matched, unmatched


def _registration_dims(raw: dict, fallback_path: Path | None) -> tuple[int, int]:
    """Registration (control-point) pixel dims for a slice.

    Prefer the JSON ``width``/``height`` (authoritative for the marker space);
    fall back to the registration image's own dimensions when the JSON omits them.
    """
    wr = int(raw.get("width", 0) or 0)
    hr = int(raw.get("height", 0) or 0)
    if wr > 0 and hr > 0:
        return wr, hr
    if fallback_path is not None:
        return image_dimensions(fallback_path)
    return 0, 0


def build_quint_project(
    json_path: str | Path,
    project_dir: str | Path,
    *,
    registration_paths: dict[int, Path],
    original_paths: dict[int, Path] | None = None,
    atlas_name: str | None = None,
    interpolation_axis: str | None = None,
) -> Project:
    """Build a self-contained VERSO :class:`Project` from a QuickNII/VisuAlign JSON.

    Parses the alignment via :func:`~verso.engine.io.quint_io.load_visualign`
    (which also handles marker-free QuickNII/DeepSlice files), then resolves image
    paths, caches dimensions, derives the working scale, and rescales the imported
    control points into VERSO's working-resolution grid. Does **not** create the
    folder or generate thumbnails — the caller owns that I/O.

    Args:
        json_path: Path to the QuickNII/VisuAlign ``*.json`` file.
        project_dir: Destination project folder (used to place absolute
            ``thumbnail_path`` entries under ``{project_dir}/thumbnails``).
        registration_paths: 0-based slice index → resolved registration image
            (from :func:`match_registration_images`). Must cover every slice.
        original_paths: Optional 0-based slice index → full-resolution original.
            When ``None`` (or missing an index), the registration image is reused
            as the original.
        atlas_name: Optional BrainGlobe atlas name to force. When ``None`` the
            atlas comes from the JSON ``target`` (already resolved to a BrainGlobe
            name by ``load_visualign``).
        interpolation_axis: Optional slicing/interpolation axis name (``"AP"`` /
            ``"ML"`` / ``"DV"``) to force. When ``None`` the axis inferred from the
            anchoring geometry by ``load_visualign`` is kept. An unknown value is
            ignored.

    Returns:
        A :class:`Project` with sections carrying real ``original_path`` /
        ``thumbnail_path`` / cached dimensions, a uniform ``working_scale``, and
        working-resolution control points. Ready to ``save()`` and generate
        thumbnails for.

    Raises:
        ValueError: If a section has no resolved image.
    """
    json_path = Path(json_path)
    thumbnails_dir = Path(project_dir) / "thumbnails"
    raw_sections = _read_slice_entries(json_path)
    original_paths = original_paths or {}

    project = load_visualign(json_path)
    if atlas_name:
        project.atlas = AtlasRef(name=atlas_name)
    if interpolation_axis in AXIS_NAME_TO_INDEX:
        project.interpolation_axis = interpolation_axis

    n = len(project.sections)
    working_sources = [
        str(original_paths.get(i) or registration_paths[i])
        for i in range(n)
        if i in registration_paths
    ]
    project.working_scale = compute_working_scale(working_sources)
    scale = project.working_scale

    for i, section in enumerate(project.sections):
        reg = registration_paths.get(i)
        orig = original_paths.get(i) or reg
        if orig is None:
            raise ValueError(f"No image resolved for section index {i} ({section.original_path!r})")

        section.original_path = str(orig)
        section.scene_index = 0
        section.thumbnail_path = str(thumbnails_dir / thumbnail_filename(orig, 0))

        raw = raw_sections[i] if i < len(raw_sections) else {}
        wr, hr = _registration_dims(raw, reg)
        wo, ho = image_dimensions(orig)
        ww, hw = max(1, round(wo * scale)), max(1, round(ho * scale))
        section.resolution_original_wh = (wo, ho)
        section.resolution_thumbnail_wh = (ww, hw)

        # Rescale imported control points from registration-image pixel space
        # (Wr, Hr) into VERSO's working grid (Ww, Hw). Anchoring is normalised and
        # needs no change.
        if section.warp.control_points and wr > 0 and hr > 0:
            fx, fy = ww / wr, hw / hr
            section.warp.control_points = [
                ControlPoint(
                    src_x=cp.src_x * fx,
                    src_y=cp.src_y * fy,
                    dst_x=cp.dst_x * fx,
                    dst_y=cp.dst_y * fy,
                    auto=cp.auto,
                )
                for cp in section.warp.control_points
            ]

    # Order by physical slice index and re-number ids to follow the series, so the
    # imported project reads like a New Project one (s001, s002, … in AP order).
    project.sections.sort(key=lambda s: (s.slice_index, s.id))
    for i, section in enumerate(project.sections):
        section.id = f"s{i + 1:03d}"

    return project
