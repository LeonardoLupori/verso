"""Region quantification — public entry points.

Quantify raw image pixels and user annotations of an aligned VERSO project against
the atlas. Three analyses, each usable from the GUI or as a scripting call that
reads only ``project-verso.json``:

* :func:`quantify_intensity` — per-region, per-channel mean + total pixel value.
* :func:`quantify_area` — the same, restricted to an area annotation's footprint.
* :func:`quantify_dots` — per-dot atlas coordinates + per-region counts/density.

All return **lists of dicts** (``pd.DataFrame(rows)`` for pandas users). With
``QuantifyOptions.out_dir`` set they are also written as CSVs under
``exports/quantification_<ts>/``. See ``.claude/quantification.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from verso.engine.quantification.aggregate import (
    LEVELS,
    RegionAggregator,
    add_dot_aggregation_columns,
    regroup_dots_region,
    regroup_intensity,
)
from verso.engine.quantification.area import area_scope
from verso.engine.quantification.dots import (
    add_region_counts,
    process_section_dots,
)
from verso.engine.quantification.intensity import IntensityAccumulator, match_to_raw
from verso.engine.quantification.region_map import region_map
from verso.engine.quantification.tables import dots_region_rows, intensity_rows

if TYPE_CHECKING:
    from collections.abc import Callable

    from verso.engine.atlas import AtlasVolume
    from verso.engine.model.annotation import PointSeries
    from verso.engine.model.project import Project, Section
    from verso.engine.registration import VersoRegistration

__all__ = [
    "QuantificationError",
    "QuantifyOptions",
    "check_originals_reachable",
    "quantify_area",
    "quantify_dots",
    "quantify_intensity",
]


class QuantificationError(RuntimeError):
    """A precondition failed (unreachable images, missing alignment/CPs/masks)."""


@dataclass
class QuantifyOptions:
    """Options shared by the three quantifications.

    Attributes:
        include_unwarped_affine: If False (default), abort when any section lacks
            warp control points; if True, such sections are mapped affine-only.
        include_unmasked_wholeframe: If False (default), abort when any section
            lacks a slice mask; if True, unmasked sections quantify the whole frame.
        channels: Channel names to quantify (intensity/area). ``None`` = all.
        aggregate: Subset of ``("mid", "coarse")`` — extra aggregated outputs.
        per_slice: If True, produce one independent output per section instead of
            a pooled project-wide result.
        split_hemispheres: If True, split every region into left/right buckets
            (a ``hemisphere`` column, ``l``/``r``, or ``none`` for out-of-atlas
            background). Hemisphere is sampled per-pixel from the atlas through the
            same warp as the region labels. Regions present on only one side yield a
            single row; no assumption of symmetry or of the interpolation axis.
        out_dir: Base directory to write CSVs under (a timestamped
            ``quantification_<ts>/`` folder is created inside). ``None`` = return
            records only, write nothing.
    """

    include_unwarped_affine: bool = False
    include_unmasked_wholeframe: bool = False
    channels: list[str] | None = None
    aggregate: tuple[str, ...] = ()
    per_slice: bool = False
    split_hemispheres: bool = False
    out_dir: str | Path | None = None

    def __post_init__(self) -> None:
        bad = [a for a in self.aggregate if a not in LEVELS]
        if bad:
            raise ValueError(f"aggregate must be a subset of {LEVELS}, got {bad}")


# ---------------------------------------------------------------------------
# Setup / preconditions
# ---------------------------------------------------------------------------


def _prepare(
    project: str | Path | Project,
    project_dir: str | Path | None,
    atlas: AtlasVolume | None,
) -> tuple[VersoRegistration, Project, AtlasVolume, Path]:
    """Resolve inputs to ``(reg, project, atlas, project_dir)``."""
    from verso.engine.atlas import AtlasVolume as _AtlasVolume
    from verso.engine.model.project import Project as _Project
    from verso.engine.registration import VersoRegistration

    if isinstance(project, (str, Path)):
        path = Path(project)
        proj = _Project.load(path)
        pdir = path.parent if project_dir is None else Path(project_dir)
    else:
        proj = project
        if project_dir is None:
            raise ValueError("project_dir is required when passing a Project object")
        pdir = Path(project_dir)

    reg = VersoRegistration.from_project(proj)
    atl = atlas if atlas is not None else _AtlasVolume(proj.atlas.name)
    return reg, proj, atl, pdir


def _resolve_original(section: Section, project_dir: Path) -> Path:
    """Resolve a section's original image path (relative to the project dir)."""
    p = Path(section.original_path)
    return p if p.is_absolute() else project_dir / p


def check_originals_reachable(project: Project, project_dir: str | Path) -> list[str]:
    """Return the basenames of sections whose original image is not on disk."""
    pdir = Path(project_dir)
    return [
        Path(s.original_path).name
        for s in project.sections
        if not _resolve_original(s, pdir).exists()
    ]


def _check_preconditions(project: Project, project_dir: Path, options: QuantifyOptions) -> None:
    """Raise :class:`QuantificationError` if the run cannot proceed (plan §2/§8)."""
    if not project.sections:
        raise QuantificationError("The project has no sections.")

    missing = check_originals_reachable(project, project_dir)
    if missing:
        raise QuantificationError(
            "Some original images are not reachable; fix the paths and retry:\n  "
            + "\n  ".join(missing)
        )

    unaligned = [s for s in project.sections if not s.alignment.is_anchored]
    if unaligned:
        raise QuantificationError(
            "These sections have no alignment (required):\n  "
            + "\n  ".join(Path(s.original_path).name for s in unaligned)
        )

    if not options.include_unwarped_affine:
        no_cp = [s for s in project.sections if not s.warp.control_points]
        if no_cp:
            raise QuantificationError(
                "These sections have no warp control points (tick 'use sections "
                "without control points' to map them affine-only):\n  "
                + "\n  ".join(Path(s.original_path).name for s in no_cp)
            )

    if not options.include_unmasked_wholeframe:
        no_mask = [
            s
            for s in project.sections
            if not (
                s.preprocessing.slice_mask_path and Path(s.preprocessing.slice_mask_path).exists()
            )
        ]
        if no_mask:
            raise QuantificationError(
                "These sections have no slice mask (tick 'use sections without a "
                "mask' to quantify the whole frame):\n  "
                + "\n  ".join(Path(s.original_path).name for s in no_mask)
            )


def _channel_plan(
    project: Project, selected: list[str] | None, sample_path: Path
) -> tuple[list[int], list[str]]:
    """Return ``(channel_indices, channel_names)`` for the selected channels."""
    names_all = [c.name for c in project.channels]
    if not names_all:
        from verso.engine.io.image_io import probe_channels

        names_all = probe_channels(sample_path)
    if selected is None:
        return list(range(len(names_all))), list(names_all)
    idxs: list[int] = []
    names: list[str] = []
    for n in selected:
        if n not in names_all:
            raise QuantificationError(f"Unknown channel {n!r}; available: {names_all}")
        idxs.append(names_all.index(n))
        names.append(n)
    return idxs, names


def _aggregator(atlas: AtlasVolume, options: QuantifyOptions) -> RegionAggregator | None:
    return RegionAggregator(atlas) if options.aggregate else None


def _make_ticker(
    on_progress: Callable[[int, int, str], None] | None, total: int
) -> Callable[[Section], None] | None:
    """Wrap ``on_progress`` as a per-section callback counting across the run.

    The pooled and per-slice paths both funnel every section through one of the
    ``_*_unit`` helpers, so a single counter here covers both. The returned
    callback reports ``(sections done, total, image name)`` *before* each
    section is processed; ``None`` in gives ``None`` out (no bookkeeping).
    """
    if on_progress is None:
        return None
    done = 0

    def tick(section: Section) -> None:
        nonlocal done
        on_progress(done, total, Path(section.original_path).name)
        done += 1

    return tick


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_pooled(options: QuantifyOptions, files: dict[str, list[dict]]) -> None:
    if options.out_dir is None:
        return
    from verso.engine.io.quant_export import make_output_dir, write_result_tables

    root = make_output_dir(options.out_dir)
    write_result_tables(root, files)


def _write_per_slice(options: QuantifyOptions, file_map: dict[str, dict[str, list[dict]]]) -> None:
    if options.out_dir is None:
        return
    from verso.engine.io.quant_export import make_output_dir, write_result_tables

    root = make_output_dir(options.out_dir)
    for slug, files in file_map.items():
        write_result_tables(root / slug, files)


# ---------------------------------------------------------------------------
# Pixel analyses (intensity + area)
# ---------------------------------------------------------------------------


def _pixel_unit(
    reg: VersoRegistration,
    atlas: AtlasVolume,
    sections: list[Section],
    project_dir: Path,
    options: QuantifyOptions,
    idxs: list[int],
    names: list[str],
    agg: RegionAggregator | None,
    scope_fn,
    file_prefix: str,
    tick: Callable[[Section], None] | None = None,
) -> tuple[dict, dict]:
    """Accumulate intensity for ``sections`` and build (return_dict, file_tables)."""
    from verso.engine.io.image_io import load_full_res_raw

    acc = IntensityAccumulator(len(idxs))
    for section in sections:
        if tick is not None:
            tick(section)
        labels, slice_sc, hemi = region_map(
            reg, atlas, section, split_hemispheres=options.split_hemispheres
        )
        scope = scope_fn(section, slice_sc)
        raw = load_full_res_raw(_resolve_original(section, project_dir), section.scene_index)
        labels, scope, hemi = match_to_raw(labels, scope, raw.shape[:2], hemi)
        acc.add(labels, scope, raw[..., idxs], hemi)

    totals = acc.totals_as_lists()
    regions = intensity_rows(acc.counts, totals, atlas, names)
    ret: dict = {"regions": regions}
    files: dict = {file_prefix: regions}
    for level in options.aggregate:
        assert agg is not None
        grouped = regroup_intensity(acc.counts, totals, agg, level, atlas, names)
        ret[f"regions_{level}"] = grouped
        files[f"{file_prefix}_{level}"] = grouped
    return ret, files


def _run_pixel_analysis(
    project, project_dir, atlas, options, scope_factory, file_prefix, on_progress=None
) -> dict:
    """Shared driver for intensity/area (``scope_factory`` builds the per-section scope)."""
    reg, proj, atl, pdir = _prepare(project, project_dir, atlas)
    _check_preconditions(proj, pdir, options)
    sample = _resolve_original(proj.sections[0], pdir)
    idxs, names = _channel_plan(proj, options.channels, sample)
    agg = _aggregator(atl, options)
    scope_fn = scope_factory(proj, pdir)
    tick = _make_ticker(on_progress, len(proj.sections))

    if options.per_slice:
        from verso.engine.io.quant_export import slug_for_section

        result: dict = {}
        file_map: dict[str, dict] = {}
        used: set[str] = set()
        for section in proj.sections:
            ret, files = _pixel_unit(
                reg, atl, [section], pdir, options, idxs, names, agg, scope_fn, file_prefix, tick
            )
            slug = slug_for_section(section, used)
            result[slug] = ret
            file_map[slug] = files
        _write_per_slice(options, file_map)
        return result

    ret, files = _pixel_unit(
        reg, atl, proj.sections, pdir, options, idxs, names, agg, scope_fn, file_prefix, tick
    )
    _write_pooled(options, files)
    return ret


def quantify_intensity(
    project: str | Path | Project,
    *,
    project_dir: str | Path | None = None,
    atlas: AtlasVolume | None = None,
    options: QuantifyOptions | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Per-region, per-channel intensity (mean + total) within the slice mask.

    Args:
        project: Path to ``project-verso.json`` or an in-memory :class:`Project`.
        project_dir: Project directory; required only when ``project`` is a
            :class:`Project` (defaults to the JSON's parent otherwise).
        atlas: Pre-loaded :class:`AtlasVolume` to reuse (else built from the project).
        options: :class:`QuantifyOptions`.
        on_progress: Called as ``(done, total, image_name)`` before each section,
            for driving a progress bar. Optional.

    Returns:
        Pooled: ``{"regions": [...], "regions_mid": [...], "regions_coarse": [...]}``
        (aggregation keys only for requested levels). Per-slice: a dict keyed by the
        slugified unique image name, each value the same shape.
    """
    options = options or QuantifyOptions()

    def scope_factory(_proj, _pdir):
        return lambda _section, slice_sc: slice_sc

    return _run_pixel_analysis(
        project, project_dir, atlas, options, scope_factory, "intensity", on_progress
    )


def quantify_area(
    project: str | Path | Project,
    annotation: str,
    *,
    project_dir: str | Path | None = None,
    atlas: AtlasVolume | None = None,
    options: QuantifyOptions | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Per-region intensity restricted to an area annotation (``slice ∧ area``).

    Args:
        annotation: Title of the :class:`AreaAnnotation` to quantify.
        (other args as :func:`quantify_intensity`.)

    Returns:
        Same shape as :func:`quantify_intensity`.
    """
    options = options or QuantifyOptions()

    def scope_factory(proj, pdir):
        area = _load_annotation(pdir, annotation, kind="area")

        def scope_fn(section, slice_sc):
            return area_scope(section, slice_sc, area)

        return scope_fn

    return _run_pixel_analysis(
        project, project_dir, atlas, options, scope_factory, "area", on_progress
    )


# ---------------------------------------------------------------------------
# Dots analysis
# ---------------------------------------------------------------------------


def _load_annotation(project_dir: Path, title: str, *, kind: str):
    """Load a single annotation by title (``kind`` = ``"area"`` or ``"points"``)."""
    from verso.engine.io.annotation_io import load_annotations
    from verso.engine.model.annotation import AreaAnnotation, PointSeries

    want = AreaAnnotation if kind == "area" else PointSeries
    matches = [a for a in load_annotations(project_dir) if isinstance(a, want) and a.title == title]
    if not matches:
        available = [a.title for a in load_annotations(project_dir) if isinstance(a, want)]
        raise QuantificationError(f"No {kind} annotation named {title!r}. Available: {available}")
    return matches[0]


def _dots_unit(
    reg: VersoRegistration,
    atlas: AtlasVolume,
    sections: list[Section],
    project_dir: Path,
    options: QuantifyOptions,
    series: PointSeries,
    coords_by_image: dict,
    agg: RegionAggregator | None,
    intensity_idxs: list[int],
    all_names: list[str],
    dot_diameter_px: float,
    tick: Callable[[Section], None] | None = None,
) -> tuple[dict, dict]:
    """Quantify dots for ``sections`` and build (return_dict, file_tables)."""
    from verso.engine.io.image_io import load_full_res_raw

    counts: dict[tuple[int, int | None], int] = {}
    n_dots: dict[tuple[int, int | None], int] = {}
    per_dot: list[dict] = []

    for section in sections:
        if tick is not None:
            tick(section)
        labels, scope, hemi = region_map(
            reg, atlas, section, split_hemispheres=options.split_hemispheres
        )
        raw = None
        if intensity_idxs:
            raw = load_full_res_raw(_resolve_original(section, project_dir), section.scene_index)
            labels, scope, hemi = match_to_raw(labels, scope, raw.shape[:2], hemi)
        add_region_counts(counts, labels, scope, hemi)

        key = section.image_key.lower()
        pts = coords_by_image.get(key)
        if pts is None:
            continue
        xy = np.column_stack(pts)
        recs, nd = process_section_dots(
            reg,
            atlas,
            section,
            xy,
            labels,
            scope,
            hemi=hemi,
            raw=raw,
            intensity_channels=intensity_idxs,
            channel_names=all_names,
            dot_diameter_px=dot_diameter_px,
        )
        per_dot.extend(recs)
        for r, c in nd.items():
            n_dots[r] = n_dots.get(r, 0) + c

    if options.aggregate:
        assert agg is not None
        add_dot_aggregation_columns(per_dot, agg, atlas, options.aggregate)

    regions = dots_region_rows(counts, n_dots, atlas)
    ret: dict = {"dots": per_dot, "regions": regions}
    files: dict = {"dots": per_dot, "dots_regions": regions}
    for level in options.aggregate:
        assert agg is not None
        grouped = regroup_dots_region(counts, n_dots, agg, level, atlas)
        ret[f"regions_{level}"] = grouped
        files[f"dots_regions_{level}"] = grouped
    return ret, files


def quantify_dots(
    project: str | Path | Project,
    annotation: str,
    *,
    intensity_channels: list[str] | None = None,
    dot_diameter_px: float = 1.0,
    project_dir: str | Path | None = None,
    atlas: AtlasVolume | None = None,
    options: QuantifyOptions | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Quantify a point series: per-dot atlas table + per-region counts/density.

    Args:
        annotation: Title of the :class:`PointSeries` to quantify.
        intensity_channels: Channel names to add a ``mean_intensity`` per dot for
            (a disk of ``dot_diameter_px``); ``None`` = no intensity.
        dot_diameter_px: Disk diameter in original pixels (default 1 = single pixel).
        (other args as :func:`quantify_intensity`.)

    Returns:
        Pooled: ``{"dots": [...], "regions": [...], "regions_mid": [...], …}``.
        Per-slice: keyed by slugified unique image name.
    """
    options = options or QuantifyOptions()
    reg, proj, atl, pdir = _prepare(project, project_dir, atlas)
    _check_preconditions(proj, pdir, options)

    from verso.engine.annotations import point_coords_by_image

    series = _load_annotation(pdir, annotation, kind="points")
    coords_by_image = point_coords_by_image(series)

    sample = _resolve_original(proj.sections[0], pdir)
    if intensity_channels:
        intensity_idxs, _ = _channel_plan(proj, intensity_channels, sample)
    else:
        intensity_idxs = []
    _, all_names = _channel_plan(proj, None, sample)
    agg = _aggregator(atl, options)
    tick = _make_ticker(on_progress, len(proj.sections))

    if options.per_slice:
        from verso.engine.io.quant_export import slug_for_section

        result: dict = {}
        file_map: dict[str, dict] = {}
        used: set[str] = set()
        for section in proj.sections:
            ret, files = _dots_unit(
                reg,
                atl,
                [section],
                pdir,
                options,
                series,
                coords_by_image,
                agg,
                intensity_idxs,
                all_names,
                dot_diameter_px,
                tick,
            )
            slug = slug_for_section(section, used)
            result[slug] = ret
            file_map[slug] = files
        _write_per_slice(options, file_map)
        return result

    ret, files = _dots_unit(
        reg,
        atl,
        proj.sections,
        pdir,
        options,
        series,
        coords_by_image,
        agg,
        intensity_idxs,
        all_names,
        dot_diameter_px,
        tick,
    )
    _write_pooled(options, files)
    return ret
