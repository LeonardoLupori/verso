"""QuickNII / VisuAlign / DeepSlice JSON compatibility layer.

QuickNII format (per-section):
    {
        "filename": "IMG_0234.png",
        "nr": 1,
        "width": 1000,
        "height": 800,
        "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz]
    }

VisuAlign extends each section with a ``"markers"`` array of nonlinear warp
control points. Each marker is a 4-element pixel-coordinate array at the
section's working resolution:
    "markers": [[src_x, src_y, dst_x, dst_y], ...]

    src = (src_x, src_y) — position on the atlas overlay
    dst = (dst_x, dst_y) — matching position on the histological section

VERSO stores control points in working-resolution pixel coordinates too, so
markers pass through with no scaling. A legacy normalised dict form
``{"x", "y", "dx", "dy"}`` (with ``dst = (x + dx, y + dy)``) is still accepted on
load for backward compatibility but is never written.
"""

from __future__ import annotations

import contextlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verso.engine.anchoring import infer_interpolation_axis, is_anchored
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import AXIS_INDEX_TO_NAME, AtlasRef, Project, Section

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Atlas name mapping: brainglobe → VisuAlign target identifiers
# ---------------------------------------------------------------------------

# VisuAlign identifies atlases by the .cutlas bundle filename shipped with it.
# This maps BrainGlobe atlas names to those identifiers so exported JSON files
# can be opened by VisuAlign without a "labels.txt not found" error.
_BG_TO_VA_TARGET: dict[str, str] = {
    "allen_mouse_25um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_10um": "ABA_Mouse_CCFv3_2017_10um.cutlas",
    "allen_mouse_50um": "ABA_Mouse_CCFv3_2017_50um.cutlas",
}

# Volume dimensions [LR, AP, DV] voxels for each VisuAlign atlas target —
# written as "target-resolution" in the JSON (mirrors the working CC4B example).
_VA_TARGET_RESOLUTION: dict[str, list[float]] = {
    "ABA_Mouse_CCFv3_2017_25um.cutlas": [456.0, 528.0, 320.0],
    "ABA_Mouse_CCFv3_2017_10um.cutlas": [1140.0, 1320.0, 800.0],
    "ABA_Mouse_CCFv3_2017_50um.cutlas": [228.0, 264.0, 160.0],
}


# BrainGlobe annotation volume shape (AP, DV, LR) for known atlases — used to
# apply the inverse QuickNII→BrainGlobe convention conversion on load.
_BG_ATLAS_SHAPE: dict[str, tuple[int, int, int]] = {
    "allen_mouse_25um": (528, 320, 456),
    "allen_mouse_10um": (1320, 800, 1140),
    "allen_mouse_50um": (264, 160, 228),
}

# Inverse of _BG_TO_VA_TARGET: VisuAlign .cutlas identifier → BrainGlobe name.
_VA_TO_BG_TARGET: dict[str, str] = {va: bg for bg, va in _BG_TO_VA_TARGET.items()}


def _resolve_atlas_name(target: str) -> str:
    """Map a JSON ``target`` string to a BrainGlobe atlas name.

    QuickNII/DeepSlice exports name the atlas by its BrainGlobe id directly;
    VisuAlign exports name it by the ``.cutlas`` bundle identifier. Mapping the
    latter back is what lets a VisuAlign file load with a usable atlas name *and*
    have its anchoring converted from QuickNII to BrainGlobe convention (that
    conversion is keyed on :data:`_BG_ATLAS_SHAPE`). Unknown identifiers pass
    through unchanged so the caller can surface a warning.
    """
    return _VA_TO_BG_TARGET.get(target, target)


def _visualign_target(atlas_name: str) -> tuple[str, list[float] | None]:
    """Return (VisuAlign target string, target-resolution or None) for an atlas."""
    va_name = _BG_TO_VA_TARGET.get(atlas_name, atlas_name)
    resolution = _VA_TARGET_RESOLUTION.get(va_name)
    return va_name, resolution


# ---------------------------------------------------------------------------
# Low-level format helpers
# ---------------------------------------------------------------------------


def _parse_section_quicknii(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract QuickNII fields from a raw section dict."""
    return {
        "filename": raw.get("filename", ""),
        "nr": int(raw.get("nr", 0)),
        "anchoring": [float(x) for x in raw.get("anchoring", [0.0] * 9)],
    }


def _markers_to_control_points(
    markers: list,
    width: int,
    height: int,
) -> list[ControlPoint]:
    """Convert VisuAlign pixel-coordinate markers to VERSO control points.

    VisuAlign marker format: each marker is a 4-element array
        [src_x_px, src_y_px, dst_x_px, dst_y_px]
    where coordinates are in image pixels at the working resolution.

    Also accepts the legacy VERSO dict format ``{"x", "y", "dx", "dy"}``
    in normalised [0,1] coords for backward compatibility when loading old exports.

    Args:
        markers: List of 4-element arrays or legacy dicts.
        width: Section image width in pixels.
        height: Section image height in pixels.

    Returns:
        List of :class:`ControlPoint` with working-resolution pixel coordinates.
    """
    cps: list[ControlPoint] = []
    for m in markers:
        if isinstance(m, (list, tuple)) and len(m) == 4:
            sx, sy, dx, dy = float(m[0]), float(m[1]), float(m[2]), float(m[3])
            cps.append(ControlPoint(src_x=sx, src_y=sy, dst_x=dx, dst_y=dy))
        elif isinstance(m, dict):
            x, y = float(m["x"]), float(m["y"])
            ddx, ddy = float(m["dx"]), float(m["dy"])
            w = float(width) if width else 1.0
            h = float(height) if height else 1.0
            cps.append(
                ControlPoint(
                    src_x=x * w,
                    src_y=y * h,
                    dst_x=(x + ddx) * w,
                    dst_y=(y + ddy) * h,
                )
            )
    return cps


def _control_points_to_markers(
    control_points: list[ControlPoint],
) -> list[list[float]]:
    """Convert VERSO control points to VisuAlign pixel-coordinate markers.

    VisuAlign stores markers as 4-element arrays:
        [src_x_px, src_y_px, dst_x_px, dst_y_px]
    in image pixels at the working resolution.  Control points are already
    stored in working-resolution pixel coordinates, so no conversion is needed.

    Args:
        control_points: List of :class:`ControlPoint` with pixel coordinates.

    Returns:
        List of ``[src_x_px, src_y_px, dst_x_px, dst_y_px]`` arrays.
    """
    markers: list[list[float]] = []
    for cp in control_points:
        markers.append(
            [
                round(cp.src_x, 6),
                round(cp.src_y, 6),
                round(cp.dst_x, 6),
                round(cp.dst_y, 6),
            ]
        )
    return markers


# ---------------------------------------------------------------------------
# Document reading (JSON + QuickNII XML)
# ---------------------------------------------------------------------------

_ANCHORING_KEYS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")


def _parse_anchoring_query(text: str) -> list[float]:
    """Parse a QuickNII XML anchoring string (``ox=..&oy=..&...``) into 9 floats.

    Components missing or unparseable default to ``0.0`` so a bare/partial
    anchoring degrades to "unaligned" rather than raising.
    """
    values: dict[str, float] = {}
    for pair in text.split("&"):
        key, _, val = pair.partition("=")
        key = key.strip()
        if key in _ANCHORING_KEYS and val.strip():
            with contextlib.suppress(ValueError):
                values[key] = float(val)
    return [values.get(k, 0.0) for k in _ANCHORING_KEYS]


def parse_quicknii_xml(path: Path) -> dict[str, Any]:
    """Parse a QuickNII native ``.xml`` alignment into the JSON document shape.

    QuickNII stores one ``<slice>`` per section under a ``<series>`` root, with the
    9-component plane packed into the ``anchoring`` attribute as an
    ``ox=..&oy=..&...`` query string (bare ``<slice>`` entries are unaligned; the
    XML entity ``&amp;`` is decoded by the parser). Returns
    ``{"name", "target", "slices": [...]}`` with each slice carrying
    ``filename``/``nr``/``width``/``height``/``anchoring`` (a 9-float list) — i.e.
    exactly what :func:`load_quicknii` consumes, so XML and JSON share one path.
    QuickNII XML has no warp-marker equivalent, so no ``markers`` are produced.
    """
    root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
    slices: list[dict[str, Any]] = []
    for el in root.findall("slice"):
        entry: dict[str, Any] = {
            "filename": el.get("filename", ""),
            "nr": int(el.get("nr") or 0),
            "width": int(el.get("width") or 0),
            "height": int(el.get("height") or 0),
        }
        anchoring = el.get("anchoring")
        if anchoring:
            entry["anchoring"] = _parse_anchoring_query(anchoring)
        slices.append(entry)
    return {
        "name": root.get("name", Path(path).stem),
        "target": root.get("target", ""),
        "slices": slices,
    }


def read_quint_document(path: Path) -> dict[str, Any]:
    """Read a QuickNII/VisuAlign/DeepSlice alignment file as a document dict.

    Dispatches on extension: ``.xml`` → :func:`parse_quicknii_xml`, anything else
    → JSON. This is the single entry point every loader uses so QuickNII XML and
    the JSON variants are accepted uniformly.
    """
    path = Path(path)
    if path.suffix.lower() == ".xml":
        return parse_quicknii_xml(path)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Public load functions
# ---------------------------------------------------------------------------

# NOTE — coordinate space of exports
# Exports are written entirely in display space — the same space VERSO stores
# its anchoring and control points in. Anchoring uses the saved
# ``stored_anchoring``; markers are the stored control points as-is. No flip is
# applied to either, so save → load is an identity round-trip regardless of a
# section's flip flag.
# Horizontal/vertical flips are represented outside the alignment (e.g. baked
# into the exported images), never by mirroring the saved coordinates.


def load_quicknii(path: Path, atlas_name: str = "allen_mouse_25um") -> Project:
    """Load a QuickNII JSON file into a VERSO :class:`Project`.

    Supports both ``"slices"`` (native QuickNII) and ``"sections"`` (VERSO)
    as the section list key.

    Args:
        path: Path to the QuickNII ``*.json`` file.
        atlas_name: Fallback atlas name if not present in the JSON.

    Returns:
        A :class:`Project` with one :class:`Section` per entry in the JSON.
    """
    data = read_quint_document(Path(path))
    atlas_name = _resolve_atlas_name(data.get("target", atlas_name))
    project_name = data.get("name", Path(path).stem)

    # QuickNII/VisuAlign use "slices"; accept "sections" for forward compatibility
    raw_sections = data.get("slices") or data.get("sections", [])

    # QuickNII files store anchoring in QuickNII convention; convert to the
    # BrainGlobe (VERSO internal) convention so atlas slicing is correct.
    # _to_quicknii_convention is self-inverse, so it also converts QN→BG.
    bg_shape = _BG_ATLAS_SHAPE.get(atlas_name)

    sections: list[Section] = []
    anchorings: list[list[float]] = []
    for i, raw in enumerate(raw_sections):
        parsed = _parse_section_quicknii(raw)
        anchoring = parsed["anchoring"]
        if bg_shape is not None and any(anchoring):
            anchoring = _to_quicknii_convention(anchoring, bg_shape)
        anchorings.append(anchoring)
        status = AlignmentStatus.COMPLETE if any(anchoring) else AlignmentStatus.NOT_STARTED
        # A COMPLETE import is a saved plane; __post_init__ mirrors it into
        # stored_anchoring (the persisted source of truth).
        alignment = Alignment(
            current_anchoring=anchoring,
            status=status,
        )
        section = Section(
            id=f"s{i + 1:03d}",
            slice_index=parsed["nr"],
            original_path=parsed["filename"],
            thumbnail_path="",
            alignment=alignment,
        )
        sections.append(section)

    # QuickNII/VisuAlign do not record the cutting plane, but the anchoring
    # geometry does — recover it so the imported project has the right slicing
    # orientation (and interpolation axis) instead of always defaulting to coronal.
    interpolation_axis = AXIS_INDEX_TO_NAME[infer_interpolation_axis(anchorings)]

    return Project(
        name=project_name,
        atlas=AtlasRef(name=atlas_name),
        sections=sections,
        interpolation_axis=interpolation_axis,
    )


def load_visualign(
    path: Path,
    atlas_name: str = "allen_mouse_25um",
) -> Project:
    """Load a VisuAlign JSON file (with control points) into a VERSO :class:`Project`.

    Markers are 4-element ``[src_x, src_y, dst_x, dst_y]`` pixel arrays at the
    section's working resolution and map directly onto VERSO control points (a
    legacy normalised-dict form is also accepted; see
    :func:`_markers_to_control_points`).

    Args:
        path: Path to the VisuAlign ``*.json`` file.
        atlas_name: Fallback atlas name if not present in the JSON.

    Returns:
        A :class:`Project` including :class:`WarpState` for each section.
    """
    project = load_quicknii(path, atlas_name=atlas_name)

    data = read_quint_document(Path(path))
    raw_sections = data.get("slices") or data.get("sections", [])
    for section, raw in zip(project.sections, raw_sections, strict=False):
        markers = raw.get("markers", [])
        if markers:
            w = int(raw.get("width", 0))
            h = int(raw.get("height", 0))
            cps = _markers_to_control_points(markers, w, h)
            section.warp = WarpState(
                control_points=cps,
                status=AlignmentStatus.COMPLETE,
            )

    return project


def load_deepslice(path: Path, atlas_name: str = "allen_mouse_25um") -> Project:
    """Load DeepSlice output JSON as initial registration.

    DeepSlice output is QuickNII-compatible, so this is an alias for
    :func:`load_quicknii` with all alignment statuses set to ``IN_PROGRESS``
    (since DeepSlice results typically still need manual refinement).
    """
    project = load_quicknii(path, atlas_name=atlas_name)
    for section in project.sections:
        if section.alignment.status == AlignmentStatus.COMPLETE:
            section.alignment.status = AlignmentStatus.IN_PROGRESS
            section.alignment.source = "deepslice"
    return project


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flip_anchoring(anchoring: list[float]) -> list[float]:
    """Convert anchoring from horizontally-flipped display space to original image space.

    When the section is displayed flipped, the user aligns against s_display = 1 - s_original.
    The equivalent anchoring for the unflipped image is:
        o' = o + u,  u' = -u,  v' = v
    """
    from verso.engine.anchoring import flip_anchoring_horizontal

    return flip_anchoring_horizontal(anchoring)


def _to_quicknii_convention(
    anchoring: list[float],
    atlas_shape: tuple[int, int, int],
) -> list[float]:
    """Convert anchoring from VERSO/brainglobe to QuickNII atlas convention.

    VERSO stores cut planes in BrainGlobe array order, where AP and DV indices
    increase in the opposite anatomical direction to QuickNII's display
    convention. LR is shared.

    The conventions are related by:
        oy_qn  = (ap_max - 1) - oy_bg
        oz_qn  = (dv_max - 1) - oz_bg
        uy_qn  = -uy_bg
        uz_qn  = -uz_bg
        vy_qn  = -vy_bg
        vz_qn  = -vz_bg

    This mirrors both the AP and DV axes while preserving LR.

    The origin offset is ``ap_max - 1`` (not ``ap_max``): the QuickNII/VisuAlign
    atlas volume is the BrainGlobe annotation with its AP/DV axes *array-reversed*
    (``annotation[::-1, ::-1, :]``), i.e. index ``i`` maps to ``N - 1 - i``.
    Flipping the continuous origin about ``N`` instead of ``N - 1`` shifts the
    sampled plane by exactly one voxel in AP and DV — verified against the stock
    ``ABA_Mouse_CCFv3_2017_25um.cutlas``: with ``N - 1`` the sampled labels match
    VERSO's plane voxel-for-voxel (100%), with ``N`` they disagree (~6.5%).

    This function is its own inverse (applying it twice restores the original).

    Args:
        anchoring: 9-element anchoring in brainglobe (VERSO internal) convention.
        atlas_shape: BrainGlobe annotation shape ``(AP, DV, LR)``.

    Returns:
        9-element anchoring in QuickNII convention, ready to write to file.
    """
    ap_max, dv_max, _lr_max = atlas_shape
    ox, oy, oz = anchoring[0], anchoring[1], anchoring[2]
    ux, uy, uz = anchoring[3], anchoring[4], anchoring[5]
    vx, vy, vz = anchoring[6], anchoring[7], anchoring[8]
    return [ox, (ap_max - 1) - oy, (dv_max - 1) - oz, ux, -uy, -uz, vx, -vy, -vz]


def _export_image_filename(section) -> str:
    """Return the PNG filename for a section in a QuickNII/VisuAlign export.

    Strips all compound extensions (e.g. ``.ome.tif`` → bare stem) so the
    exported name matches the original image name with a ``.png`` extension.
    The path is relative to the exported JSON/XML and points into a
    ``thumbnails/`` subfolder (forward-slash separator, cross-platform).
    """
    p = Path(section.original_path)
    while p.suffix:
        p = p.with_suffix("")
    return f"thumbnails/{p.name}-thumb.png"


def write_section_pngs(
    project: Project,
    output_dir: Path,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Write RGB PNG copies of each section's working image into *output_dir*.

    Only sections whose PNG is not already present are written.  Uses the
    project channel specs to composite multichannel images to RGB.

    Args:
        project: VERSO project whose sections will be exported.
        output_dir: Folder that will receive the PNG files.
        on_progress: Called as ``(done, total, image_name)`` before each section
            (including ones skipped as already present), for driving a progress
            bar. Optional.
    """
    import numpy as np
    from PIL import Image

    from verso.engine.io.image_io import ensure_working_copy
    from verso.engine.preprocessing import composite_channels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(project.sections)
    for done, section in enumerate(project.sections):
        png_path = output_dir / _export_image_filename(section)
        if on_progress is not None:
            on_progress(done, total, png_path.name)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        if png_path.exists():
            continue
        img = ensure_working_copy(section, project.working_scale)
        if img is None:
            continue
        if project.channels:
            rgb = composite_channels(img, project.channels)
        else:
            plane = img[:, :, 0] if img.ndim == 3 else img
            rgb = np.stack([plane, plane, plane], axis=-1)
        Image.fromarray(rgb).save(str(png_path))


# ---------------------------------------------------------------------------
# Public save functions
# ---------------------------------------------------------------------------


def _export_anchoring(
    anchoring: list[float],
    atlas_shape: tuple[int, int, int] | None,
) -> list[float]:
    """Apply export-time anchoring transforms.

    ``anchoring`` must already be in original image space (stored_anchoring
    invariant). Only the atlas-axis convention conversion is applied.

    Args:
        anchoring: Anchoring in original image space (brainglobe convention).
        atlas_shape: Atlas dimensions ``(AP, DV, LR)``. Pass ``None`` to skip
            atlas convention conversion.

    Returns:
        Anchoring ready to write into a QuickNII/VisuAlign file.
    """
    if atlas_shape is not None:
        return _to_quicknii_convention(anchoring, atlas_shape)
    return list(anchoring)


def save_quicknii_xml(
    project: Project,
    path: Path,
    atlas_shape: tuple[int, int, int] | None = None,
) -> None:
    """Write alignment data in QuickNII native XML format.

    Only sections with a stored (COMPLETE) alignment receive an anchoring.
    Un-stored sections appear as bare ``<slice>`` entries so that QuickNII
    will interpolate them itself.

    Args:
        project: VERSO project to export.
        path: Destination ``*.xml`` path.
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the
            ``target-resolution`` attribute and the DV-convention conversion.
            When *None*, inferred from the project atlas name using the built-in
            lookup table (matching :func:`save_quicknii` / :func:`save_visualign`).
            If it cannot be inferred, the axis conversion is skipped.
    """
    if atlas_shape is None and project.atlas:
        atlas_shape = _BG_ATLAS_SHAPE.get(project.atlas.name)
    lines: list[str] = ["<?xml version='1.0' encoding='UTF-8'?>"]
    res_attr = ""
    if atlas_shape is not None:
        ap, dv, lr = atlas_shape
        res_attr = f" target-resolution='{ap} {dv} {lr}'"
    atlas_name = project.atlas.name if project.atlas else ""
    lines.append(f"<series name='{project.name}' target='{atlas_name}'{res_attr}>")

    prefixes = [
        "' anchoring='ox=",
        "&amp;oy=",
        "&amp;oz=",
        "&amp;ux=",
        "&amp;uy=",
        "&amp;uz=",
        "&amp;vx=",
        "&amp;vy=",
        "&amp;vz=",
    ]
    for section in project.sections:
        w, h = section.resolution_thumbnail_wh
        filename = _export_image_filename(section)
        line = (
            f"    <slice filename='{filename}' nr='{section.slice_index}' width='{w}' height='{h}"
        )
        stored = section.alignment.stored_anchoring
        if section.alignment.status == AlignmentStatus.COMPLETE and is_anchored(stored):
            a = _export_anchoring(stored, atlas_shape)
            for prefix, val in zip(prefixes, [round(v, 4) for v in a], strict=False):
                line += f"{prefix}{val}"
        line += "'/>"
        lines.append(line)

    lines.append("</series>")
    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def save_quicknii(
    project: Project,
    path: Path,
    atlas_shape: tuple[int, int, int] | None = None,
) -> None:
    """Write alignment data in QuickNII-compatible JSON format.

    Only sections with a stored (COMPLETE) alignment receive an anchoring.

    Args:
        project: VERSO project to export.
        path: Destination ``*.json`` path.
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the BrainGlobe→
            QuickNII axis convention conversion. When *None*, inferred from the
            project atlas name using the built-in lookup table.
    """
    if atlas_shape is None and project.atlas:
        atlas_shape = _BG_ATLAS_SHAPE.get(project.atlas.name)

    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = section.resolution_thumbnail_wh
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section),
            "nr": section.slice_index,
            "width": w,
            "height": h,
        }
        stored = section.alignment.stored_anchoring
        if section.alignment.status == AlignmentStatus.COMPLETE and is_anchored(stored):
            entry["anchoring"] = [round(v, 4) for v in _export_anchoring(stored, atlas_shape)]
        slices_out.append(entry)

    data: dict[str, Any] = {
        "name": project.name,
        "target": project.atlas.name if project.atlas else "",
        "slices": slices_out,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_visualign(
    project: Project,
    path: Path,
    atlas_shape: tuple[int, int, int] | None = None,
) -> None:
    """Write alignment + warp data in VisuAlign-compatible JSON format.

    Only sections with a stored (COMPLETE) alignment receive an anchoring.
    Control point y-coordinates do not need conversion — the normalised t
    coordinate has the same meaning (0=top, 1=bottom) in both conventions.

    Args:
        project: VERSO project to export.
        path: Destination ``*.json`` path.
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the BrainGlobe→
            QuickNII axis convention conversion. When *None*, inferred from the
            project atlas name using the built-in lookup table.
    """
    if atlas_shape is None and project.atlas:
        atlas_shape = _BG_ATLAS_SHAPE.get(project.atlas.name)

    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = section.resolution_thumbnail_wh
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section),
            "nr": section.slice_index,
            "width": w,
            "height": h,
        }
        stored = section.alignment.stored_anchoring
        if section.alignment.status == AlignmentStatus.COMPLETE and is_anchored(stored):
            entry["anchoring"] = [round(v, 4) for v in _export_anchoring(stored, atlas_shape)]
        if section.warp.control_points:
            # Markers are written in display space — the same space VERSO stores
            # control points in — with no flip applied. Flips are represented
            # outside the alignment (e.g. baked into the exported images), never
            # by mirroring the saved coordinates.
            entry["markers"] = _control_points_to_markers(section.warp.control_points)
        slices_out.append(entry)

    va_target, va_resolution = _visualign_target(project.atlas.name if project.atlas else "")
    data: dict[str, Any] = {"name": project.name, "target": va_target}
    if va_resolution is not None:
        data["target-resolution"] = va_resolution
    data["slices"] = slices_out
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
