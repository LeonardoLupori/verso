"""QuickNII / VisuAlign / DeepSlice JSON compatibility layer.

QuickNII format (per-section):
    {
        "filename": "IMG_0234.png",
        "nr": 1,
        "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz]
    }

VisuAlign extends each section with a ``"markers"`` array:
    {
        ...QuickNII fields...
        "markers": [
            {"x": 0.5, "y": 0.3, "dx": 0.02, "dy": -0.01},
            ...
        ]
    }

Marker coordinates:
    x, y  — normalised section position in [0, 1]²  (left=0, top=0)
    dx, dy — displacement from (x, y) to the matching landmark on the
              histological section, also in normalised section space.

    In warp terms:
        src = (x, y)           — position on the atlas overlay
        dst = (x + dx, y + dy) — matching position on the section

VERSO stores control points in working-resolution pixel coordinates.  The
conversion functions here handle the normalisation/denormalisation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.project import AtlasRef, Project, Section

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
    markers: list[dict[str, float]],
) -> list[ControlPoint]:
    """Convert VisuAlign normalised markers to VERSO control points.

    VisuAlign marker format (normalised [0, 1]):
        x, y   — atlas overlay position (src in VERSO terms)
        dx, dy — displacement to matching section position: dst = (x+dx, y+dy)

    VERSO ControlPoint stores both src and dst as normalised [0, 1] coords,
    so this is a direct pass-through with no scaling.

    Args:
        markers: List of ``{"x", "y", "dx", "dy"}`` dicts in normalised coords.

    Returns:
        List of :class:`ControlPoint` with normalised [0, 1] coordinates.
    """
    cps: list[ControlPoint] = []
    for m in markers:
        x, y = float(m["x"]), float(m["y"])
        dx, dy = float(m["dx"]), float(m["dy"])
        cps.append(ControlPoint(src_x=x, src_y=y, dst_x=x + dx, dst_y=y + dy))
    return cps


def _control_points_to_markers(
    control_points: list[ControlPoint],
) -> list[dict[str, float]]:
    """Convert VERSO control points to VisuAlign normalised markers.

    VERSO ControlPoint stores normalised [0, 1] coords; the VisuAlign format
    stores the atlas position (x, y) and the section displacement (dx, dy).

    Args:
        control_points: List of :class:`ControlPoint` with normalised coords.

    Returns:
        List of ``{"x", "y", "dx", "dy"}`` dicts in normalised [0, 1] coords.
    """
    markers: list[dict[str, float]] = []
    for cp in control_points:
        dx = cp.dst_x - cp.src_x
        dy = cp.dst_y - cp.src_y
        markers.append({
            "x": round(cp.src_x, 6),
            "y": round(cp.src_y, 6),
            "dx": round(dx, 6),
            "dy": round(dy, 6),
        })
    return markers


# ---------------------------------------------------------------------------
# Public load functions
# ---------------------------------------------------------------------------

# NOTE — flip correction on import (not yet implemented)
# The three save functions apply _flip_anchoring / _flip_control_points when
# section.preprocessing.flip_horizontal is True, so exported files always
# contain coordinates in the original (unflipped) image space.
#
# The inverse is NOT yet applied on load.  This means that if a user:
#   1. sets flip_horizontal=True on a section in VERSO,
#   2. exports to QuickNII/VisuAlign JSON,
#   3. re-imports that JSON into VERSO,
# the loaded anchoring will be in original-image space while VERSO expects it in
# flipped-display space, so the overlay will appear mirror-reversed.
#
# To fix this, load_quicknii / load_visualign would need to know which sections
# are flipped (information not present in the QuickNII/VisuAlign JSON itself)
# and call _flip_anchoring / _flip_control_points on load.  The natural place to
# apply this is after the section's flip flag has been resolved — either by
# carrying the flag in a VERSO-specific JSON field or by asking the user to
# re-apply the flip after import.

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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    atlas_name = data.get("target", atlas_name)
    project_name = data.get("name", Path(path).stem)

    # QuickNII uses "slices"; VERSO project.json uses "sections"
    raw_sections = data.get("slices") or data.get("sections", [])

    sections: list[Section] = []
    for i, raw in enumerate(raw_sections):
        parsed = _parse_section_quicknii(raw)
        status = (
            AlignmentStatus.COMPLETE
            if any(parsed["anchoring"])
            else AlignmentStatus.NOT_STARTED
        )
        alignment = Alignment(
            anchoring=parsed["anchoring"],
            status=status,
        )
        section = Section(
            id=f"s{i + 1:03d}",
            serial_number=parsed["nr"],
            original_path=parsed["filename"],
            thumbnail_path=parsed["filename"],
            alignment=alignment,
        )
        sections.append(section)

    return Project(
        name=project_name,
        atlas=AtlasRef(name=atlas_name),
        sections=sections,
    )


def load_visualign(
    path: Path,
    atlas_name: str = "allen_mouse_25um",
) -> Project:
    """Load a VisuAlign JSON file (with control points) into a VERSO :class:`Project`.

    Marker coordinates are normalised [0, 1] in both atlas and section space,
    matching the VisuAlign JSON format directly.

    Args:
        path: Path to the VisuAlign ``*.json`` file.
        atlas_name: Fallback atlas name if not present in the JSON.

    Returns:
        A :class:`Project` including :class:`WarpState` for each section.
    """
    project = load_quicknii(path, atlas_name=atlas_name)

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_sections = data.get("slices") or data.get("sections", [])
    for section, raw in zip(project.sections, raw_sections):
        markers = raw.get("markers", [])
        if markers:
            cps = _markers_to_control_points(markers)
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
    from verso.engine.registration import flip_anchoring_horizontal

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
        oy_qn  = ap_max - oy_bg
        oz_qn  = dv_max - oz_bg
        uy_qn  = -uy_bg
        uz_qn  = -uz_bg
        vy_qn  = -vy_bg
        vz_qn  = -vz_bg

    This mirrors both the AP and DV axes while preserving LR.

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
    return [ox, ap_max - oy, dv_max - oz, ux, -uy, -uz, vx, -vy, -vz]


def _flip_control_points(
    cps: list[ControlPoint],
) -> list[ControlPoint]:
    """Convert control points from flipped display space to original image space.

    Both src and dst horizontal coordinates mirror: x' = 1 - x.
    """
    return [
        ControlPoint(
            src_x=1.0 - cp.src_x,
            src_y=cp.src_y,
            dst_x=1.0 - cp.dst_x,
            dst_y=cp.dst_y,
        )
        for cp in cps
    ]


def _export_image_filename(section, output_path: Path) -> str:
    """Return the image filename QuickNII/VisuAlign should load for a section.

    QuickNII resolves image filenames relative to the XML/JSON file location by
    concatenating the series folder and the stored filename.  VERSO aligns
    against the registration thumbnail when present, and the exported
    ``width``/``height`` fields describe that same image, so the filename must
    point to the thumbnail rather than the original high-resolution source.

    If the chosen image is on the same drive as the export, write a relative
    path.  If it is on a different drive, fall back to the native absolute path:
    QuickNII can still load it as a file path, while a bare basename would
    almost certainly point at a non-existent file beside the export.
    """
    thumbnail = Path(section.thumbnail_path)
    src = thumbnail if thumbnail.exists() else Path(section.original_path)
    out_dir = Path(output_path).resolve().parent
    try:
        return Path(os.path.relpath(src.resolve(), out_dir)).as_posix()
    except ValueError:
        return str(src)


def _registration_dims(section) -> tuple[int, int]:
    """Return registration image dimensions (width, height) in pixels.

    QuickNII's ``width`` and ``height`` are part of the anchoring scale:
    ``HStretch = len(u) / width`` and ``VStretch = len(v) / height``.  VERSO
    aligns against the working thumbnail when present, so exports must report
    that image size instead of reconstructing full-resolution dimensions from
    ``section.scale``.
    """
    try:
        from verso.engine.io.image_io import registration_dimensions
        return registration_dimensions(section)
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Public save functions
# ---------------------------------------------------------------------------

def _export_anchoring(
    anchoring: list[float],
    flip_horizontal: bool,
    atlas_shape: tuple[int, int, int] | None,
) -> list[float]:
    """Apply all export-time anchoring transforms.

    1. Horizontal-flip correction (if the section was displayed flipped).
    2. Atlas-axis convention conversion from brainglobe/VERSO to QuickNII,
       which requires knowing the atlas dimensions.

    Args:
        anchoring: Internal VERSO anchoring in brainglobe convention.
        flip_horizontal: Whether the section was displayed horizontally flipped.
        atlas_shape: Atlas dimensions ``(AP, DV, LR)``. Pass ``None`` to skip
            atlas convention conversion.

    Returns:
        Anchoring ready to write into a QuickNII/VisuAlign file.
    """
    a = anchoring
    if flip_horizontal:
        a = _flip_anchoring(a)
    if atlas_shape is not None:
        a = _to_quicknii_convention(a, atlas_shape)
    return a


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
            If *None*, the DV conversion is skipped (non-standard output).
    """
    lines: list[str] = ["<?xml version='1.0' encoding='UTF-8'?>"]
    res_attr = ""
    if atlas_shape is not None:
        ap, dv, lr = atlas_shape
        res_attr = f" target-resolution='{ap} {dv} {lr}'"
    atlas_name = project.atlas.name if project.atlas else ""
    lines.append(f"<series name='{project.name}' target='{atlas_name}'{res_attr}>")

    prefixes = ["' anchoring='ox=", "&amp;oy=", "&amp;oz=", "&amp;ux=",
                "&amp;uy=", "&amp;uz=", "&amp;vx=", "&amp;vy=", "&amp;vz="]
    for section in project.sections:
        w, h = _registration_dims(section)
        filename = _export_image_filename(section, path)
        line = (
            f"    <slice filename='{filename}'"
            f" nr='{section.serial_number}'"
            f" width='{w}' height='{h}"
        )
        if (section.alignment.status == AlignmentStatus.COMPLETE
                and section.alignment.anchoring
                and any(section.alignment.anchoring)):
            a = _export_anchoring(
                section.alignment.anchoring,
                section.preprocessing.flip_horizontal,
                atlas_shape,
            )
            for prefix, val in zip(prefixes, [round(v, 4) for v in a]):
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
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the DV-convention
            conversion. Pass *None* to skip conversion (non-standard output).
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section, path),
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if (section.alignment.status == AlignmentStatus.COMPLETE
                and section.alignment.anchoring
                and any(section.alignment.anchoring)):
            a = _export_anchoring(
                section.alignment.anchoring,
                section.preprocessing.flip_horizontal,
                atlas_shape,
            )
            entry["anchoring"] = [round(v, 4) for v in a]
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
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the DV-convention
            conversion. Pass *None* to skip conversion (non-standard output).
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section, path),
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if (section.alignment.status == AlignmentStatus.COMPLETE
                and section.alignment.anchoring
                and any(section.alignment.anchoring)):
            a = _export_anchoring(
                section.alignment.anchoring,
                section.preprocessing.flip_horizontal,
                atlas_shape,
            )
            entry["anchoring"] = [round(v, 4) for v in a]
        if section.warp.control_points:
            cps = section.warp.control_points
            if section.preprocessing.flip_horizontal:
                cps = _flip_control_points(cps)
            entry["markers"] = _control_points_to_markers(cps)
        slices_out.append(entry)

    data: dict[str, Any] = {
        "name": project.name,
        "target": project.atlas.name if project.atlas else "",
        "slices": slices_out,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
