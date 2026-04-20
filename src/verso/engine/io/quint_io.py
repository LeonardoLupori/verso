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

def save_quicknii_xml(
    project: Project,
    path: Path,
    atlas_shape: tuple[int, int, int] | None = None,
) -> None:
    """Write alignment data in QuickNII native XML format.

    Produces the same ``<series>/<slice>`` structure that QuickNII writes,
    with anchoring encoded as ``ox=...&oy=...&oz=...&ux=...&uy=...&uz=...&vx=...&vy=...&vz=...``.

    Args:
        project: VERSO project to export.
        path: Destination ``*.xml`` path.
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions for the ``target-resolution``
            attribute. If *None*, the attribute is omitted.
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
        line = (
            f"    <slice filename='{section.original_path}'"
            f" nr='{section.serial_number}'"
            f" width='{w}' height='{h}"
        )
        if section.alignment.anchoring and any(section.alignment.anchoring):
            a = [round(v, 4) for v in section.alignment.anchoring]
            for prefix, val in zip(prefixes, a):
                line += f"{prefix}{val}"
        line += "'/>"
        lines.append(line)

    lines.append("</series>")
    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def save_quicknii(project: Project, path: Path) -> None:
    """Write alignment data in QuickNII-compatible JSON format.

    Uses the ``"slices"`` key and includes ``"width"``, ``"height"``, and
    ``"target-resolution"`` fields to match the native QuickNII format.

    Only the affine alignment (anchoring) is written.
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": section.original_path,
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if section.alignment.anchoring and any(section.alignment.anchoring):
            entry["anchoring"] = [round(v, 4) for v in section.alignment.anchoring]
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
) -> None:
    """Write alignment + warp data in VisuAlign-compatible JSON format.

    Control point coordinates are stored as normalised [0, 1] values,
    matching the VisuAlign ``{"x", "y", "dx", "dy"}`` marker format directly.
    Width/height describe the registration image used by VERSO, matching
    QuickNII's stretch calculation.

    Args:
        project: VERSO project to export.
        path: Destination ``*.json`` path.
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": section.original_path,
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if section.alignment.anchoring and any(section.alignment.anchoring):
            entry["anchoring"] = [round(v, 4) for v in section.alignment.anchoring]
        if section.warp.control_points:
            entry["markers"] = _control_points_to_markers(section.warp.control_points)
        slices_out.append(entry)

    data: dict[str, Any] = {
        "name": project.name,
        "target": project.atlas.name if project.atlas else "",
        "slices": slices_out,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
