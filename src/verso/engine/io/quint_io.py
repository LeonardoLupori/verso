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
    overlay_width: int,
    overlay_height: int,
) -> list[ControlPoint]:
    """Convert VisuAlign normalised markers to VERSO pixel-space control points.

    Args:
        markers: List of ``{"x", "y", "dx", "dy"}`` dicts in normalised coords.
        overlay_width: Width of the atlas overlay in working-resolution pixels.
        overlay_height: Height of the atlas overlay in working-resolution pixels.

    Returns:
        List of :class:`ControlPoint` in pixel coordinates.
    """
    cps: list[ControlPoint] = []
    for m in markers:
        x, y = float(m["x"]), float(m["y"])
        dx, dy = float(m["dx"]), float(m["dy"])
        cps.append(
            ControlPoint(
                src_x=x * overlay_width,
                src_y=y * overlay_height,
                dst_x=(x + dx) * overlay_width,
                dst_y=(y + dy) * overlay_height,
            )
        )
    return cps


def _control_points_to_markers(
    control_points: list[ControlPoint],
    overlay_width: int,
    overlay_height: int,
) -> list[dict[str, float]]:
    """Convert VERSO pixel-space control points to VisuAlign normalised markers.

    Args:
        control_points: List of :class:`ControlPoint` in pixel coordinates.
        overlay_width: Width of the atlas overlay in working-resolution pixels.
        overlay_height: Height of the atlas overlay in working-resolution pixels.

    Returns:
        List of ``{"x", "y", "dx", "dy"}`` dicts in normalised coords.
    """
    markers: list[dict[str, float]] = []
    for cp in control_points:
        x = cp.src_x / overlay_width
        y = cp.src_y / overlay_height
        dx = (cp.dst_x - cp.src_x) / overlay_width
        dy = (cp.dst_y - cp.src_y) / overlay_height
        markers.append({"x": round(x, 6), "y": round(y, 6), "dx": round(dx, 6), "dy": round(dy, 6)})
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
        alignment = Alignment(
            anchoring=parsed["anchoring"],
            status=AlignmentStatus.COMPLETE if any(parsed["anchoring"]) else AlignmentStatus.NOT_STARTED,
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
    overlay_width: int = 456,
    overlay_height: int = 320,
    atlas_name: str = "allen_mouse_25um",
) -> Project:
    """Load a VisuAlign JSON file (with control points) into a VERSO :class:`Project`.

    Args:
        path: Path to the VisuAlign ``*.json`` file.
        overlay_width: Width of the atlas overlay in pixels, used to convert
            normalised marker coords to pixel coords.  Should match the working
            resolution overlay dimensions used when the file was created.
        overlay_height: Height of the atlas overlay in pixels.
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
            cps = _markers_to_control_points(markers, overlay_width, overlay_height)
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
# Public save functions
# ---------------------------------------------------------------------------

def save_quicknii(project: Project, path: Path) -> None:
    """Write alignment data in QuickNII-compatible JSON format.

    Uses the ``"slices"`` key and includes ``"width"``, ``"height"``, and
    ``"target-resolution"`` fields to match the native QuickNII format.

    Only the affine alignment (anchoring) is written.
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        entry: dict[str, Any] = {
            "filename": section.original_path,
            "nr": section.serial_number,
        }
        # width/height in original pixels if available; fall back to 0
        from PIL import Image as PILImage
        try:
            with PILImage.open(section.original_path) as im:
                entry["width"], entry["height"] = im.size
        except Exception:
            entry["width"] = 0
            entry["height"] = 0
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
    overlay_width: int = 456,
    overlay_height: int = 320,
) -> None:
    """Write alignment + warp data in VisuAlign-compatible JSON format.

    Args:
        project: VERSO project to export.
        path: Destination ``*.json`` path.
        overlay_width: Width of the atlas overlay used for normalising control
            point pixel coordinates.
        overlay_height: Height of the atlas overlay.
    """
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        entry: dict[str, Any] = {
            "filename": section.original_path,
            "nr": section.serial_number,
        }
        from PIL import Image as PILImage
        try:
            with PILImage.open(section.original_path) as im:
                entry["width"], entry["height"] = im.size
        except Exception:
            entry["width"] = 0
            entry["height"] = 0
        if section.alignment.anchoring and any(section.alignment.anchoring):
            entry["anchoring"] = [round(v, 4) for v in section.alignment.anchoring]
        if section.warp.control_points:
            entry["markers"] = _control_points_to_markers(
                section.warp.control_points, overlay_width, overlay_height
            )
        slices_out.append(entry)

    data: dict[str, Any] = {
        "name": project.name,
        "target": project.atlas.name if project.atlas else "",
        "slices": slices_out,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
