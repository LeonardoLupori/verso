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
    x, y  — normalised atlas overlay position in [0, 1]²  (left=0, top=0)
    dx, dy — displacement from atlas position (x, y) to the matching
              position on the histological section (dst − src).

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
    in normalised coords for backward compatibility when loading old exports.

    Args:
        markers: List of 4-element arrays or legacy dicts.
        width: Section image width in pixels (for normalisation).
        height: Section image height in pixels (for normalisation).

    Returns:
        List of :class:`ControlPoint` with normalised [0, 1] coordinates.
    """
    cps: list[ControlPoint] = []
    for m in markers:
        if isinstance(m, (list, tuple)) and len(m) == 4:
            sx, sy, dx, dy = float(m[0]), float(m[1]), float(m[2]), float(m[3])
            w = float(width) if width else 1.0
            h = float(height) if height else 1.0
            cps.append(ControlPoint(src_x=sx / w, src_y=sy / h, dst_x=dx / w, dst_y=dy / h))
        elif isinstance(m, dict):
            x, y = float(m["x"]), float(m["y"])
            ddx, ddy = float(m["dx"]), float(m["dy"])
            cps.append(ControlPoint(src_x=x, src_y=y, dst_x=x + ddx, dst_y=y + ddy))
    return cps


def _control_points_to_markers(
    control_points: list[ControlPoint],
    width: int,
    height: int,
) -> list[list[float]]:
    """Convert VERSO control points to VisuAlign pixel-coordinate markers.

    VisuAlign stores markers as 4-element arrays:
        [src_x_px, src_y_px, dst_x_px, dst_y_px]
    in image pixels at the working resolution.

    Args:
        control_points: List of :class:`ControlPoint` with normalised coords.
        width: Section image width in pixels.
        height: Section image height in pixels.

    Returns:
        List of ``[src_x_px, src_y_px, dst_x_px, dst_y_px]`` arrays.
    """
    w, h = float(width), float(height)
    markers: list[list[float]] = []
    for cp in control_points:
        markers.append([
            round(cp.src_x * w, 6),
            round(cp.src_y * h, 6),
            round(cp.dst_x * w, 6),
            round(cp.dst_y * h, 6),
        ])
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

    # QuickNII/VisuAlign use "slices"; accept "sections" for forward compatibility
    raw_sections = data.get("slices") or data.get("sections", [])

    # QuickNII files store anchoring in QuickNII convention; convert to the
    # BrainGlobe (VERSO internal) convention so atlas slicing is correct.
    # _to_quicknii_convention is self-inverse, so it also converts QN→BG.
    bg_shape = _BG_ATLAS_SHAPE.get(atlas_name)

    sections: list[Section] = []
    for i, raw in enumerate(raw_sections):
        parsed = _parse_section_quicknii(raw)
        anchoring = parsed["anchoring"]
        if bg_shape is not None and any(anchoring):
            anchoring = _to_quicknii_convention(anchoring, bg_shape)
        status = (
            AlignmentStatus.COMPLETE
            if any(anchoring)
            else AlignmentStatus.NOT_STARTED
        )
        alignment = Alignment(
            anchoring=anchoring,
            status=status,
        )
        section = Section(
            id=f"s{i + 1:03d}",
            serial_number=parsed["nr"],
            original_path=parsed["filename"],
            thumbnail_path="",
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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_sections = data.get("slices") or data.get("sections", [])
    for section, raw in zip(project.sections, raw_sections):
        if section.alignment.status == AlignmentStatus.COMPLETE:
            section.alignment.status = AlignmentStatus.IN_PROGRESS
            section.alignment.source = "deepslice"
            section.alignment.proposal_anchoring = list(section.alignment.anchoring)
            confidence = raw.get("confidence")
            if confidence is not None:
                section.alignment.proposal_confidence = float(confidence)
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


def write_section_pngs(project: Project, output_dir: Path) -> None:
    """Write RGB PNG copies of each section's working image into *output_dir*.

    Only sections whose PNG is not already present are written.  Uses the
    project channel specs to composite multichannel images to RGB.

    Args:
        project: VERSO project whose sections will be exported.
        output_dir: Folder that will receive the PNG files.
    """
    import numpy as np
    from PIL import Image

    from verso.engine.io.image_io import ensure_working_copy
    from verso.engine.preprocessing import composite_channels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for section in project.sections:
        png_path = output_dir / _export_image_filename(section)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        if png_path.exists():
            continue
        img = ensure_working_copy(section)
        if img is None:
            continue
        if project.channels:
            rgb = composite_channels(img, project.channels)
        else:
            plane = img[:, :, 0] if img.ndim == 3 else img
            rgb = np.stack([plane, plane, plane], axis=-1)
        Image.fromarray(rgb).save(str(png_path))


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
            If *None*, the DV conversion is skipped (non-standard output).
    """
    lines: list[str] = ["<?xml version='1.0' encoding='UTF-8'?>"]
    res_attr = ""
    if atlas_shape is not None:
        ap, dv, lr = atlas_shape
        res_attr = f" target-resolution='{ap} {dv} {lr}'"
    atlas_name = project.atlas.name if project.atlas else ""
    lines.append(f"<series name='{project.name}' target='{atlas_name}'{res_attr}>")

    from verso.engine.registration import _display_space_anchoring
    prefixes = ["' anchoring='ox=", "&amp;oy=", "&amp;oz=", "&amp;ux=",
                "&amp;uy=", "&amp;uz=", "&amp;vx=", "&amp;vy=", "&amp;vz="]
    for section in project.sections:
        w, h = _registration_dims(section)
        filename = _export_image_filename(section)
        line = (
            f"    <slice filename='{filename}'"
            f" nr='{section.serial_number}'"
            f" width='{w}' height='{h}"
        )
        if section.alignment.status == AlignmentStatus.COMPLETE:
            original = _display_space_anchoring(section)
            if any(original):
                a = _export_anchoring(original, atlas_shape)
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
        atlas_shape: ``(AP, DV, LR)`` voxel dimensions used for the BrainGlobe→
            QuickNII axis convention conversion. When *None*, inferred from the
            project atlas name using the built-in lookup table.
    """
    if atlas_shape is None and project.atlas:
        atlas_shape = _BG_ATLAS_SHAPE.get(project.atlas.name)
    from verso.engine.registration import _display_space_anchoring
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section),
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if section.alignment.status == AlignmentStatus.COMPLETE:
            original = _display_space_anchoring(section)
            if any(original):
                entry["anchoring"] = [round(v, 4) for v in _export_anchoring(original, atlas_shape)]
        slices_out.append(entry)

    data: dict[str, Any] = {
        "name": project.name,
        "target": project.atlas.name if project.atlas else "",
        "slices": slices_out,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_brainglobe_atlas_for_visualign(
    atlas_name: str,
    output_dir: Path,
) -> Path:
    """Export a BrainGlobe atlas as a VisuAlign-compatible ``.cutlas`` directory.

    Creates ``{output_dir}/{atlas_name}.cutlas/`` containing:

    - ``labels.nii.gz`` — annotation volume in QuickNII/VisuAlign axis order
    - ``labels.txt`` — ITK-SNAP region labels with RGB colours

    The atlas volume is stored with axes ``(LR=x, AP=y, DV=z)`` as expected
    by VisuAlign, with AP and DV flipped to match QuickNII indexing convention
    (AP 0 = posterior, DV 0 = inferior) but the **LR axis is not flipped**
    (index 0 = right hemisphere).  This is consistent with the anchoring
    coordinates VERSO exports: ``anchoring[0] = LR_brainglobe`` with 0 = right.

    Place the generated ``.cutlas`` folder in VisuAlign's atlas directory.
    When the target name in the exported JSON matches the folder name,
    VisuAlign will load the BrainGlobe atlas, giving an exact match with
    VERSO's atlas overlay.

    Args:
        atlas_name: BrainGlobe atlas identifier (e.g. ``"allen_mouse_25um"``).
        output_dir: Directory that will receive the ``.cutlas`` folder.

    Returns:
        Path to the generated ``{atlas_name}.cutlas`` directory.
    """
    import gzip
    import struct

    import numpy as np
    from brainglobe_atlasapi import BrainGlobeAtlas

    output_dir = Path(output_dir)
    bg = BrainGlobeAtlas(atlas_name, check_latest=False)
    annotation = bg.annotation   # (AP, DV, LR) uint32
    structures = bg.structures   # {id: {"name": ..., "rgb_triplet": [r, g, b], ...}}

    # Flip AP (0→posterior) and DV (0→inferior); preserve LR direction
    # (0→right, matching BrainGlobe "asr" convention and VERSO's exported
    # anchoring where anchoring[0] = LR_bg with 0 = right hemisphere).
    # Reorder from BrainGlobe (AP, DV, LR) to VisuAlign (LR, AP, DV).
    qn_volume = np.ascontiguousarray(
        np.transpose(annotation[::-1, ::-1, :], (2, 0, 1)),
        dtype=np.uint32,
    )

    cutlas_dir = output_dir / f"{atlas_name}.cutlas"
    cutlas_dir.mkdir(parents=True, exist_ok=True)

    # --- labels.nii.gz ---
    x, y, z = qn_volume.shape
    hdr = bytearray(348)
    struct.pack_into('<i', hdr, 0, 348)       # sizeof_hdr
    struct.pack_into('<h', hdr, 40, 3)        # dim[0] = ndims
    struct.pack_into('<h', hdr, 42, x)        # dim[1] = LR
    struct.pack_into('<h', hdr, 44, y)        # dim[2] = AP
    struct.pack_into('<h', hdr, 46, z)        # dim[3] = DV
    struct.pack_into('<h', hdr, 70, 768)      # datatype = uint32
    struct.pack_into('<h', hdr, 72, 32)       # bitpix
    struct.pack_into('<f', hdr, 76, 1.0)      # pixdim[0]
    struct.pack_into('<f', hdr, 108, 352.0)   # vox_offset (348 header + 4 pad)
    hdr[344:348] = b'n+1\x00'                 # NIfTI-1 magic
    with gzip.open(cutlas_dir / "labels.nii.gz", 'wb', compresslevel=6) as f:
        f.write(bytes(hdr))
        f.write(b'\x00' * 4)                  # padding to reach byte offset 352
        f.write(qn_volume.tobytes())

    # --- labels.txt ---
    txt_lines = [
        "################################################\n",
        "# ITK-SnAP Label Description File\n",
        "# File format:\n",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n",
        "################################################\n",
        '    0     0    0    0        0  0  0    "Clear Label"\n',
    ]
    for label_id, info in sorted(structures.items()):
        r, g, b = info.get("rgb_triplet", [128, 128, 128])
        name = str(info.get("name", f"Region {label_id}")).replace('"', "'")
        txt_lines.append(f'{label_id}  {int(r)}  {int(g)}  {int(b)}  1  1  1    "{name}"\n')
    (cutlas_dir / "labels.txt").write_text("".join(txt_lines), encoding="utf-8")

    return cutlas_dir


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
    from verso.engine.registration import _display_space_anchoring
    slices_out: list[dict[str, Any]] = []
    for section in project.sections:
        w, h = _registration_dims(section)
        entry: dict[str, Any] = {
            "filename": _export_image_filename(section),
            "nr": section.serial_number,
            "width": w,
            "height": h,
        }
        if section.alignment.status == AlignmentStatus.COMPLETE:
            original = _display_space_anchoring(section)
            if any(original):
                entry["anchoring"] = [round(v, 4) for v in _export_anchoring(original, atlas_shape)]
        if section.warp.control_points:
            cps = section.warp.control_points
            if section.preprocessing.flip_horizontal:
                cps = _flip_control_points(cps)
            entry["markers"] = _control_points_to_markers(cps, w, h)
        slices_out.append(entry)

    va_target, va_resolution = _visualign_target(project.atlas.name if project.atlas else "")
    data: dict[str, Any] = {"name": project.name, "target": va_target}
    if va_resolution is not None:
        data["target-resolution"] = va_resolution
    data["slices"] = slices_out
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
