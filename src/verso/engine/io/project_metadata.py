"""In-place operations on a VERSO project.

Serialization itself lives on the data model (``Project.save`` / ``Project.load``).
This module holds helpers that augment a project: populating derived metadata
(image dimensions, atlas resolution/shape) at creation time and importing styling
(channel colors, control-point style) from another project.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from verso.engine.model.project import Project

if TYPE_CHECKING:
    from verso.engine.atlas import AtlasVolume


class AtlasUnavailableError(RuntimeError):
    """The reference atlas could not be obtained (e.g. offline first download)."""


def _resolve(path: str, project_dir: Path) -> Path:
    """Resolve a stored artifact path against the project directory.

    Absolute paths (e.g. ``original_path``) are used as-is; relative paths
    (e.g. a ``thumbnails/…`` ``thumbnail_path``) are joined to ``project_dir``.
    """
    p = Path(path)
    return p if p.is_absolute() else project_dir / p


def populate_metadata(
    project: Project,
    project_dir: Path,
    *,
    atlas: AtlasVolume | None = None,
) -> None:
    """Populate derived metadata (image dims + atlas resolution/shape) in place.

    Called at project creation to make the file self-contained for pixel <->
    atlas voxel mapping. Only fields that are still unpopulated (``0`` / ``0.0``
    / ``(0, 0, 0)``) are filled, so already-complete projects are untouched.
    Image dimensions come from the image files via :func:`image_dimensions`;
    atlas resolution and shape come from ``atlas`` if supplied, else from a
    freshly constructed :class:`~verso.engine.atlas.AtlasVolume`.

    Args:
        project: Project to update in place.
        project_dir: Directory containing the project file, used to resolve
            relative ``thumbnail_path`` entries.
        atlas: Optional already-loaded atlas volume to read metadata from,
            avoiding a re-fetch by name.

    Raises:
        FileNotFoundError: If an image needed for dimensions is missing.
        AtlasUnavailableError: If the atlas metadata cannot be obtained (e.g. the
            atlas is not cached locally and cannot be downloaded).
    """
    from verso.engine.io.image_io import ensure_working_copy, image_dimensions

    project_dir = Path(project_dir)

    for section in project.sections:
        if min(section.resolution_original_wh) <= 0:
            orig = _resolve(section.original_path, project_dir)
            if not orig.exists():
                raise FileNotFoundError(
                    f"Cannot populate dimensions for section {section.id!r}: "
                    f"original image not found at {orig}"
                )
            section.resolution_original_wh = image_dimensions(orig)
        if min(section.resolution_thumbnail_wh) <= 0:
            thumb = _resolve(section.thumbnail_path, project_dir)
            if not thumb.exists():  # noqa: SIM102 — keep the regeneration side effect explicit
                # Regenerate the working copy from the original if the thumbnail
                # is absent (e.g. generation was skipped at import).
                if ensure_working_copy(section, project.working_scale) is None:
                    raise FileNotFoundError(
                        f"Cannot populate dimensions for section {section.id!r}: "
                        f"thumbnail missing and could not be regenerated from "
                        f"{_resolve(section.original_path, project_dir)}"
                    )
            section.resolution_thumbnail_wh = image_dimensions(thumb)

    ref = project.atlas
    if ref.resolution_um <= 0 or any(d <= 0 for d in ref.shape):
        if atlas is None:
            try:
                from verso.engine.atlas import AtlasVolume

                atlas = AtlasVolume(ref.name)
            except Exception as exc:
                raise AtlasUnavailableError(
                    f"Cannot populate atlas metadata for {ref.name!r}: {exc}"
                ) from exc
        ref.resolution_um = float(atlas.resolution_um)
        ref.shape = tuple(int(v) for v in atlas.shape)  # type: ignore[assignment]


def import_project_styling(target: Project, source: Project) -> None:
    """Copy color and styling settings from *source* into *target* in place.

    Imports:
      - Per-channel ``color``, ``scale``, ``visible``, matched by position
        (index). Channel ``name`` is preserved on the target. Channels past
        the end of either list are left untouched.
      - Control-point style: ``cp_size``, ``cp_shape``, ``cp_color``.

    Args:
        target: Project to update in place.
        source: Project whose styling is read.
    """
    overlap = min(len(target.channels), len(source.channels))
    for i in range(overlap):
        src = source.channels[i]
        tgt = target.channels[i]
        tgt.color = src.color
        tgt.scale = src.scale
        tgt.visible = src.visible

    target.cp_size = source.cp_size
    target.cp_shape = source.cp_shape
    target.cp_color = source.cp_color
