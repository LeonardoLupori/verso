from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verso.engine.model.alignment import Alignment, WarpState
from verso.engine.model.elastix import ElastixParams

DEFAULT_PROJECT_FILENAME = "project-verso.json"

# Current project-schema version. Bumped to 1.2 when per-section pixel
# dimensions and atlas resolution/shape were added so the file is self-contained
# for pixel <-> atlas voxel mapping. Older files are migrated on load.
SCHEMA_VERSION = "1.2"

# Mapping between the stored axis-name field and the QuickNII voxel axis index.
# QuickNII voxel space ordering is (LR=0, AP=1, DV=2); "ML" is the storage name
# for the mediolateral / LR axis.
AXIS_NAME_TO_INDEX: dict[str, int] = {"AP": 1, "ML": 0, "DV": 2}
# Slicing orientation used in the New Project dialog. Each orientation declares
# which atlas axis the cutting series runs along (= which axis interpolation
# should target).
SLICING_ORIENTATION_TO_AXIS: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


@dataclass
class AtlasRef:
    """Reference to a brainglobe atlas.

    ``resolution_um`` (isotropic, microns per voxel) and ``shape`` (voxel
    dimensions in QuickNII/brainglobe order) are cached here so the project file
    is self-contained for coordinate work without re-fetching the atlas. They
    are ``0.0`` / ``(0, 0, 0)`` until populated (see ``backfill_metadata``).
    """

    name: str
    source: str = "brainglobe"
    resolution_um: float = 0.0
    shape: tuple[int, int, int] = (0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "resolution_um": self.resolution_um,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AtlasRef:
        raw_shape = d.get("shape") or (0, 0, 0)
        return cls(
            name=d["name"],
            source=d.get("source", "brainglobe"),
            resolution_um=float(d.get("resolution_um", 0.0)),
            shape=(int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2])),
        )


@dataclass
class ChannelSpec:
    """Per-channel display settings, shared across all sections in a project."""

    name: str
    color: tuple[int, int, int] = (255, 255, 255)
    scale: float = 1.0
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": list(self.color),
            "scale": self.scale,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChannelSpec:
        color = d.get("color", [255, 255, 255])
        return cls(
            name=d["name"],
            color=(int(color[0]), int(color[1]), int(color[2])),
            scale=float(d.get("scale", 1.0)),
            visible=bool(d.get("visible", True)),
        )


@dataclass
class Preprocessing:
    """Non-destructive preprocessing parameters stored per section."""

    flip_horizontal: bool = False
    flip_vertical: bool = False
    slice_mask_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "slice_mask_path": self.slice_mask_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessing:
        # Unknown keys (e.g. legacy ``lr_mask_path``/``lr_line``) are ignored.
        return cls(
            flip_horizontal=d.get("flip_horizontal", False),
            flip_vertical=d.get("flip_vertical", False),
            slice_mask_path=d.get("slice_mask_path"),
        )


@dataclass
class Section:
    """One histological section within a project."""

    id: str
    # Physical position of the section along the project's interpolation axis
    # (e.g. AP). Ground truth for ordering. Need not be contiguous (1, 2, 18, 19
    # encodes a gap), and may repeat when one physical slice broke into several
    # images. The section ``id`` breaks ties for a stable order.
    slice_index: int
    original_path: str
    thumbnail_path: str
    resolution_original_wh: tuple[int, int] = (0, 0)
    resolution_thumbnail_wh: tuple[int, int] = (0, 0)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    alignment: Alignment = field(default_factory=Alignment)
    warp: WarpState = field(default_factory=WarpState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slice_index": self.slice_index,
            "original_path": self.original_path,
            "thumbnail_path": self.thumbnail_path,
            "resolution_original_wh": list(self.resolution_original_wh),
            "resolution_thumbnail_wh": list(self.resolution_thumbnail_wh),
            "preprocessing": self.preprocessing.to_dict(),
            "alignment": self.alignment.to_dict(),
            "warp": self.warp.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Section:
        return cls(
            id=d["id"],
            slice_index=d["slice_index"],
            original_path=d["original_path"],
            thumbnail_path=d["thumbnail_path"],
            resolution_original_wh=tuple(d.get("resolution_original_wh", [0, 0])),
            resolution_thumbnail_wh=tuple(d.get("resolution_thumbnail_wh", [0, 0])),
            preprocessing=Preprocessing.from_dict(d.get("preprocessing", {})),
            alignment=Alignment.from_dict(d.get("alignment", {})),
            warp=WarpState.from_dict(d.get("warp", {})),
        )


@dataclass
class DialogPrefs:
    """Per-project flags controlling which dialogs are shown.

    Each flag is ``True`` (show the dialog) by default; the GUI sets it to
    ``False`` when the user ticks "do not show again".
    """

    show_align_deletion: bool = True
    show_overview_tutorial: bool = True
    show_preprocessing_tutorial: bool = True
    show_align_tutorial: bool = True
    show_warp_tutorial: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "show_align_deletion": self.show_align_deletion,
            "show_overview_tutorial": self.show_overview_tutorial,
            "show_preprocessing_tutorial": self.show_preprocessing_tutorial,
            "show_align_tutorial": self.show_align_tutorial,
            "show_warp_tutorial": self.show_warp_tutorial,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DialogPrefs:
        return cls(
            show_align_deletion=d.get("show_align_deletion", True),
            show_overview_tutorial=d.get("show_overview_tutorial", True),
            show_preprocessing_tutorial=d.get("show_preprocessing_tutorial", True),
            show_align_tutorial=d.get("show_align_tutorial", True),
            show_warp_tutorial=d.get("show_warp_tutorial", True),
        )


@dataclass
class Project:
    """Top-level project container."""

    name: str
    atlas: AtlasRef
    sections: list[Section] = field(default_factory=list)
    channels: list[ChannelSpec] = field(default_factory=list)
    cp_size: int = 10
    cp_shape: str = "Cross"
    cp_color: str = "#fff500"
    interpolation_axis: str = "AP"
    # Derived once at import from the largest image (see compute_working_scale);
    working_scale: float = 0.2
    # Per-project parameters for automatic elastix control-point generation.
    # None means "use the built-in ElastixParams defaults" until edited.
    elastix_params: ElastixParams | None = None
    dialog_prefs: DialogPrefs = field(default_factory=DialogPrefs)
    version: str = SCHEMA_VERSION

    @property
    def interpolation_axis_index(self) -> int:
        """QuickNII voxel axis index (0=ML, 1=AP, 2=DV) for ``interpolation_axis``."""
        return AXIS_NAME_TO_INDEX[self.interpolation_axis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "atlas": self.atlas.to_dict(),
            "interpolation_axis": self.interpolation_axis,
            "channels": [c.to_dict() for c in self.channels],
            "cp_size": self.cp_size,
            "cp_shape": self.cp_shape,
            "cp_color": self.cp_color,
            "working_scale": self.working_scale,
            "elastix_params": (
                self.elastix_params.to_dict() if self.elastix_params is not None else None
            ),
            "dialog_prefs": self.dialog_prefs.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
        }

    def sort_sections(self) -> None:
        """Sort ``sections`` in place by ``(slice_index, id)``.

        This is the canonical display/navigation order: filmstrip, overview, and
        interpolation all follow increasing ``slice_index``, with the section
        ``id`` (import order) breaking ties for duplicate indices.
        """
        self.sections.sort(key=lambda s: (s.slice_index, s.id))

    def save(self, path: Path) -> None:
        """Write project state to disk."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        cp_color = str(d.get("cp_color", "#fff500"))
        raw_axis = str(d.get("interpolation_axis", "AP")).upper()
        interpolation_axis = raw_axis if raw_axis in AXIS_NAME_TO_INDEX else "AP"
        raw_elastix = d.get("elastix_params")
        elastix_params = ElastixParams.from_dict(raw_elastix) if raw_elastix else None
        project = cls(
            name=d["name"],
            atlas=AtlasRef.from_dict(d["atlas"]),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            channels=[ChannelSpec.from_dict(c) for c in d.get("channels", [])],
            cp_size=int(d.get("cp_size", 10)),
            cp_shape=str(d.get("cp_shape", "Cross")),
            cp_color=cp_color,
            interpolation_axis=interpolation_axis,
            working_scale=float(d.get("working_scale", 0.2)),
            elastix_params=elastix_params,
            dialog_prefs=DialogPrefs.from_dict(d.get("dialog_prefs", {})),
            version=str(d.get("version", "1.1")),
        )
        project.sort_sections()
        return project

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from disk.

        Args:
            path: Path to a project JSON file.

        Returns:
            The loaded project.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
