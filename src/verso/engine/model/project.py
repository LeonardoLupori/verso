from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from verso.engine.model.alignment import Alignment, WarpState
from verso.engine.model.elastix import ElastixParams

DEFAULT_PROJECT_FILENAME = "project-verso.json"

# Project-schema version, stamped on every saved file.
SCHEMA_VERSION = "1.0"

# Mapping between the stored axis-name field and the anchoring voxel axis index.
# Anchoring voxel space ordering is (LR=0, AP=1, DV=2); "ML" is the storage name
# for the mediolateral / LR axis.
AXIS_NAME_TO_INDEX: dict[str, int] = {"AP": 1, "ML": 0, "DV": 2}
AXIS_INDEX_TO_NAME: dict[int, str] = {v: k for k, v in AXIS_NAME_TO_INDEX.items()}
# Slicing orientation used in the New Project dialog. Each orientation declares
# which atlas axis the cutting series runs along (= which axis interpolation
# should target).
SLICING_ORIENTATION_TO_AXIS: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}
# Reverse of the above: atlas axis name → slicing-orientation label.
AXIS_TO_SLICING_ORIENTATION: dict[str, str] = {
    v: k for k, v in SLICING_ORIENTATION_TO_AXIS.items()
}


@dataclass
class AtlasRef:
    """Reference to a brainglobe atlas.

    ``resolution_um`` (isotropic, microns per voxel) and ``shape`` (voxel
    dimensions in BrainGlobe's ``(AP, DV, LR)`` order) are cached here so the
    project file is self-contained for coordinate work without re-fetching the
    atlas. They are ``0.0`` / ``(0, 0, 0)`` until populated (see
    ``populate_metadata``).
    """

    name: str
    source: str = "brainglobe"
    resolution_um: float = 0.0
    shape: tuple[int, int, int] = (0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    gamma: float = 1.0
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChannelSpec:
        color = d.get("color", [255, 255, 255])
        return cls(
            name=d["name"],
            color=(int(color[0]), int(color[1]), int(color[2])),
            scale=float(d.get("scale", 1.0)),
            gamma=float(d.get("gamma", 1.0)),
            visible=bool(d.get("visible", True)),
        )


@dataclass
class Preprocessing:
    """Non-destructive preprocessing parameters stored per section."""

    flip_horizontal: bool = False
    flip_vertical: bool = False
    slice_mask_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    # Index of the scene/image within a multi-scene container file (CZI).
    # ``0`` for single-image formats (TIFF/PNG/JPG) and for the first scene of a
    # container. Sections from the same container share ``original_path`` but
    # differ in ``scene_index``; it also disambiguates their thumbnail names.
    scene_index: int = 0
    resolution_original_wh: tuple[int, int] = (0, 0)
    resolution_thumbnail_wh: tuple[int, int] = (0, 0)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    alignment: Alignment = field(default_factory=Alignment)
    warp: WarpState = field(default_factory=WarpState)

    @property
    def image_key(self) -> str:
        """Canonical per-section identity for annotations (masks, points).

        Annotations key their masks and points by this string rather than the raw
        image basename: a multi-scene container (CZI) yields several sections that
        share ``original_path``, so the basename alone is not unique and would make
        every scene's annotations collide onto one another. Scene ``0`` keeps the
        plain filename — backward compatible with single-image projects and with
        externally-produced point CSVs that reference images by name — while each
        further scene gets a ``-scene{NN}`` suffix on the stem to stay distinct.
        """
        name = Path(self.original_path).name
        if self.scene_index:
            return f"{Path(name).stem}-scene{self.scene_index:02d}"
        return name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slice_index": self.slice_index,
            "original_path": self.original_path,
            "thumbnail_path": self.thumbnail_path,
            "scene_index": self.scene_index,
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
            scene_index=int(d.get("scene_index", 0)),
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
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DialogPrefs:
        defaults = cls()
        return cls(**{f.name: bool(d.get(f.name, getattr(defaults, f.name))) for f in fields(cls)})


@dataclass
class Project:
    """Top-level project container."""

    name: str
    atlas: AtlasRef
    sections: list[Section] = field(default_factory=list)
    channels: list[ChannelSpec] = field(default_factory=list)
    cp_size: int = 10
    cp_shape: str = "Circle"
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
        """Anchoring voxel axis index (0=ML, 1=AP, 2=DV) for ``interpolation_axis``."""
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
            cp_shape=str(d.get("cp_shape", "Circle")),
            cp_color=cp_color,
            interpolation_axis=interpolation_axis,
            working_scale=float(d.get("working_scale", 0.2)),
            elastix_params=elastix_params,
            dialog_prefs=DialogPrefs.from_dict(d.get("dialog_prefs", {})),
            version=str(d.get("version", SCHEMA_VERSION)),
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
