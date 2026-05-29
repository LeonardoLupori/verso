from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verso.engine.model.alignment import Alignment, WarpState

DEFAULT_PROJECT_FILENAME = "project-verso.json"

# Mapping between the stored axis-name field and the QuickNII voxel axis index.
# QuickNII voxel space ordering is (LR=0, AP=1, DV=2); "ML" is the storage name
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
AXIS_TO_SLICING_ORIENTATION: dict[str, str] = {
    v: k for k, v in SLICING_ORIENTATION_TO_AXIS.items()
}


@dataclass
class AtlasRef:
    """Reference to a brainglobe atlas."""

    name: str
    source: str = "brainglobe"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AtlasRef:
        return cls(name=d["name"], source=d.get("source", "brainglobe"))


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
    lr_mask_path: str | None = None
    # Endpoints of the user-drawn L/R separating line, in unflipped working-
    # resolution pixel coords: [[x0, y0], [x1, y1]]. None means the L/R mask
    # (if any) is uniform (all-left or all-right) or simply not yet edited.
    lr_line: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "slice_mask_path": self.slice_mask_path,
            "lr_mask_path": self.lr_mask_path,
            "lr_line": (
                [[float(self.lr_line[0][0]), float(self.lr_line[0][1])],
                 [float(self.lr_line[1][0]), float(self.lr_line[1][1])]]
                if self.lr_line is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessing:
        raw = d.get("lr_line")
        line: list[list[float]] | None = None
        if raw and len(raw) == 2 and len(raw[0]) == 2 and len(raw[1]) == 2:
            line = [
                [float(raw[0][0]), float(raw[0][1])],
                [float(raw[1][0]), float(raw[1][1])],
            ]
        return cls(
            flip_horizontal=d.get("flip_horizontal", False),
            flip_vertical=d.get("flip_vertical", False),
            slice_mask_path=d.get("slice_mask_path"),
            lr_mask_path=d.get("lr_mask_path"),
            lr_line=line,
        )


@dataclass
class Section:
    """One histological section within a project."""

    id: str
    serial_number: int
    original_path: str
    thumbnail_path: str
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    alignment: Alignment = field(default_factory=Alignment)
    warp: WarpState = field(default_factory=WarpState)
    # Ratio: working_long_side / original_long_side (uniform, same for x and y)
    scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "serial_number": self.serial_number,
            "original_path": self.original_path,
            "thumbnail_path": self.thumbnail_path,
            "preprocessing": self.preprocessing.to_dict(),
            "alignment": self.alignment.to_dict(),
            "warp": self.warp.to_dict(),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Section:
        return cls(
            id=d["id"],
            serial_number=d["serial_number"],
            original_path=d["original_path"],
            thumbnail_path=d["thumbnail_path"],
            preprocessing=Preprocessing.from_dict(d.get("preprocessing", {})),
            alignment=Alignment.from_dict(d.get("alignment", {})),
            warp=WarpState.from_dict(d.get("warp", {})),
            scale=d.get("scale", 1.0),
        )


_LEGACY_CP_COLORS: dict[str, str] = {
    "Orange": "#ff6000",
    "Cyan": "#00ffff",
    "Yellow": "#fff500",
    "Red": "#ff2020",
    "White": "#ffffff",
    "Magenta": "#ff00ff",
}


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
    version: str = "1.1"

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
            "sections": [s.to_dict() for s in self.sections],
        }

    def save(self, path: Path) -> None:
        """Write project state to disk."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        raw_color = str(d.get("cp_color", "#fff500"))
        cp_color = (
            raw_color if raw_color.startswith("#") else _LEGACY_CP_COLORS.get(raw_color, "#fff500")
        )
        raw_axis = str(d.get("interpolation_axis", "AP")).upper()
        interpolation_axis = raw_axis if raw_axis in AXIS_NAME_TO_INDEX else "AP"
        return cls(
            name=d["name"],
            atlas=AtlasRef.from_dict(d["atlas"]),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            channels=[ChannelSpec.from_dict(c) for c in d.get("channels", [])],
            cp_size=int(d.get("cp_size", 10)),
            cp_shape=str(d.get("cp_shape", "Cross")),
            cp_color=cp_color,
            interpolation_axis=interpolation_axis,
            version="1.1",
        )

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
