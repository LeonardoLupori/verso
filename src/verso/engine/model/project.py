from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verso.engine.model.alignment import Alignment, WarpState

DEFAULT_PROJECT_FILENAME = "project-verso.json"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "slice_mask_path": self.slice_mask_path,
            "lr_mask_path": self.lr_mask_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessing:
        return cls(
            flip_horizontal=d.get("flip_horizontal", False),
            flip_vertical=d.get("flip_vertical", False),
            slice_mask_path=d.get("slice_mask_path"),
            lr_mask_path=d.get("lr_mask_path"),
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


@dataclass
class Project:
    """Top-level project container."""

    name: str
    atlas: AtlasRef
    sections: list[Section] = field(default_factory=list)
    channels: list[ChannelSpec] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "atlas": self.atlas.to_dict(),
            "channels": [c.to_dict() for c in self.channels],
            "sections": [s.to_dict() for s in self.sections],
        }

    def save(self, path: Path) -> None:
        """Write project state to disk."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        return cls(
            name=d["name"],
            atlas=AtlasRef.from_dict(d["atlas"]),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            channels=[ChannelSpec.from_dict(c) for c in d.get("channels", [])],
            version=d.get("version", "1.0"),
        )

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
