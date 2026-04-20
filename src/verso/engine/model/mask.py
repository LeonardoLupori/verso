from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MaskType(StrEnum):
    SLICE = "slice"
    LEFT_RIGHT = "lr"


@dataclass
class Mask:
    """Metadata for a single mask PNG associated with a section."""

    path: str
    mask_type: MaskType

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "mask_type": self.mask_type.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mask:
        return cls(path=d["path"], mask_type=MaskType(d["mask_type"]))
