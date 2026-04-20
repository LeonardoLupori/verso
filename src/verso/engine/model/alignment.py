from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AlignmentStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass
class ControlPoint:
    """A warp control point in normalised [0, 1] coordinates.

    src_* : normalised position in atlas overlay space — fixed when the point
            is created; identifies which atlas feature is being moved.
    dst_* : normalised position in section image space — updated as the user
            drags the point to indicate where the feature should appear.
    """

    src_x: float
    src_y: float
    dst_x: float
    dst_y: float

    def to_dict(self) -> dict[str, float]:
        return {"src_x": self.src_x, "src_y": self.src_y, "dst_x": self.dst_x, "dst_y": self.dst_y}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ControlPoint:
        return cls(src_x=d["src_x"], src_y=d["src_y"], dst_x=d["dst_x"], dst_y=d["dst_y"])


@dataclass
class Alignment:
    """Affine registration of a section to the atlas.

    anchoring: 9-element list [ox, oy, oz, ux, uy, uz, vx, vy, vz]
        o = origin corner of the section plane in atlas voxel space
        u = right direction vector (along section width, unnormalized)
        v = down direction vector (along section height, unnormalized)

    For a point at normalized coords (s, t) ∈ [0, 1]²:
        atlas_voxel = o + s·u + t·v

    This matches the QuickNII anchoring format exactly.
    """

    anchoring: list[float] = field(default_factory=lambda: [0.0] * 9)
    ap_position_mm: float | None = None
    status: AlignmentStatus = AlignmentStatus.NOT_STARTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchoring": self.anchoring,
            "ap_position_mm": self.ap_position_mm,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alignment:
        return cls(
            anchoring=d.get("anchoring", [0.0] * 9),
            ap_position_mm=d.get("ap_position_mm"),
            status=AlignmentStatus(d.get("status", "not_started")),
        )


@dataclass
class WarpState:
    """Nonlinear warp refinement state for one section."""

    control_points: list[ControlPoint] = field(default_factory=list)
    status: AlignmentStatus = AlignmentStatus.NOT_STARTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_points": [cp.to_dict() for cp in self.control_points],
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WarpState:
        cps = [ControlPoint.from_dict(cp) for cp in d.get("control_points", [])]
        return cls(
            control_points=cps,
            status=AlignmentStatus(d.get("status", "not_started")),
        )
