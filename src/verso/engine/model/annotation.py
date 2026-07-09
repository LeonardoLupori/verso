"""Annotation data model.

An *annotation* is a named overlay the user draws on the section canvas in the
Annotate view, with its own colour, opacity, and visibility. Annotations are
persisted outside ``project-verso.json``, in an ``annotations/`` subfolder of the
project (see :mod:`verso.engine.io.annotation_io`).

For now there is a single type — :class:`PointSeries`, a collection of
``(x, y, image)`` points that may span many sections. The ``type`` discriminator
and the metadata (de)serialisation split are structured so a future ``Area``
annotation (a multi-section mask) can be added without touching existing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Annotation type discriminators, persisted as ``type`` in annotation.json.
POINT_SERIES = "point_series"


@dataclass
class AnnotationPoint:
    """A single annotated point in original full-resolution image pixels.

    Attributes:
        x: X coordinate in original-resolution pixels.
        y: Y coordinate in original-resolution pixels.
        image: Basename of the section's original image file
            (``Path(section.original_path).name``). Maps the point to the
            section it belongs to and lets externally-produced CSVs reference
            images by name.
    """

    x: float
    y: float
    image: str


@dataclass
class PointSeries:
    """A collection of annotated points that may span multiple sections.

    Rendered as one coloured scatter overlay in the Annotate view. Coordinates
    are original-resolution pixels (see :class:`AnnotationPoint`); the GUI scales
    them to working resolution on render via ``Project.working_scale``.
    """

    title: str
    color: tuple[int, int, int] = (255, 64, 64)
    opacity: float = 1.0
    visible: bool = True
    points: list[AnnotationPoint] = field(default_factory=list)
    #: Type discriminator persisted in annotation.json.
    type: str = POINT_SERIES

    def metadata_to_dict(self) -> dict[str, Any]:
        """Serialise everything except the points (which live in points.csv)."""
        return {
            "type": self.type,
            "title": self.title,
            "color": list(self.color),
            "opacity": self.opacity,
            "visible": self.visible,
        }

    @classmethod
    def from_metadata(cls, d: dict[str, Any], points: list[AnnotationPoint]) -> PointSeries:
        """Rebuild a series from its metadata dict plus already-loaded points."""
        color = d.get("color", [255, 64, 64])
        return cls(
            title=d["title"],
            color=(int(color[0]), int(color[1]), int(color[2])),
            opacity=float(d.get("opacity", 1.0)),
            visible=bool(d.get("visible", True)),
            points=list(points),
            type=str(d.get("type", POINT_SERIES)),
        )
