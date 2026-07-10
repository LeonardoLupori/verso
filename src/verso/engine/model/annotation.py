"""Annotation data model.

An *annotation* is a named overlay the user draws on the section canvas in the
Annotate view, with its own colour and visibility (areas also carry an opacity;
point scatters always render fully opaque). Annotations are persisted outside
``project-verso.json``, in an ``annotations/`` subfolder of the project (see
:mod:`verso.engine.io.annotation_io`).

Two types exist:

* :class:`PointSeries` — a collection of ``(x, y, image)`` points that may span
  many sections (stored as ``points.csv``).
* :class:`AreaAnnotation` — a per-section binary mask (working resolution) that may
  span many sections (stored as ``masks/<image>.png``); e.g. an injection region.

The ``type`` discriminator and the metadata (de)serialisation split let each type
be persisted independently. :data:`Annotation` is the union used by callers that
handle both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Annotation type discriminators, persisted as ``type`` in annotation.json.
POINT_SERIES = "point_series"
AREA = "area"


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
    visible: bool = True
    points: list[AnnotationPoint] = field(default_factory=list)
    #: Diameter (screen px) points are rendered at in the Annotate view.
    point_size: int = 9
    #: Type discriminator persisted in annotation.json.
    type: str = POINT_SERIES

    def metadata_to_dict(self) -> dict[str, Any]:
        """Serialise everything except the points (which live in points.csv)."""
        return {
            "type": self.type,
            "title": self.title,
            "color": list(self.color),
            "visible": self.visible,
            "point_size": self.point_size,
        }

    @classmethod
    def from_metadata(cls, d: dict[str, Any], points: list[AnnotationPoint]) -> PointSeries:
        """Rebuild a series from its metadata dict plus already-loaded points.

        A legacy ``opacity`` key (point series used to carry one) is ignored.
        """
        color = d.get("color", [255, 64, 64])
        return cls(
            title=d["title"],
            color=(int(color[0]), int(color[1]), int(color[2])),
            visible=bool(d.get("visible", True)),
            points=list(points),
            point_size=int(d.get("point_size", 9)),
            type=str(d.get("type", POINT_SERIES)),
        )


@dataclass
class AreaAnnotation:
    """A per-section binary mask that may span multiple sections.

    Rendered as one coloured, semi-transparent overlay in the Annotate view. The
    ``masks`` dict maps an image basename (``Path(section.original_path).name``) to
    a working-resolution boolean mask (``True`` = inside the area); sections with
    no mask are simply absent from the dict. Masks live on disk as 1-bit PNGs (see
    :mod:`verso.engine.io.annotation_io`), not in ``annotation.json``.
    """

    title: str
    color: tuple[int, int, int] = (255, 64, 64)
    # Masks read better semi-transparent, so areas default to 50% opacity.
    opacity: float = 0.5
    visible: bool = True
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    #: Type discriminator persisted in annotation.json.
    type: str = AREA

    def metadata_to_dict(self) -> dict[str, Any]:
        """Serialise everything except the masks (which live as PNGs)."""
        return {
            "type": self.type,
            "title": self.title,
            "color": list(self.color),
            "opacity": self.opacity,
            "visible": self.visible,
        }

    @classmethod
    def from_metadata(cls, d: dict[str, Any], masks: dict[str, np.ndarray]) -> AreaAnnotation:
        """Rebuild an area from its metadata dict plus already-loaded masks."""
        color = d.get("color", [255, 64, 64])
        return cls(
            title=d["title"],
            color=(int(color[0]), int(color[1]), int(color[2])),
            opacity=float(d.get("opacity", 0.5)),
            visible=bool(d.get("visible", True)),
            masks=dict(masks),
            type=str(d.get("type", AREA)),
        )


#: Any annotation type, for callers (controller/view/IO) that handle both.
Annotation = PointSeries | AreaAnnotation
