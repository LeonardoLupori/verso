"""Public API surface for the VERSO engine.

This module re-exports only the **high-level** entry points that form VERSO's
stable scripting contract: the registration facade, the data model, atlas
access, project-format I/O, high-level export, and point/overlay warping.

    from verso.engine import VersoRegistration
    from verso.engine import Project, Section, Alignment, ControlPoint
    from verso.engine import load_quicknii, save_visualign, warp_points_section_to_atlas

Low-level primitives (anchoring plane math, series-interpolation packing,
coordinate transforms, rendering, working-copy I/O, DeepSlice/elastix
workflows, section bookkeeping) are intentionally **not** surfaced here so
they can be refactored without breaking users. They remain importable from
their own submodules, e.g.:

    from verso.engine.anchoring import rotate_anchoring, propagate_series_anchorings
    from verso.engine.deepslice import run_deepslice_suggestions
    from verso.engine.elastix import auto_control_points
"""

from verso.engine.annotations import points_in_polygon
from verso.engine.atlas import AtlasVolume, orientation_labels
from verso.engine.io.annotation_io import (
    annotations_dir,
    load_annotations,
    save_annotations,
)
from verso.engine.io.export_images import ExportOptions, export_section
from verso.engine.io.export_stack import ExportStackOptions, export_aligned_stack
from verso.engine.io.quint_io import (
    load_deepslice,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_visualign,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.annotation import AnnotationPoint, PointSeries
from verso.engine.model.elastix import ElastixParams
from verso.engine.model.project import (
    AtlasRef,
    ChannelSpec,
    DialogPrefs,
    Preprocessing,
    Project,
    Section,
)
from verso.engine.registration import AtlasToImageResult, VersoRegistration
from verso.engine.warping import (
    find_atlas_position,
    warp_overlay,
    warp_points_atlas_to_section,
    warp_points_section_to_atlas,
)

# Kept intentionally lean — see the module docstring. Grouped by module in the
# imports above; listed flat and sorted here.
__all__ = [
    "Alignment",
    "AlignmentStatus",
    "AnnotationPoint",
    "AtlasRef",
    "AtlasToImageResult",
    "AtlasVolume",
    "ChannelSpec",
    "ControlPoint",
    "DialogPrefs",
    "ElastixParams",
    "ExportOptions",
    "ExportStackOptions",
    "PointSeries",
    "Preprocessing",
    "Project",
    "Section",
    "VersoRegistration",
    "WarpState",
    "annotations_dir",
    "export_aligned_stack",
    "export_section",
    "find_atlas_position",
    "load_annotations",
    "load_deepslice",
    "load_quicknii",
    "load_visualign",
    "orientation_labels",
    "points_in_polygon",
    "save_annotations",
    "save_quicknii",
    "save_visualign",
    "warp_overlay",
    "warp_points_atlas_to_section",
    "warp_points_section_to_atlas",
]
