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

Imports are **lazy** (PEP 562): each name is bound to its submodule the first
time it is accessed, not at ``import verso.engine``. This keeps the facade from
eagerly pulling in heavy dependencies — importing any ``verso.engine.*``
submodule runs this ``__init__`` first, and the eager form dragged in
``scipy.spatial`` (via ``warping``) at GUI startup even though warping is only
needed in the Warp view. The scripting contract above is unchanged; the objects
just resolve on first use.
"""

from typing import TYPE_CHECKING

# name -> submodule that defines it. Kept in sync with __all__ below (guarded by
# tests/engine/test_public_api.py). Grouped by module.
_EXPORTS = {
    # annotations
    "annotation_images": "verso.engine.annotations",
    "point_coords_by_image": "verso.engine.annotations",
    "points_in_polygon": "verso.engine.annotations",
    # atlas
    "AtlasVolume": "verso.engine.atlas",
    "orientation_labels": "verso.engine.atlas",
    # io.annotation_io
    "annotations_dir": "verso.engine.io.annotation_io",
    "load_annotations": "verso.engine.io.annotation_io",
    "save_annotations": "verso.engine.io.annotation_io",
    # io.export_images
    "ExportOptions": "verso.engine.io.export_images",
    "export_section": "verso.engine.io.export_images",
    # io.export_stack
    "ExportStackOptions": "verso.engine.io.export_stack",
    "export_aligned_stack": "verso.engine.io.export_stack",
    # io.quint_import
    "build_quint_project": "verso.engine.io.quint_import",
    "match_registration_images": "verso.engine.io.quint_import",
    # io.quint_io
    "load_deepslice": "verso.engine.io.quint_io",
    "load_quicknii": "verso.engine.io.quint_io",
    "load_visualign": "verso.engine.io.quint_io",
    "save_quicknii": "verso.engine.io.quint_io",
    "save_visualign": "verso.engine.io.quint_io",
    # model.alignment
    "Alignment": "verso.engine.model.alignment",
    "AlignmentStatus": "verso.engine.model.alignment",
    "ControlPoint": "verso.engine.model.alignment",
    "WarpState": "verso.engine.model.alignment",
    # model.annotation
    "AnnotationPoint": "verso.engine.model.annotation",
    "AreaAnnotation": "verso.engine.model.annotation",
    "PointSeries": "verso.engine.model.annotation",
    # model.elastix
    "ElastixParams": "verso.engine.model.elastix",
    # quantification
    "QuantifyOptions": "verso.engine.quantification",
    "QuantificationError": "verso.engine.quantification",
    "quantify_area": "verso.engine.quantification",
    "quantify_dots": "verso.engine.quantification",
    "quantify_intensity": "verso.engine.quantification",
    # model.project
    "AtlasRef": "verso.engine.model.project",
    "ChannelSpec": "verso.engine.model.project",
    "DialogPrefs": "verso.engine.model.project",
    "Preprocessing": "verso.engine.model.project",
    "Project": "verso.engine.model.project",
    "Section": "verso.engine.model.project",
    # registration
    "AtlasToImageResult": "verso.engine.registration",
    "VersoRegistration": "verso.engine.registration",
    # warping
    "find_atlas_position": "verso.engine.warping",
    "warp_overlay": "verso.engine.warping",
    "warp_points_atlas_to_section": "verso.engine.warping",
    "warp_points_section_to_atlas": "verso.engine.warping",
}

# Kept intentionally lean — see the module docstring. Listed flat and sorted;
# must stay in sync with _EXPORTS (enforced by the public-API test).
__all__ = [
    "Alignment",
    "AlignmentStatus",
    "AnnotationPoint",
    "AreaAnnotation",
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
    "QuantificationError",
    "QuantifyOptions",
    "Section",
    "VersoRegistration",
    "WarpState",
    "annotation_images",
    "annotations_dir",
    "build_quint_project",
    "export_aligned_stack",
    "export_section",
    "find_atlas_position",
    "load_annotations",
    "load_deepslice",
    "load_quicknii",
    "load_visualign",
    "match_registration_images",
    "orientation_labels",
    "point_coords_by_image",
    "points_in_polygon",
    "quantify_area",
    "quantify_dots",
    "quantify_intensity",
    "save_annotations",
    "save_quicknii",
    "save_visualign",
    "warp_overlay",
    "warp_points_atlas_to_section",
    "warp_points_section_to_atlas",
]


def __getattr__(name: str) -> object:
    """Resolve a public export to its submodule on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(module), name)
    globals()[name] = obj  # cache so subsequent lookups skip __getattr__
    return obj


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Give type checkers and IDEs the real symbols (the lazy __getattr__ above is
    # opaque to static analysis). Not executed at runtime, so no import cost.
    from verso.engine.annotations import (
        annotation_images,
        point_coords_by_image,
        points_in_polygon,
    )
    from verso.engine.atlas import AtlasVolume, orientation_labels
    from verso.engine.io.annotation_io import (
        annotations_dir,
        load_annotations,
        save_annotations,
    )
    from verso.engine.io.export_images import ExportOptions, export_section
    from verso.engine.io.export_stack import ExportStackOptions, export_aligned_stack
    from verso.engine.io.quint_import import build_quint_project, match_registration_images
    from verso.engine.io.quint_io import (
        load_deepslice,
        load_quicknii,
        load_visualign,
        save_quicknii,
        save_visualign,
    )
    from verso.engine.model.alignment import (
        Alignment,
        AlignmentStatus,
        ControlPoint,
        WarpState,
    )
    from verso.engine.model.annotation import AnnotationPoint, AreaAnnotation, PointSeries
    from verso.engine.model.elastix import ElastixParams
    from verso.engine.model.project import (
        AtlasRef,
        ChannelSpec,
        DialogPrefs,
        Preprocessing,
        Project,
        Section,
    )
    from verso.engine.quantification import (
        QuantificationError,
        QuantifyOptions,
        quantify_area,
        quantify_dots,
        quantify_intensity,
    )
    from verso.engine.registration import AtlasToImageResult, VersoRegistration
    from verso.engine.warping import (
        find_atlas_position,
        warp_overlay,
        warp_points_atlas_to_section,
        warp_points_section_to_atlas,
    )
