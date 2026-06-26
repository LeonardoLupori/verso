"""Public API surface for the VERSO engine.

User scripts and the GUI should import from here:

    from verso.engine import warp_overlay, load_quicknii, save_visualign
    from verso.engine import Project, Section, Alignment, ControlPoint
"""

from verso.engine.atlas import AtlasVolume, orientation_labels
from verso.engine.deepslice import (
    DeepSliceError,
    DeepSliceOptions,
    DeepSliceRunResult,
    DeepSliceSectionSuggestion,
    apply_deepslice_suggestions,
    apply_deepslice_suggestions_with_atlas,
    reset_in_progress_to_default_proposals,
    run_deepslice_suggestions,
)
from verso.engine.elastix import (
    anchor_source_points,
    auto_control_points,
    is_supported_atlas,
    load_anchor_lines,
)
from verso.engine.io.export_images import (
    ExportOptions,
    export_section,
    render_overlay_rgba,
    render_section_rgb,
)
from verso.engine.io.export_stack import (
    ExportStackOptions,
    build_canonical_remap,
    export_aligned_stack,
    export_section_aligned,
    finalize_aligned_pages,
    write_aligned_stack,
)
from verso.engine.io.image_io import (
    compute_working_scale,
    ensure_working_copy,
    image_dimensions,
    load_filmstrip_thumbnail,
    load_image,
    probe_channels,
    registration_dimensions,
)
from verso.engine.io.quint_io import (
    export_brainglobe_atlas_for_visualign,
    load_deepslice,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_visualign,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.elastix import ElastixParams
from verso.engine.model.project import AtlasRef, ChannelSpec, Preprocessing, Project, Section
from verso.engine.registration import (
    anchoring_center,
    anchoring_to_vectors,
    atlas_to_normalized,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    quicknii_default_anchoring,
    quicknii_pack_anchoring,
    quicknii_series_anchorings,
    quicknii_unpack_anchoring,
    rotate_anchoring,
    scale_anchoring,
    set_center_position_along_axis,
    set_position_along_axis,
    vectors_to_anchoring,
)
from verso.engine.sections import (
    make_added_sections,
    next_section_ids,
    removed_section_artifacts,
)
from verso.engine.warping import (
    find_atlas_position,
    warp_overlay,
    warp_points_atlas_to_section,
)

__all__ = [
    # Atlas
    "AtlasVolume",
    "orientation_labels",
    # Elastix auto control points
    "ElastixParams",
    "anchor_source_points",
    "auto_control_points",
    "is_supported_atlas",
    "load_anchor_lines",
    "DeepSliceError",
    "DeepSliceOptions",
    "DeepSliceRunResult",
    "DeepSliceSectionSuggestion",
    # I/O — image
    "compute_working_scale",
    "ensure_working_copy",
    "image_dimensions",
    "load_filmstrip_thumbnail",
    "load_image",
    "probe_channels",
    "registration_dimensions",
    # Model
    "Alignment",
    "AlignmentStatus",
    "AtlasRef",
    "ChannelSpec",
    "ControlPoint",
    "Preprocessing",
    "Project",
    "Section",
    "WarpState",
    # Registration
    "anchoring_center",
    "anchoring_to_vectors",
    "atlas_to_normalized",
    "make_atlas_sample_grid",
    "normalized_to_atlas",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "quicknii_default_anchoring",
    "quicknii_pack_anchoring",
    "quicknii_series_anchorings",
    "quicknii_unpack_anchoring",
    "rotate_anchoring",
    "scale_anchoring",
    "set_center_position_along_axis",
    "set_position_along_axis",
    "vectors_to_anchoring",
    # Sections (add/remove)
    "make_added_sections",
    "next_section_ids",
    "removed_section_artifacts",
    "apply_deepslice_suggestions",
    "apply_deepslice_suggestions_with_atlas",
    "reset_in_progress_to_default_proposals",
    "run_deepslice_suggestions",
    # Warping
    "find_atlas_position",
    "warp_overlay",
    "warp_points_atlas_to_section",
    # Export
    "ExportOptions",
    "export_section",
    "render_overlay_rgba",
    "render_section_rgb",
    "ExportStackOptions",
    "build_canonical_remap",
    "export_aligned_stack",
    "export_section_aligned",
    "finalize_aligned_pages",
    "write_aligned_stack",
    # I/O
    "export_brainglobe_atlas_for_visualign",
    "load_deepslice",
    "load_quicknii",
    "load_visualign",
    "save_quicknii",
    "save_visualign",
]
