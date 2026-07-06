"""Public API surface for the VERSO engine.

User scripts and the GUI should import from here:

    from verso.engine import warp_overlay, load_quicknii, save_visualign
    from verso.engine import Project, Section, Alignment, ControlPoint
"""

from verso.engine.anchoring import (
    anchoring_center,
    anchoring_to_vectors,
    atlas_to_normalized,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    plane_tilt_deg,
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
from verso.engine.model.project import (
    AtlasRef,
    ChannelSpec,
    DialogPrefs,
    Preprocessing,
    Project,
    Section,
)
from verso.engine.registration import AtlasToImageResult, VersoRegistration
from verso.engine.sections import (
    make_added_sections,
    next_section_ids,
    removed_section_artifacts,
)
from verso.engine.warping import (
    find_atlas_position,
    warp_overlay,
    warp_points_atlas_to_section,
    warp_points_section_to_atlas,
)

__all__ = [
    # Model
    "Alignment",
    "AlignmentStatus",
    "AtlasRef",
    "AtlasToImageResult",
    # Atlas
    "AtlasVolume",
    "ChannelSpec",
    "ControlPoint",
    "DeepSliceError",
    "DeepSliceOptions",
    "DeepSliceRunResult",
    "DeepSliceSectionSuggestion",
    "DialogPrefs",
    # Elastix auto control points
    "ElastixParams",
    # Export
    "ExportOptions",
    "ExportStackOptions",
    "Preprocessing",
    "Project",
    "Section",
    # Registration API (high-level pixel <-> atlas)
    "VersoRegistration",
    "WarpState",
    "anchor_source_points",
    # Anchoring (plane math + coordinate transforms)
    "anchoring_center",
    "anchoring_to_vectors",
    "apply_deepslice_suggestions",
    "apply_deepslice_suggestions_with_atlas",
    "atlas_to_normalized",
    "auto_control_points",
    "build_canonical_remap",
    # I/O — image
    "compute_working_scale",
    "ensure_working_copy",
    "export_aligned_stack",
    # I/O
    "export_brainglobe_atlas_for_visualign",
    "export_section",
    "export_section_aligned",
    "finalize_aligned_pages",
    # Warping
    "find_atlas_position",
    "image_dimensions",
    "is_supported_atlas",
    "load_anchor_lines",
    "load_deepslice",
    "load_filmstrip_thumbnail",
    "load_image",
    "load_quicknii",
    "load_visualign",
    # Sections (add/remove)
    "make_added_sections",
    "make_atlas_sample_grid",
    "next_section_ids",
    "normalized_to_atlas",
    "normalized_to_pixel",
    "orientation_labels",
    "pixel_to_normalized",
    "plane_tilt_deg",
    "probe_channels",
    "quicknii_default_anchoring",
    "quicknii_pack_anchoring",
    "quicknii_series_anchorings",
    "quicknii_unpack_anchoring",
    "registration_dimensions",
    "removed_section_artifacts",
    "render_overlay_rgba",
    "render_section_rgb",
    "reset_in_progress_to_default_proposals",
    "rotate_anchoring",
    "run_deepslice_suggestions",
    "save_quicknii",
    "save_visualign",
    "scale_anchoring",
    "set_center_position_along_axis",
    "set_position_along_axis",
    "vectors_to_anchoring",
    "warp_overlay",
    "warp_points_atlas_to_section",
    "warp_points_section_to_atlas",
    "write_aligned_stack",
]
