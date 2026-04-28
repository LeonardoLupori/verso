"""Public API surface for the VERSO engine.

User scripts and the GUI should import from here:

    from verso.engine import warp_overlay, load_quicknii, save_visualign
    from verso.engine import Project, Section, Alignment, ControlPoint
"""

from verso.engine.atlas import AtlasVolume
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
from verso.engine.io.image_io import (
    ensure_working_copy,
    image_dimensions,
    load_filmstrip_thumbnail,
    load_for_display,
    load_image,
    registration_dimensions,
)
from verso.engine.io.quint_io import (
    load_deepslice,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_visualign,
)
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.mask import Mask, MaskType
from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section
from verso.engine.registration import (
    anchoring_to_vectors,
    atlas_to_normalized,
    make_atlas_sample_grid,
    normalized_to_atlas,
    normalized_to_pixel,
    pixel_to_normalized,
    quicknii_coronal_default_anchoring,
    quicknii_coronal_series_anchorings,
    quicknii_pack_anchoring,
    quicknii_unpack_anchoring,
    rotate_anchoring,
    scale_anchoring,
    set_ap_position,
    vectors_to_anchoring,
)
from verso.engine.warping import find_atlas_position, warp_overlay

__all__ = [
    # Atlas
    "AtlasVolume",
    "DeepSliceError",
    "DeepSliceOptions",
    "DeepSliceRunResult",
    "DeepSliceSectionSuggestion",
    # I/O — image
    "ensure_working_copy",
    "image_dimensions",
    "load_filmstrip_thumbnail",
    "load_for_display",
    "load_image",
    "registration_dimensions",
    # Model
    "Alignment",
    "AlignmentStatus",
    "AtlasRef",
    "ControlPoint",
    "Mask",
    "MaskType",
    "Preprocessing",
    "Project",
    "Section",
    "WarpState",
    # Registration
    "anchoring_to_vectors",
    "atlas_to_normalized",
    "make_atlas_sample_grid",
    "normalized_to_atlas",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "quicknii_coronal_default_anchoring",
    "quicknii_coronal_series_anchorings",
    "quicknii_pack_anchoring",
    "quicknii_unpack_anchoring",
    "rotate_anchoring",
    "scale_anchoring",
    "set_ap_position",
    "vectors_to_anchoring",
    "apply_deepslice_suggestions",
    "apply_deepslice_suggestions_with_atlas",
    "reset_in_progress_to_default_proposals",
    "run_deepslice_suggestions",
    # Warping
    "find_atlas_position",
    "warp_overlay",
    # I/O
    "load_deepslice",
    "load_quicknii",
    "load_visualign",
    "save_quicknii",
    "save_visualign",
]
