"""Public API surface for the VERSO engine.

User scripts and the GUI should import from here:

    from verso.engine import warp_overlay, load_quicknii, save_visualign
    from verso.engine import Project, Section, Alignment, ControlPoint
"""

from verso.engine.atlas import AtlasVolume
from verso.engine.model.alignment import Alignment, AlignmentStatus, ControlPoint, WarpState
from verso.engine.model.mask import Mask, MaskType
from verso.engine.model.project import AtlasRef, Preprocessing, Project, Section
from verso.engine.registration import (
    anchoring_to_vectors,
    atlas_to_normalized,
    make_atlas_sample_grid,
    normalized_to_atlas,
    pixel_to_normalized,
    normalized_to_pixel,
    rotate_anchoring,
    scale_anchoring,
    set_ap_position,
    vectors_to_anchoring,
)
from verso.engine.warping import find_atlas_position, warp_overlay
from verso.engine.io.image_io import (
    ensure_working_copy,
    load_filmstrip_thumbnail,
    load_for_display,
    load_image,
)
from verso.engine.io.quint_io import (
    load_deepslice,
    load_quicknii,
    load_visualign,
    save_quicknii,
    save_visualign,
)

__all__ = [
    # Atlas
    "AtlasVolume",
    # I/O — image
    "ensure_working_copy",
    "load_filmstrip_thumbnail",
    "load_for_display",
    "load_image",
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
    "rotate_anchoring",
    "scale_anchoring",
    "set_ap_position",
    "vectors_to_anchoring",
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
