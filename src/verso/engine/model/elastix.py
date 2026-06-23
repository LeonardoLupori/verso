from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ElastixParams:
    """Tunable parameters for automatic elastix-based control-point generation.

    Stored per-project (in ``project-verso.json``) so a tuned configuration
    travels with the experiment. ``None`` on a project means "use these
    built-in defaults" until the user edits them.

    grid_spacing
        B-spline control-point spacing (px) at the finest resolution. The
        primary flexibility knob: smaller = more local deformation, larger =
        stiffer / more global.
    n_resolutions
        Coarse-to-fine pyramid levels. More levels widen the capture range.
    max_iterations
        Optimizer steps per resolution level.
    n_samples
        Random spatial samples per iteration for the mutual-information metric.
    registration_scale
        Downsample factor applied to the images before registration (the
        transform is still applied at working resolution). ``1.0`` = no
        downsampling; ``0.5`` = register at half size for speed.
    mask_dilation_register
        Radius (px) the tissue mask is expanded by before being used to gate
        the registration metric, so tissue near the edges still contributes.
    mask_dilation_cp
        Larger radius (px) the tissue mask is expanded by to decide where new
        control points may be created. Only crossings inside this mask are kept.
    """

    grid_spacing: int = 128
    n_resolutions: int = 2
    max_iterations: int = 250
    n_samples: int = 2048
    registration_scale: float = 1.0
    mask_dilation_register: int = 50
    mask_dilation_cp: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ElastixParams:
        defaults = cls()
        return cls(
            grid_spacing=int(d.get("grid_spacing", defaults.grid_spacing)),
            n_resolutions=int(d.get("n_resolutions", defaults.n_resolutions)),
            max_iterations=int(d.get("max_iterations", defaults.max_iterations)),
            n_samples=int(d.get("n_samples", defaults.n_samples)),
            registration_scale=float(d.get("registration_scale", defaults.registration_scale)),
            mask_dilation_register=int(
                d.get("mask_dilation_register", defaults.mask_dilation_register)
            ),
            mask_dilation_cp=int(d.get("mask_dilation_cp", defaults.mask_dilation_cp)),
        )
