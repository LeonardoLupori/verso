"""Non-destructive preprocessing: flip, masking, contrast.

All operations return parameters to store in project.json; they never
modify the original image on disk.
"""

from __future__ import annotations

import numpy as np

from verso.engine.model.project import Preprocessing


def apply_flip(image: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    """Return a flipped copy of *image* according to *preprocessing* flags.

    Args:
        image: H×W or H×W×C uint8 array.
        preprocessing: Preprocessing parameters from the project model.

    Returns:
        Flipped image array (same dtype and shape).
    """
    raise NotImplementedError


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside *mask*.

    Args:
        image: H×W or H×W×C uint8 array.
        mask: Boolean H×W array; True = keep, False = zero.

    Returns:
        Masked image array.
    """
    raise NotImplementedError
