"""Tests for engine/atlas.py voxel-selection convention."""

import numpy as np
import pytest

from verso.engine.atlas import _sample_voxel_indices, orientation_labels


def test_orientation_labels_coronal_ap():
    assert orientation_labels("AP") == {
        "top": "Dorsal",
        "bottom": "Ventral",
        "left": "Left",
        "right": "Right",
    }


def test_orientation_labels_sagittal_ml():
    assert orientation_labels("ML") == {
        "top": "Dorsal",
        "bottom": "Ventral",
        "left": "Anterior",
        "right": "Posterior",
    }


def test_orientation_labels_horizontal_dv():
    assert orientation_labels("DV") == {
        "top": "Anterior",
        "bottom": "Posterior",
        "left": "Left",
        "right": "Right",
    }


def test_orientation_labels_unknown_axis_raises():
    with pytest.raises(KeyError):
        orientation_labels("XX")


def test_orientation_labels_returns_independent_copy():
    """Callers may mutate the result without affecting later calls."""
    labels = orientation_labels("AP")
    labels["top"] = "MUTATED"
    assert orientation_labels("AP")["top"] == "Dorsal"


def test_sample_voxel_indices_ceils_ap_dv_and_floors_lr():
    """VisuAlign/QUINT floor the voxel in anchoring space; AP/DV are array-reversed
    there vs BrainGlobe, so they become ceil, while LR (shared) stays floor."""
    lr = np.array([3.2, 3.0, 3.9])
    ap = np.array([10.7, 10.0, 10.5])
    dv = np.array([5.5, 5.0, 5.1])

    lr_i, ap_i, dv_i = _sample_voxel_indices(lr, ap, dv)

    np.testing.assert_array_equal(lr_i, [3.0, 3.0, 3.0])  # floor (LR)
    np.testing.assert_array_equal(ap_i, [11.0, 10.0, 11.0])  # ceil (AP)
    np.testing.assert_array_equal(dv_i, [6.0, 5.0, 6.0])  # ceil (DV)


def test_sample_voxel_indices_scalar():
    """Works on scalars too (the cursor region-lookup path)."""
    lr_i, ap_i, dv_i = _sample_voxel_indices(3.2, 10.7, 5.5)
    assert (float(lr_i), float(ap_i), float(dv_i)) == (3.0, 11.0, 6.0)


def test_ceil_equals_floor_in_reversed_axis():
    """The rationale: ceil(c) == (N-1) - floor((N-1) - c), i.e. ceiling a
    BrainGlobe coordinate equals flooring it in the reversed anchoring axis."""
    n = 528
    for c in (0.0, 10.0, 10.3, 10.49, 10.5, 10.51, 10.7, 527.0):
        assert np.ceil(c) == (n - 1) - np.floor((n - 1) - c)
