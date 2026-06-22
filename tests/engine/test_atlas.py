"""Tests for engine/atlas.py voxel-selection convention."""

import numpy as np

from verso.engine.atlas import _quicknii_floor_indices


def test_quicknii_floor_indices_ceils_ap_dv_and_floors_lr():
    """VisuAlign/QUINT floor the voxel in QuickNII space; AP/DV are array-reversed
    there vs BrainGlobe, so they become ceil, while LR (shared) stays floor."""
    lr = np.array([3.2, 3.0, 3.9])
    ap = np.array([10.7, 10.0, 10.5])
    dv = np.array([5.5, 5.0, 5.1])

    lr_i, ap_i, dv_i = _quicknii_floor_indices(lr, ap, dv)

    np.testing.assert_array_equal(lr_i, [3.0, 3.0, 3.0])  # floor (LR)
    np.testing.assert_array_equal(ap_i, [11.0, 10.0, 11.0])  # ceil (AP)
    np.testing.assert_array_equal(dv_i, [6.0, 5.0, 6.0])  # ceil (DV)


def test_quicknii_floor_indices_scalar():
    """Works on scalars too (the cursor region-lookup path)."""
    lr_i, ap_i, dv_i = _quicknii_floor_indices(3.2, 10.7, 5.5)
    assert (float(lr_i), float(ap_i), float(dv_i)) == (3.0, 11.0, 6.0)


def test_ceil_equals_floor_in_reversed_axis():
    """The rationale: ceil(c) == (N-1) - floor((N-1) - c), i.e. ceiling a
    BrainGlobe coordinate equals flooring it in the reversed QuickNII axis."""
    n = 528
    for c in (0.0, 10.0, 10.3, 10.49, 10.5, 10.51, 10.7, 527.0):
        assert np.ceil(c) == (n - 1) - np.floor((n - 1) - c)
