"""Tests for automatic (elastix) control-point generation.

These cover the pure-Python parts: anchor-line loading, mirroring, voxel
scaling, plane crossings, and the data-model round-trips. The native elastix
registration itself is not exercised here — it is slow, needs the atlas, and the
optimizer segfaults under pytest's output capture (it runs in an isolated
subprocess in production; see ``verso.engine.elastix``).
"""

from __future__ import annotations

import sys
import types

import numpy as np

from verso.engine import elastix
from verso.engine.elastix import (
    anchor_source_points,
    is_supported_atlas,
    load_anchor_lines,
)
from verso.engine.model.alignment import ControlPoint
from verso.engine.model.elastix import ElastixParams
from verso.engine.model.project import AtlasRef, Project

# A centered coronal plane for the 25 µm Allen atlas (shape AP=528, DV=320, LR=456):
# origin at mid-AP, u spans the full ML width, v spans the full DV height.
_CORONAL_25UM = [0.0, 264.0, 0.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0]
_SHAPE_25UM = (528, 320, 456)


# ---------------------------------------------------------------------------
# Data-model round-trips
# ---------------------------------------------------------------------------


def test_control_point_auto_round_trip():
    auto = ControlPoint(0.5, 0.5, 0.51, 0.49, auto=True)
    assert ControlPoint.from_dict(auto.to_dict()).auto is True
    # auto=False stays out of the serialized dict (legacy-compatible JSON).
    manual = ControlPoint(0.1, 0.2, 0.3, 0.4)
    assert "auto" not in manual.to_dict()
    assert ControlPoint.from_dict(manual.to_dict()).auto is False
    # Legacy dicts without the key default to manual.
    assert ControlPoint.from_dict({"src_x": 0, "src_y": 0, "dst_x": 0, "dst_y": 0}).auto is False


def test_elastix_params_round_trip_and_partial_defaults():
    params = ElastixParams(grid_spacing=64, n_resolutions=3, registration_scale=0.5)
    assert ElastixParams.from_dict(params.to_dict()) == params
    # Missing keys fall back to defaults.
    partial = ElastixParams.from_dict({"grid_spacing": 32})
    assert partial.grid_spacing == 32
    assert partial.n_resolutions == ElastixParams().n_resolutions


def test_project_elastix_params_round_trip():
    p = Project(
        name="x",
        atlas=AtlasRef(name="allen_mouse_25um"),
        elastix_params=ElastixParams(n_samples=999),
    )
    assert Project.from_dict(p.to_dict()).elastix_params.n_samples == 999
    # Absent → None (use built-in defaults).
    p2 = Project(name="y", atlas=AtlasRef(name="allen_mouse_25um"))
    assert Project.from_dict(p2.to_dict()).elastix_params is None


# ---------------------------------------------------------------------------
# Atlas support gate
# ---------------------------------------------------------------------------


def test_is_supported_atlas():
    assert is_supported_atlas("allen_mouse_25um")
    assert is_supported_atlas("allen_mouse_10um")
    assert not is_supported_atlas("kim_mouse_25um")
    assert not is_supported_atlas("allen_human_500um")


# ---------------------------------------------------------------------------
# Curated anchor-line resource
# ---------------------------------------------------------------------------


def test_anchor_lines_loaded_and_mirrored():
    res = load_anchor_lines()
    assert res["resolution_um"] == 25
    assert res["shape"] == [528, 320, 456]
    lines = res["lines"]
    assert lines, "no anchor lines packaged"

    base_names = [n for n in lines if not n.endswith("_mirror")]
    for name in base_names:
        if "mid" in name.lower():
            # Midline lines are not mirrored.
            assert f"{name}_mirror" not in lines
        else:
            assert f"{name}_mirror" in lines, f"missing mirror for {name}"

    # Each polyline is a dense (N, 3) list of floats.
    sample = next(iter(lines.values()))
    assert len(sample) >= 2 and len(sample[0]) == 3


def test_anchor_lines_mirror_reflects_ml():
    res = load_anchor_lines()
    ml_max = res["shape"][2] - 1
    lines = res["lines"]
    name = next(n for n in lines if not n.endswith("_mirror") and "mid" not in n.lower())
    base = np.asarray(lines[name])
    mirror = np.asarray(lines[f"{name}_mirror"])
    # ML (x = column 2) is reflected; AP (z) and DV (y) unchanged.
    assert np.allclose(mirror[:, 2], ml_max - base[:, 2], atol=1e-6)
    assert np.allclose(mirror[:, 0], base[:, 0], atol=1e-6)
    assert np.allclose(mirror[:, 1], base[:, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Plane crossing → source control points
# ---------------------------------------------------------------------------


def test_source_points_in_unit_square():
    src = anchor_source_points(_CORONAL_25UM, _SHAPE_25UM, 456, 320)
    assert src.ndim == 2 and src.shape[1] == 2
    assert len(src) > 0
    assert ((src >= 0.0) & (src <= 1.0)).all()


def test_source_points_respect_mask_gate():
    full = anchor_source_points(_CORONAL_25UM, _SHAPE_25UM, 456, 320)
    mask = np.zeros((320, 456), dtype=bool)
    mask[120:200, 150:300] = True
    gated = anchor_source_points(_CORONAL_25UM, _SHAPE_25UM, 456, 320, cp_mask=mask)
    assert len(gated) <= len(full)
    # Every gated crossing lands inside the mask.
    for s, t in gated:
        rx = int(round(s * 456))
        ry = int(round(t * 320))
        assert mask[min(ry, 319), min(rx, 455)]


def test_source_points_scale_to_other_resolution():
    # The same anatomical plane on a 10 µm grid (2.5× finer) still yields points
    # in [0, 1]; the curated 25 µm coords are scaled by the atlas shape ratio.
    shape_10um = (1320, 800, 1140)
    coronal_10um = [0.0, 660.0, 0.0, 1140.0, 0.0, 0.0, 0.0, 0.0, 800.0]
    src = anchor_source_points(coronal_10um, shape_10um, 1140, 800)
    assert len(src) > 0
    assert ((src >= 0.0) & (src <= 1.0)).all()


def test_degenerate_anchoring_returns_empty():
    # u and v parallel → zero normal → no crossings.
    bad = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    assert len(anchor_source_points(bad, _SHAPE_25UM, 100, 100)) == 0


# ---------------------------------------------------------------------------
# auto_control_points: destination gating
# ---------------------------------------------------------------------------


def _make_fake_itk(offset: float) -> types.ModuleType:
    """A minimal stand-in for the native ``itk`` module.

    Models transformix on a coordinate ramp as a *uniform translation*: the
    resampled value at each pixel is the input value plus ``offset``. With
    ``offset == 0`` the recovered deformation is the identity (destinations land
    back on their sources); a large ``offset`` shifts every destination far off
    the section, which is what the real optimizer can do near tissue edges.
    """
    ns = types.ModuleType("itk")

    class _Img:
        def __init__(self, arr):
            self.array = np.asarray(arr, dtype=np.float32)

        def SetSpacing(self, spacing):  # noqa: N802 (itk camelCase API)
            pass

    class _ParamObj:
        @staticmethod
        def New():  # noqa: N802
            return _ParamObj()

        def GetDefaultParameterMap(self, name, n):  # noqa: N802
            return {}

        def AddParameterMap(self, m):  # noqa: N802
            pass

    class _Transform:
        def SetParameter(self, idx, key, val):  # noqa: N802
            pass

    ns.ParameterObject = _ParamObj
    ns.image_from_array = lambda arr: _Img(arr)
    ns.array_from_image = lambda img: img.array
    ns.transformix_filter = lambda image, tp: _Img(image.array + offset)
    ns.elastix_registration_method = (
        lambda fixed, moving, parameter_object=None, **kw: (None, _Transform())
    )
    return ns


def test_auto_cps_inside_image_are_kept(monkeypatch):
    # Identity deformation: every destination stays on the section, so all the
    # plane-crossing source points survive as auto control points.
    w, h = 120, 100
    section = np.zeros((h, w), dtype=np.float32)
    template = np.zeros((h, w), dtype=np.float32)
    monkeypatch.setitem(sys.modules, "itk", _make_fake_itk(0.0))

    cps = elastix.auto_control_points(section, template, _CORONAL_25UM, _SHAPE_25UM)

    assert len(cps) > 0
    for cp in cps:
        assert cp.auto is True
        assert 0.0 <= cp.dst_x <= 1.0 and 0.0 <= cp.dst_y <= 1.0


def test_auto_cps_outside_image_are_dropped(monkeypatch):
    # A large uniform displacement throws every destination far off-section; the
    # destination gate must discard them all rather than emit off-image points.
    w, h = 120, 100
    section = np.zeros((h, w), dtype=np.float32)
    template = np.zeros((h, w), dtype=np.float32)
    monkeypatch.setitem(sys.modules, "itk", _make_fake_itk(10.0 * w))

    cps = elastix.auto_control_points(section, template, _CORONAL_25UM, _SHAPE_25UM)

    assert cps == []
