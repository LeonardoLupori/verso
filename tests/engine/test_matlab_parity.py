"""Cross-language numerical parity between the Python engine and the MATLAB port.

The MATLAB port (``matlab/+verso``) re-implements the coordinate math by hand, so
the two engines can silently drift.  This module is the Python half of a
golden-fixture contract that pins them together:

- ``build_cases()`` runs the **live Python engine** over a fixed set of inputs and
  records the outputs.
- In normal mode this test asserts the freshly-computed cases equal the committed
  fixture ``matlab/tests/fixtures/parity.json`` — so any change to Python numerics
  turns CI red until the fixture is regenerated on purpose.
- The MATLAB suite (``matlab/tests/tParity.m``) reads the *same* JSON, runs the
  MATLAB implementation over the identical inputs, and asserts it matches the
  committed expected values.

Transitively ``python == fixture`` and ``matlab == fixture`` give ``python ==
matlab`` with no Python-in-MATLAB bridge: JSON is the language-neutral boundary.

Regenerate after an intentional numeric change::

    UPDATE_PARITY_FIXTURES=1 uv run pytest tests/engine/test_matlab_parity.py

then re-run the MATLAB suite (``runtests('matlab/tests')``) and commit the JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from verso.engine.anchoring import anchoring_to_vectors
from verso.engine.atlas import _sample_voxel_indices
from verso.engine.registration import VersoRegistration
from verso.engine.warping import (
    warp_points_atlas_to_section,
    warp_points_section_to_atlas,
)

FIXTURE = Path(__file__).resolve().parents[2] / "matlab" / "tests" / "fixtures" / "parity.json"
TOL = 1e-9

# Small synthetic atlas geometry shared with tParity.m / tVersoRegistration.m.
_AP, _DV, _LR = 20, 16, 24
_RES_UM = 25.0


# ---------------------------------------------------------------------------
# Project builders (mirror matlab/tests helpers so both sides load identical JSON)
# ---------------------------------------------------------------------------


def _canonical_anchoring(position: float) -> list[float]:
    """Coronal (AP) plane spanning full LR x DV at the given AP position."""
    return [0.0, float(position), 0.0, float(_LR), 0.0, 0.0, 0.0, 0.0, float(_DV)]


def _section(
    sid: str,
    position: float,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
    control_points: list[dict] | None = None,
    work: tuple[int, int] = (48, 32),
    full: tuple[int, int] = (96, 64),
) -> dict:
    return {
        "id": sid,
        "slice_index": int(position),
        "original_path": f"{sid}.tif",
        "thumbnail_path": f"{sid}.ome.tif",
        "resolution_original_wh": list(full),
        "resolution_thumbnail_wh": list(work),
        "preprocessing": {
            "flip_horizontal": flip_h,
            "flip_vertical": flip_v,
            "slice_mask_path": "",
        },
        "alignment": {
            "anchoring": _canonical_anchoring(position),
            "position_mm": 0,
            "status": "complete",
            "source": "manual",
        },
        "warp": {"control_points": control_points or [], "status": "in_progress"},
    }


def _project(sections: list[dict]) -> dict:
    return {
        "version": "1.3",
        "name": "parity",
        "atlas": {
            "name": "fake",
            "source": "brainglobe",
            "resolution_um": _RES_UM,
            "shape": [_AP, _DV, _LR],
        },
        "interpolation_axis": "AP",
        "channels": [],
        "cp_size": 10,
        "cp_shape": "Cross",
        "cp_color": "#fff500",
        "working_scale": 0.5,
        "sections": sections,
    }


def _reg(project: dict) -> VersoRegistration:
    tmp = Path(tempfile.mkdtemp()) / "project-verso.json"
    tmp.write_text(json.dumps(project))
    return VersoRegistration(tmp)


# ---------------------------------------------------------------------------
# Case inputs
# ---------------------------------------------------------------------------

# (name, points_norm, src_px, dst_px, work_w, work_h) — shared by both warp dirs.
_WARP_INPUTS = [
    ("empty", [[0.3, 0.4], [0.7, 0.2]], [], [], 48, 32),
    ("identity", [[0.3, 0.4], [0.9, 0.1]], [[10.0, 8.0]], [[10.0, 8.0]], 48, 32),
    (
        "three_cp_square",
        [[0.4, 0.5], [0.1, 0.1], [0.95, 0.95], [0.5, 0.5]],
        [[10.0, 8.0], [30.0, 20.0], [40.0, 12.0]],
        [[14.0, 6.0], [26.0, 24.0], [38.0, 16.0]],
        48,
        32,
    ),
    (
        "aspect_wide",
        [[0.25, 0.75], [0.6, 0.3]],
        [[12.0, 8.0], [50.0, 10.0], [30.0, 28.0]],
        [[16.0, 6.0], [46.0, 14.0], [28.0, 26.0]],
        64,
        32,
    ),
    (
        "outside_hull",
        [[-0.4, 0.5], [1.5, 0.5], [0.5, 0.5]],
        [[10.0, 8.0], [30.0, 20.0], [40.0, 12.0]],
        [[14.0, 6.0], [26.0, 24.0], [38.0, 16.0]],
        48,
        32,
    ),
]

_ANCHORING_INPUTS = [
    ("canonical", _canonical_anchoring(10.0)),
    ("tilted", [1.0, 2.0, 3.0, 20.0, 1.0, 0.5, 0.0, -2.0, 15.0]),
]

_VOXEL_INPUTS = [
    ("boundaries", [[3.0, 4.0, 5.0], [3.5, 4.5, 5.5], [3.9, 4.1, 5.0]]),
    ("negatives", [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [11.2, 8.9, 7.1]]),
]


def _warp_cases(fn) -> list[dict]:
    cases = []
    for name, pts, src, dst, w, h in _WARP_INPUTS:
        out = fn(
            np.array(pts, dtype=np.float64).reshape(-1, 2),
            np.array(src, dtype=np.float64).reshape(-1, 2),
            np.array(dst, dtype=np.float64).reshape(-1, 2),
            w,
            h,
        )
        cases.append(
            {
                "name": name,
                "points_norm": pts,
                "src_px": src,
                "dst_px": dst,
                "work_w": w,
                "work_h": h,
                "expected": out.tolist(),
            }
        )
    return cases


def _anchoring_cases() -> list[dict]:
    cases = []
    for name, anch in _ANCHORING_INPUTS:
        o, u, v = anchoring_to_vectors(anch)
        cases.append({"name": name, "anchoring": anch, "ouv": np.vstack([o, u, v]).tolist()})
    return cases


def _voxel_cases() -> list[dict]:
    cases = []
    for name, coords in _VOXEL_INPUTS:
        arr = np.array(coords, dtype=np.float64)
        lr, ap, dv = _sample_voxel_indices(arr[:, 0], arr[:, 1], arr[:, 2])
        cases.append(
            {
                "name": name,
                "coords_lr_ap_dv": coords,
                "expected_idx": np.column_stack([lr, ap, dv]).tolist(),
            }
        )
    return cases


def _image_to_atlas_cases(projects: dict[str, dict]) -> list[dict]:
    specs = [
        ("plain_voxel", _section("s1", 10.0), [[30.0, 20.0], [70.0, 50.0]], "full", "voxel"),
        (
            "flips_voxel",
            _section("s1", 10.0, flip_h=True, flip_v=True),
            [[30.0, 20.0]],
            "full",
            "voxel",
        ),
        ("working_voxel", _section("s1", 10.0), [[12.0, 8.0]], "working", "voxel"),
        ("units_um", _section("s1", 10.0), [[30.0, 20.0]], "full", "um"),
        (
            "warp_cps",
            _section(
                "s1",
                10.0,
                control_points=[
                    {"src_x": 10.0, "src_y": 8.0, "dst_x": 14.0, "dst_y": 6.0},
                    {"src_x": 30.0, "src_y": 20.0, "dst_x": 26.0, "dst_y": 24.0},
                    {"src_x": 40.0, "src_y": 12.0, "dst_x": 38.0, "dst_y": 16.0},
                ],
                work=(48, 32),
                full=(48, 32),
            ),
            [[14.0, 6.0], [26.0, 24.0], [38.0, 16.0], [999.0, 20.0], [-5.0, 10.0]],
            "full",
            "voxel",
        ),
    ]
    cases = []
    for name, sec, pts, space, units in specs:
        project = _project([sec])
        rel = f"projects/img_{name}.json"
        projects[rel] = project
        reg = _reg(project)
        coords, inside = reg.coord_image_to_atlas(
            "s1", pts, space=space, units=units, return_valid=True
        )
        cases.append(
            {
                "name": name,
                "project_file": rel,
                "slice": "s1",
                "points": pts,
                "space": space,
                "units": units,
                "expected_xyz": coords.tolist(),
                "expected_inside": inside.tolist(),
            }
        )
    return cases


def _atlas_to_image_case(name, project, xyz, projects) -> dict:
    rel = f"projects/atlas_{name}.json"
    projects[rel] = project
    res = _reg(project).coord_atlas_to_image(xyz, space="full", units="voxel")
    return {
        "name": name,
        "project_file": rel,
        "xyz": xyz,
        "space": "full",
        "units": "voxel",
        "expected_section_id": [str(s) for s in res.section_id],
        "expected_valid": [bool(b) for b in res.valid],
        # Non-finite (uncovered) xy/distance are stored as 0.0 so the JSON stays
        # finite; the MATLAB side only compares them where ``valid`` is true.
        "expected_xy": np.where(np.isfinite(res.xy), res.xy, 0.0).tolist(),
        "expected_distance": [float(d) if np.isfinite(d) else 0.0 for d in res.distance],
    }


def _atlas_to_image_cases(projects: dict[str, dict]) -> list[dict]:
    return [
        # Covered: each voxel lies on a section's footprint → all valid.
        _atlas_to_image_case(
            "covered_nearest_plane",
            _project([_section("s1", 8.0), _section("s2", 14.0)]),
            [[12.0, 9.0, 8.0], [12.0, 12.0, 14.0]],
            projects,
        ),
        # Uncovered: no section footprint covers the voxel → section_id "", invalid.
        _atlas_to_image_case(
            "uncovered", _project([_section("s1", 8.0)]), [[100.0, 8.0, 8.0]], projects
        ),
    ]


def build_cases() -> tuple[dict, dict[str, dict]]:
    """Run the live Python engine over every parity case.

    Returns ``(cases, projects)`` where ``projects`` maps each coord case's
    committed project-file path (relative to the fixture dir) to its project
    dict, so the MATLAB side can load the identical project directly.
    """
    projects: dict[str, dict] = {}
    cases = {
        "version": 1,
        "tolerance": TOL,
        "warp_section_to_atlas": _warp_cases(warp_points_section_to_atlas),
        "warp_atlas_to_section": _warp_cases(warp_points_atlas_to_section),
        "anchoring_to_vectors": _anchoring_cases(),
        "sample_voxel_indices": _voxel_cases(),
        "coord_image_to_atlas": _image_to_atlas_cases(projects),
        "coord_atlas_to_image": _atlas_to_image_cases(projects),
    }
    return cases, projects


# ---------------------------------------------------------------------------
# Compare / test
# ---------------------------------------------------------------------------


def _assert_close(a, b, path: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: expected dict, got {type(b).__name__}"
        assert set(a) == set(b), f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: list length"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} != {b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert abs(float(a) - float(b)) <= TOL, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_parity_fixture_matches_engine():
    fresh, projects = build_cases()

    if os.environ.get("UPDATE_PARITY_FIXTURES"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        for rel, proj in projects.items():
            path = FIXTURE.parent / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(proj, indent=2) + "\n")
        FIXTURE.write_text(json.dumps(fresh, indent=2) + "\n")
        return

    assert FIXTURE.exists(), (
        f"Parity fixture missing: {FIXTURE}. "
        "Generate it with UPDATE_PARITY_FIXTURES=1 uv run pytest "
        "tests/engine/test_matlab_parity.py"
    )
    committed = json.loads(FIXTURE.read_text())
    _assert_close(fresh, committed)
    for rel in projects:
        assert (FIXTURE.parent / rel).exists(), f"missing committed project file: {rel}"
