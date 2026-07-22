"""Automatic non-rigid registration → biologically-meaningful control points.

This module estimates a non-linear (B-spline) deformation between the atlas
template and a section using ITK-Elastix, then samples that deformation only at
a curated set of 3D anchor lines so every generated control point lands on a
named structure (cortex, ventricles, hippocampus, midline, …). The result is a
set of Delaunay control points ready for the normal VERSO warp pipeline and for
further manual adjustment.

The affine alignment is already encoded in the section anchoring, so the atlas
slice arrives pre-aligned; only the residual non-linear deformation is recovered.

``itk`` is imported lazily inside the functions that need it: it is a heavy
import and must load *after* scipy (loading itk first can clash with scipy), so
deferring it keeps app startup fast and the import order safe.

Coordinate conventions
----------------------
* Control points are normalised ``[0, 1]``: ``src`` = atlas overlay space,
  ``dst`` = section image space (see :class:`~verso.engine.model.alignment.ControlPoint`).
* The registration runs in *working-resolution pixel* physical coordinates: when
  ``registration_scale < 1`` the images are downsampled but the itk pixel spacing
  is set to ``1/scale`` so the estimated transform is expressed in
  working-resolution pixels and can be applied directly at working resolution.
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from verso.engine.anchoring import anchoring_to_vectors, atlas_to_normalized
from verso.engine.model.alignment import ControlPoint
from verso.engine.model.elastix import ElastixParams
from verso.engine.preprocessing import morph_mask

_log = logging.getLogger(__name__)

# resources/ lives at the package root (src/verso/resources), alongside gui/engine.
_RESOURCES = Path(__file__).parent.parent / "resources"
_ANCHOR_POINTS_FILE = _RESOURCES / "anchor_points.json"

# Weight of the corresponding-points metric relative to the image metric when
# manual control points are supplied. Pulls the registration toward honoring the
# user's points. Hard to tune precisely; exposed here as a single knob.
_CORRESPONDING_POINTS_WEIGHT = "1.0"


def is_supported_atlas(atlas_name: str) -> bool:
    """Whether automatic control points are available for the given atlas.

    The curated anchor lines were traced in Allen mouse CCF space, so only Allen
    mouse atlases (any resolution) are supported.
    """
    return atlas_name.startswith("allen_mouse")


@lru_cache(maxsize=1)
def load_anchor_lines() -> dict:
    """Load the packaged curated anchor-line resource (cached).

    Returns a dict with ``resolution_um``, ``shape`` ([AP, DV, ML]), and
    ``lines`` (name → dense polyline of ``[z=AP, y=DV, x=ML]`` voxel coords in
    25 µm space). Mirroring and smoothing are already baked in at build time
    (see ``scripts/build_anchor_points.py``).
    """
    return json.loads(_ANCHOR_POINTS_FILE.read_text())


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Reduce an image to a normalised float32 grayscale array in [0, 1].

    Averages across channels (works for the RGB template slice and for
    multi-channel fluorescence sections alike).
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32)
    hi = float(arr.max())
    if hi > 0:
        arr = arr / hi
    return arr


def anchor_source_points(
    anchoring: list[float],
    atlas_shape: tuple[int, int, int],
    out_w: int,
    out_h: int,
    *,
    cp_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Atlas-space source control points where the curated lines cross the plane.

    Each curated anchor line is scaled from the 25 µm curation grid to the
    project atlas voxel grid, intersected with the section's cutting plane, and
    the crossing converted to normalised section coordinates ``(s, t)``. Only
    crossings inside the section ``[0, 1]`` and (if given) inside ``cp_mask`` are
    kept.

    Args:
        anchoring: 9-element section anchoring.
        atlas_shape: project atlas annotation shape ``(AP, DV, LR)``.
        out_w, out_h: working-resolution section dimensions (for the mask gate).
        cp_mask: optional bool ``(out_h, out_w)`` gating mask.

    Returns:
        ``(K, 2)`` float64 array of working-resolution pixel ``(x, y)`` source positions.
    """
    res = load_anchor_lines()
    curated_shape = np.asarray(res["shape"], dtype=np.float64)  # [AP, DV, ML]
    # Scale curated voxel coords (axis order [z=AP, y=DV, x=ML]) to the project
    # atlas grid. atlas_shape is (AP, DV, LR) which matches [AP, DV, ML].
    scale = np.asarray(atlas_shape, dtype=np.float64) / curated_shape

    o, u, v = anchoring_to_vectors(anchoring)
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros((0, 2))
    normal = normal / norm

    src_list: list[list[float]] = []
    for pts in res["lines"].values():
        dense = np.asarray(pts, dtype=np.float64)  # (N, 3) [AP, DV, ML]
        if len(dense) < 2:
            continue
        dense = dense * scale
        ml_ap_dv = dense[:, [2, 0, 1]]  # → anchoring order [ML, AP, DV]
        signed = (ml_ap_dv - o).dot(normal)
        for idx in np.where(np.diff(np.sign(signed)))[0]:
            span = signed[idx + 1] - signed[idx]
            if span == 0:
                continue
            frac = -signed[idx] / span
            cross = ml_ap_dv[idx] + frac * (ml_ap_dv[idx + 1] - ml_ap_dv[idx])
            s, t = atlas_to_normalized(cross, anchoring)
            if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
                continue
            if cp_mask is not None:
                rx = int(np.clip(round(s * out_w), 0, out_w - 1))
                ry = int(np.clip(round(t * out_h), 0, out_h - 1))
                if not cp_mask[ry, rx]:
                    continue
            src_list.append([s * out_w, t * out_h])

    return np.array(src_list, dtype=np.float64) if src_list else np.zeros((0, 2))


def _make_parameter_map(params: ElastixParams, with_points: bool):
    import itk

    po = itk.ParameterObject.New()
    m = po.GetDefaultParameterMap("bspline", params.n_resolutions)
    m["FinalGridSpacingInPhysicalUnits"] = [str(params.grid_spacing)]
    m["MaximumNumberOfIterations"] = [str(params.max_iterations)]
    m["NumberOfHistogramBins"] = ["32"]
    m["ImageSampler"] = ["RandomCoordinate"]
    m["NumberOfSpatialSamples"] = [str(params.n_samples)]
    m["NewSamplesEveryIteration"] = ["true"]
    m["Interpolator"] = ["BSplineInterpolator"]
    m["BSplineInterpolationOrder"] = ["1"]
    m["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    m["FinalBSplineInterpolationOrder"] = ["3"]

    if with_points:
        # Add a corresponding-points metric so the registration honors the
        # user's manually-placed control points.
        m["Registration"] = ["MultiMetricMultiResolutionRegistration"]
        m["Metric"] = [
            "AdvancedMattesMutualInformation",
            "CorrespondingPointsEuclideanDistanceMetric",
        ]
        m["Metric0Weight"] = ["1.0"]
        m["Metric1Weight"] = [_CORRESPONDING_POINTS_WEIGHT]
    else:
        m["Metric"] = ["AdvancedMattesMutualInformation"]

    po.AddParameterMap(m)
    return po


def _write_point_file(path: Path, pts: np.ndarray) -> None:
    """Write a 2D point set in elastix ``.txt`` 'point' (world-coordinate) format."""
    lines = ["point", str(len(pts))]
    lines += [f"{float(x)} {float(y)}" for x, y in pts]
    path.write_text("\n".join(lines) + "\n")


def auto_control_points(
    section_img: np.ndarray,
    template_img: np.ndarray,
    anchoring: list[float],
    atlas_shape: tuple[int, int, int],
    *,
    mask: np.ndarray | None = None,
    manual_cps: list[ControlPoint] | None = None,
    params: ElastixParams | None = None,
) -> list[ControlPoint]:
    """Generate automatic control points for one section via elastix.

    Args:
        section_img: working-resolution section image ``(H, W)`` or ``(H, W, C)``.
        template_img: atlas template slice at the same ``(H, W)`` (e.g. from
            :meth:`AtlasVolume.slice_reference`).
        anchoring: 9-element section anchoring.
        atlas_shape: project atlas annotation shape ``(AP, DV, LR)``.
        mask: optional bool ``(H, W)`` tissue mask (True = tissue). Used to gate
            the registration metric and where new control points are created.
        manual_cps: control points the user placed by hand. When non-empty they
            are fed to elastix as a corresponding-points constraint so the
            registration works around them. They are *not* returned (the caller
            keeps them); only freshly generated ``auto=True`` points are returned.
        params: registration parameters; defaults to :class:`ElastixParams`.

    Returns:
        New ``auto=True`` control points. May be empty if no curated line crosses
        the plane inside the mask.
    """
    import itk

    params = params or ElastixParams()
    manual_cps = [cp for cp in (manual_cps or []) if not cp.auto]

    gray_section = _to_gray(section_img)
    gray_template = _to_gray(template_img)
    h, w = gray_section.shape

    scale = float(params.registration_scale)
    if scale <= 0:
        scale = 1.0

    def _resize(img: np.ndarray) -> np.ndarray:
        if scale == 1.0:
            return img
        import cv2

        return cv2.resize(
            img, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA
        )

    section_small = _resize(gray_section)
    template_small = _resize(gray_template)
    spacing = (1.0 / scale, 1.0 / scale)

    fixed_itk = itk.image_from_array(np.ascontiguousarray(section_small, dtype=np.float32))
    moving_itk = itk.image_from_array(np.ascontiguousarray(template_small, dtype=np.float32))
    fixed_itk.SetSpacing(spacing)
    moving_itk.SetSpacing(spacing)

    # Dilated tissue mask gates the image metric so edge tissue still contributes.
    # A mask with no foreground is dropped rather than attached: an all-empty
    # fixed mask leaves the sampler with no voxels to draw and elastix aborts with
    # a generic "Internal elastix error", so register without a mask instead.
    fixed_mask_itk = None
    if mask is not None and np.any(mask):
        reg_mask = morph_mask(mask, params.mask_dilation_register, "expand")
        reg_mask_small = _resize(reg_mask.astype(np.float32)) > 0.5 if scale != 1.0 else reg_mask
        if np.any(reg_mask_small):
            mask_itk = itk.image_from_array(np.ascontiguousarray(reg_mask_small, dtype=np.uint8))
            mask_itk.SetSpacing(spacing)
            fixed_mask_itk = mask_itk

    po = _make_parameter_map(params, with_points=bool(manual_cps))

    kwargs: dict = {"log_to_console": False}
    if fixed_mask_itk is not None:
        kwargs["fixed_mask"] = fixed_mask_itk

    with tempfile.TemporaryDirectory() as tmp:
        if manual_cps:
            # Same anatomical feature: fixed = section (dst), moving = template (src).
            # Control points are stored in working-resolution pixels = the registration's
            # physical space; pass them directly.
            fixed_pts = np.array([[cp.dst_x, cp.dst_y] for cp in manual_cps])
            moving_pts = np.array([[cp.src_x, cp.src_y] for cp in manual_cps])
            fixed_file = Path(tmp) / "fixed_points.txt"
            moving_file = Path(tmp) / "moving_points.txt"
            _write_point_file(fixed_file, fixed_pts)
            _write_point_file(moving_file, moving_pts)
            kwargs["fixed_point_set_file_name"] = str(fixed_file)
            kwargs["moving_point_set_file_name"] = str(moving_file)

        _, tp = itk.elastix_registration_method(
            fixed_itk, moving_itk, parameter_object=po, **kwargs
        )

    # Source positions: curated anchor lines crossing the plane, gated by a
    # (larger) dilated mask so points only appear on/near tissue.
    cp_mask = morph_mask(mask, params.mask_dilation_cp, "expand") if mask is not None else None
    src_px = anchor_source_points(anchoring, atlas_shape, w, h, cp_mask=cp_mask)
    if len(src_px) == 0:
        return []

    # Destination positions: apply the transform at working resolution to the
    # identity coordinate ramps, then invert one Newton step: dst ≈ 2·src − T(src).
    tp.SetParameter(0, "Size", [str(w), str(h)])
    tp.SetParameter(0, "Spacing", ["1.0", "1.0"])
    tp.SetParameter(0, "ResampleInterpolator", "FinalLinearInterpolator")

    x_img = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    y_img = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    mapped_x = itk.array_from_image(itk.transformix_filter(itk.image_from_array(x_img), tp))
    mapped_y = itk.array_from_image(itk.transformix_filter(itk.image_from_array(y_img), tp))

    out: list[ControlPoint] = []
    for cx, cy in src_px:
        rx = int(np.clip(round(cx), 0, w - 1))
        ry = int(np.clip(round(cy), 0, h - 1))
        dst_x = 2.0 * cx - float(mapped_x[ry, rx])
        dst_y = 2.0 * cy - float(mapped_y[ry, rx])
        # Drop points the single-step inversion threw outside the image: near
        # tissue edges or in poorly-registered regions the elastix displacement
        # can be large and the destination lands far off-section (see module
        # docstring). Such points are unreliable, so discard rather than clamp.
        if not (0.0 <= dst_x <= w and 0.0 <= dst_y <= h):
            continue
        out.append(
            ControlPoint(src_x=float(cx), src_y=float(cy), dst_x=dst_x, dst_y=dst_y, auto=True)
        )
    return out


def prepare_registration_inputs(
    sections: list,
    atlas,
    working_scale: float,
) -> tuple[list[dict], list[str]]:
    """Build per-section registration inputs (arrays + metadata), in-process.

    Loads each section's working image, slices the atlas template at its
    anchoring, loads its tissue mask, and collects its manual control points.
    This is the *non-itk* preparation step: it is safe to run in a host worker
    thread, and lets the heavy registration run on arrays in a separate process
    without that process needing to reload the atlas.

    Returns ``(inputs, errors)`` where each input is a dict with keys
    ``id``, ``anchoring``, ``manual_cps`` (list[ControlPoint]), ``section``
    (H×W×C array), ``template`` (H×W×3 array), and ``mask`` (bool array or None).
    Per-section failures are collected rather than raised.
    """
    from verso.engine.io.image_io import ensure_working_copy
    from verso.engine.preprocessing import load_mask

    inputs: list[dict] = []
    errors: list[str] = []
    for section in sections:
        name = Path(section.original_path).name
        try:
            anchoring = section.alignment.current_anchoring
            if not section.alignment.is_anchored:
                errors.append(f"{name}: no alignment yet")
                continue
            image = ensure_working_copy(section, working_scale)
            if image is None:
                errors.append(f"{name}: no readable image")
                continue
            h, w = image.shape[:2]
            template = atlas.slice_reference(anchoring, w, h)
            mask = None
            mask_path = section.preprocessing.slice_mask_path
            if mask_path and Path(mask_path).exists():
                mask = load_mask(mask_path, (h, w))
            inputs.append(
                {
                    "id": section.id,
                    "anchoring": list(anchoring),
                    "manual_cps": [cp for cp in section.warp.control_points if not cp.auto],
                    "section": image,
                    "template": template,
                    "mask": mask,
                }
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return inputs, errors


# --- Persistent worker-process protocol (see _elastix_worker.py) -------------
# The elastix optimizer is native (C++) code that segfaults when run inside a
# host's worker thread (e.g. a Qt QThread). It runs in a long-lived child
# process instead, kept warm across calls so only the first registration pays
# the ~15 s cold template-instantiation cost.
_WORKER_READY = "VERSO_ELASTIX_READY"
_WORKER_DONE = "VERSO_ELASTIX_DONE"
_WORKER_QUIT = "VERSO_ELASTIX_QUIT"


class ElastixWorker:
    """Manages a persistent child process that runs elastix registrations.

    Spawn it (``start``) when entering a context where it may be used, so the
    child pays the one-time native warm-up in the background. Then call
    ``generate`` (from a background thread — it blocks until the child replies)
    to register a batch of prepared inputs. The process stays warm between
    calls; ``shutdown`` stops it on application exit.

    A child crash on a pathological job is contained: ``generate`` detects it,
    returns whatever partial results were written, and the next call respawns.
    """

    def __init__(self) -> None:
        import threading

        self._proc = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Ensure the worker process is running (non-blocking; child self-warms)."""
        self._ensure_started()

    def _ensure_started(self) -> None:
        import subprocess
        import sys

        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "verso.engine._elastix_worker", "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

    def _reset(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()

    def generate(
        self,
        inputs: list[dict],
        atlas_shape: tuple[int, int, int],
        params: ElastixParams,
    ) -> tuple[dict[str, list[ControlPoint]], list[str]]:
        """Register ``inputs`` in the warm child process.

        ``inputs`` is the output of :func:`prepare_registration_inputs`. Returns
        ``(results, errors)`` where ``results`` maps ``section.id`` to its new
        ``auto=True`` control points. Must be called from a background thread:
        it blocks until the child finishes.
        """
        import shutil
        import tempfile

        if not inputs:
            return {}, []

        _log.info("Registering %d section(s) via elastix worker", len(inputs))
        self._ensure_started()
        proc = self._proc
        tmp = Path(tempfile.mkdtemp(prefix="verso_elastix_"))
        try:
            sections_meta = []
            for i, inp in enumerate(inputs):
                np.save(tmp / f"section_{i}.npy", np.asarray(inp["section"]))
                np.save(tmp / f"template_{i}.npy", np.asarray(inp["template"]))
                has_mask = inp["mask"] is not None
                if has_mask:
                    np.save(tmp / f"mask_{i}.npy", np.asarray(inp["mask"]))
                sections_meta.append(
                    {
                        "id": inp["id"],
                        "index": i,
                        "anchoring": list(inp["anchoring"]),
                        "manual_cps": [cp.to_dict() for cp in inp["manual_cps"]],
                        "has_mask": has_mask,
                    }
                )
            job = {
                "atlas_shape": list(atlas_shape),
                "params": params.to_dict(),
                "sections": sections_meta,
            }
            (tmp / "job.json").write_text(json.dumps(job))

            try:
                proc.stdin.write(f"{tmp}\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._reset()
                raise RuntimeError("automatic-registration worker is not available") from None

            crashed = False
            while True:
                line = proc.stdout.readline()
                if line == "":  # child died (EOF)
                    self._reset()
                    crashed = True
                    break
                if line.strip() == _WORKER_DONE:
                    break
            if crashed:
                _log.warning("elastix worker died mid-batch; returning partial results")
            return self._read_result(tmp, crashed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _read_result(tmp: Path, crashed: bool) -> tuple[dict[str, list[ControlPoint]], list[str]]:
        result_file = tmp / "result.json"
        if not result_file.exists():
            msg = "automatic-registration worker crashed" if crashed else "no result produced"
            return {}, [msg]
        data = json.loads(result_file.read_text())
        results = {
            sid: [ControlPoint.from_dict(d) for d in cps]
            for sid, cps in data.get("results", {}).items()
        }
        errors = list(data.get("errors", []))
        if crashed:
            errors.append("automatic-registration worker crashed; partial results returned")
        return results, errors

    def shutdown(self) -> None:
        """Stop the worker process (call on application exit)."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(f"{_WORKER_QUIT}\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            proc.kill()
