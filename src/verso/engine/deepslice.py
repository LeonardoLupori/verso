"""Optional DeepSlice proposal generation.

DeepSlice is an optional extra (``uv sync --extra deepslice``).  It is run in
a subprocess using the same Python interpreter that runs VERSO, so TensorFlow's
DLLs are isolated from VERSO's process.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from verso.engine.anchoring import is_anchored
from verso.engine.model.project import Project, Section

_log = logging.getLogger(__name__)

# Side of the square input DeepSlice's Xception expects.  Its loader resizes
# whatever it is given to exactly (299, 299) — ignoring aspect ratio, with
# nearest-neighbour — but skips the resize entirely when the file already has
# that size (keras ``load_img``: ``if img.size != width_height_tuple``).  Staging
# at 299 therefore hands the network the exact array VERSO computed, and replaces
# a nearest-neighbour point-sample of a ~2000 px image (which discards ~99.98% of
# the pixels) with a proper area average.
#
# Verified against DeepSlice 1.2.8, which pyproject pins exactly.  If that pin is
# ever loosened, re-check ``neural_network.load_images_from_path``: a change to
# ``target_size`` would silently make this resize a lossy extra step rather than
# a no-op.
_DEEPSLICE_INPUT_SIDE = 299

# DeepSlice's mouse model always predicts in the Allen CCFv3 25 µm voxel grid,
# regardless of which atlas the VERSO project targets.  This is the reference
# volume shape (AP, DV, LR) those anchoring coordinates live in; on apply they
# are rescaled from it into the project atlas's own grid (see
# :func:`_rescale_anchoring_between_atlases`).  Skipping that rescale placed a
# 25 µm plane straight into e.g. a 10 µm project, landing it at the wrong depth
# and 2.5× too small.
_DEEPSLICE_REFERENCE_SHAPE: tuple[int, int, int] = (528, 320, 456)


@dataclass
class DeepSliceOptions:
    """Options passed to the DeepSlice runner."""

    species: str = "mouse"
    ensemble: bool = True
    section_numbers: bool = True
    propagate_angles: bool = True
    enforce_index_order: bool = True
    reverse_section_order: bool = False
    section_thickness: float | None = None
    bad_section_ids: list[str] = field(default_factory=list)
    # Gamma applied to the staged PNG before it is handed to DeepSlice.  Values
    # below 1 brighten midtones and compress highlights, which makes peaky
    # fluorescence (a few hot cells on dark tissue) look more like the broad,
    # roughly uniform autofluorescence DeepSlice was trained on (2pst / Nissl
    # imagery).  Set to 1.0 to disable.
    gamma: float = 0.5


@dataclass
class DeepSliceSectionSuggestion:
    """One affine suggestion returned by DeepSlice."""

    filename: str
    slice_index: int
    anchoring: list[float]
    confidence: float | None = None


@dataclass
class DeepSliceRunResult:
    """Suggestions and diagnostics from one DeepSlice run."""

    run_id: str
    suggestions: list[DeepSliceSectionSuggestion]
    # Section IDs the user flagged as bad before the run.  DeepSlice still
    # returns predictions for them (the model can't skip an image), but the
    # apply path discards those values and lets VERSO's standard interpolation
    # fill them in from the surrounding good sections.
    bad_section_ids: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    output_json: str | None = None


class DeepSliceError(RuntimeError):
    """Raised when the optional DeepSlice runner cannot produce suggestions."""


def run_deepslice_suggestions(
    project: Project,
    options: DeepSliceOptions | None = None,
) -> DeepSliceRunResult:
    """Run DeepSlice in a subprocess and return suggestions.

    DeepSlice (and TensorFlow) run in a child process using the same Python
    interpreter as VERSO, keeping TensorFlow's DLLs out of VERSO's address
    space.  Requires the ``deepslice`` extra: ``uv sync --extra deepslice``.

    The input project is not modified.  Call
    :func:`apply_deepslice_suggestions` only after this function succeeds.
    """
    opts = options or DeepSliceOptions()
    run_id = uuid.uuid4().hex
    executable = sys.executable
    _log.info(
        "Running DeepSlice on %d section(s) (species=%s, reverse=%s)",
        len(project.sections),
        opts.species,
        opts.reverse_section_order,
    )

    with TemporaryDirectory(prefix="verso-deepslice-") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "images"
        output_base = tmp_dir / "deepslice_predictions"
        input_dir.mkdir()

        copied = _copy_registration_images(
            project.sections,
            input_dir,
            working_scale=project.working_scale,
            reverse_section_order=opts.reverse_section_order,
            channels=project.channels,
            gamma=opts.gamma,
        )
        # DeepSlice identifies bad sections by filename substring.
        sorted_entries = sorted(copied, key=lambda t: t[0].name)
        bad_filenames: list[str] = []
        if opts.bad_section_ids:
            bad_set = set(opts.bad_section_ids)
            bad_filenames = [dst.name for dst, sid in sorted_entries if sid in bad_set]

        payload = {
            "folder": str(input_dir),
            "output_base": str(output_base),
            "species": opts.species,
            "ensemble": opts.ensemble,
            "section_numbers": opts.section_numbers,
            "propagate_angles": opts.propagate_angles,
            "enforce_index_order": opts.enforce_index_order,
            "section_thickness": opts.section_thickness,
            "bad_filenames": bad_filenames,
        }

        try:
            completed = subprocess.run(
                [executable, "-c", _runner_script(), json.dumps(payload)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise DeepSliceError(f"Cannot start DeepSlice subprocess: {exc}") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if "No module named" in detail and "DeepSlice" in detail:
                raise DeepSliceError("DeepSlice is not installed. Run: uv sync --extra deepslice")
            raise DeepSliceError(f"DeepSlice failed (exit {completed.returncode}):\n{detail}")

        output_json = _find_deepslice_json(tmp_dir, output_base)
        suggestions = _load_suggestions(output_json)
        if not suggestions:
            raise DeepSliceError("DeepSlice completed but produced no usable suggestions")
        _log.info("DeepSlice produced %d suggestion(s)", len(suggestions))

        return DeepSliceRunResult(
            run_id=run_id,
            suggestions=suggestions,
            bad_section_ids=list(opts.bad_section_ids),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def apply_deepslice_suggestions(project: Project, result: DeepSliceRunResult) -> set[str]:
    """Apply DeepSlice suggestions to matching project sections.

    Matching prefers copied filename stem, then serial number.  Matching
    sections become editable ``IN_PROGRESS`` alignments. Returns the ids of the
    sections whose alignment was set.
    """
    return apply_deepslice_suggestions_with_atlas(project, result, atlas_shape=None)


def apply_deepslice_suggestions_with_atlas(
    project: Project,
    result: DeepSliceRunResult,
    atlas_shape: tuple[int, int, int] | None,
    reverse_axis: bool = False,
) -> set[str]:
    """Apply DeepSlice suggestions to project sections.

    Suggestions from :func:`run_deepslice_suggestions` carry anchorings in raw
    QuickNII convention as written by DeepSlice, in the Allen 25 µm voxel grid
    (:data:`_DEEPSLICE_REFERENCE_SHAPE`).  Two steps convert each to the
    BrainGlobe convention VERSO uses internally at the project atlas's own
    resolution: first :func:`_rescale_anchoring_between_atlases` scales the
    voxel coordinates from the 25 µm grid into *atlas_shape*, then
    ``_to_quicknii_convention`` (self-inverse) mirrors the AP/DV axes.  Pass
    *atlas_shape* as ``(AP, DV, LR)`` voxel dimensions; if ``None`` neither step
    is applied.

    DeepSlice ≥1.2.7 auto-detects the indexing direction from image content
    (``enforce_section_ordering``), so the staged-filename ``_s{nr}`` reflection
    no longer controls anything and DeepSlice may order the AP series opposite
    to VERSO's own default proposals.  After conversion the series is therefore
    re-oriented to match VERSO's default-proposal direction for *reverse_axis*
    (see :func:`_orient_series_to_convention`) — the same convention that drives
    ``propagate_series_anchorings`` — so DeepSlice and the built-in proposals
    always scroll the same way.

    Sections listed in ``result.bad_section_ids`` get their DeepSlice
    prediction discarded.  Once the good predictions are in place, VERSO's
    standard series-interpolation fills the bad sections from their
    neighbours — far more reliable than trusting a network output the user
    already marked as untrustworthy.
    """
    import re

    from verso.engine.io.quint_io import _to_quicknii_convention

    by_stem = {Path(s.thumbnail_path or s.original_path).stem: s for s in project.sections}
    by_original_stem = {Path(s.original_path).stem: s for s in project.sections}
    by_slice_index = {s.slice_index: s for s in project.sections}
    bad_ids = set(result.bad_section_ids)
    applied_section_ids: set[str] = set()

    for suggestion in result.suggestions:
        stem = Path(suggestion.filename).stem
        section = by_stem.get(stem) or by_original_stem.get(stem)
        if section is None:
            # Staged-name format: ``{user_serial}[-counter]_s{deepslice_nr}``.
            # An older format kept a leading ``s`` on the prefix; accept both.
            m = re.match(r"^s?(\d+)(?:-\d+)?_s\d+$", stem)
            if m:
                section = by_slice_index.get(int(m.group(1)))
        if section is None:
            # Legacy staged-name format: ``{section.id}_s{nr}``
            section = next(
                (s for s in project.sections if stem.startswith(f"{s.id}_")),
                None,
            )
        if section is None:
            section = by_slice_index.get(suggestion.slice_index)
        if section is None:
            continue
        if section.id in bad_ids:
            # User flagged this section as bad — drop DeepSlice's prediction
            # and let the interpolation pass below fill it in.
            continue
        raw = list(suggestion.anchoring)
        if atlas_shape is not None:
            scaled = _rescale_anchoring_between_atlases(
                raw, _DEEPSLICE_REFERENCE_SHAPE, atlas_shape
            )
            anchoring = _to_quicknii_convention(scaled, atlas_shape)
        else:
            anchoring = raw
        section.alignment.set_auto_proposal(anchoring, source="deepslice")
        section.warp.reset()
        applied_section_ids.add(section.id)

    if atlas_shape is not None and applied_section_ids:
        # Re-orient the series to VERSO's convention *before* the bad sections
        # are interpolated, so they fill from correctly-oriented neighbours.
        _orient_series_to_convention(project, applied_section_ids, reverse_axis)

    filled_ids: set[str] = set()
    if atlas_shape is not None and bad_ids and applied_section_ids:
        filled_ids = _interpolate_bad_sections(project, atlas_shape, applied_section_ids, bad_ids)

    return applied_section_ids | filled_ids


def _rescale_anchoring_between_atlases(
    anchoring: list[float],
    src_shape: tuple[int, int, int],
    dst_shape: tuple[int, int, int],
) -> list[float]:
    """Rescale a 9-float anchoring from one atlas voxel grid to another.

    Anchoring components are atlas *voxel* coordinates, so moving to an atlas of
    a different resolution multiplies every component by the per-axis ratio of
    the two volumes' voxel extents (e.g. 25 µm → 10 µm is ×2.5 on every axis).
    DeepSlice always predicts in the Allen 25 µm grid; a project on a finer or
    coarser atlas needs this rescale *before* the QuickNII→BrainGlobe convention
    flip, or the plane lands at the wrong depth and the wrong scale.

    Component order is ``[x=LR, y=AP, z=DV]`` repeated for origin/u/v; both
    shapes are BrainGlobe ``(AP, DV, LR)``.  A no-op when the shapes match.

    Args:
        anchoring: 9-element anchoring in *src_shape*'s voxel grid.
        src_shape: Source atlas shape ``(AP, DV, LR)`` the anchoring lives in.
        dst_shape: Destination atlas shape ``(AP, DV, LR)``.

    Returns:
        The anchoring rescaled into *dst_shape*'s voxel grid.
    """
    ap_s, dv_s, lr_s = src_shape
    ap_d, dv_d, lr_d = dst_shape
    factors = (lr_d / lr_s, ap_d / ap_s, dv_d / dv_s)  # x=LR, y=AP, z=DV
    return [comp * factors[i % 3] for i, comp in enumerate(anchoring)]


def _orient_series_to_convention(
    project: Project,
    section_ids: set[str],
    reverse_axis: bool,
) -> None:
    """Align DeepSlice's AP series direction with VERSO's default proposals.

    DeepSlice is coronal-only, so the slicing axis is AP (anchoring voxel
    index 1).  ``propagate_series_anchorings`` places the first section
    (lowest ``slice_index``) at the *high* AP voxel and the last at AP 0 when
    ``reverse_axis`` is False, and the reverse when True.  DeepSlice's own
    ordering can come out either way, so if its AP trend across ``slice_index``
    disagrees with that convention the assignment is reversed.  Each section
    keeps its in-plane fit (tilt, stretch, ML/DV centre); only the AP centre is
    swapped with its mirror partner.  The AP *positions* DeepSlice predicted are
    preserved as a set — they are correct — only their order is corrected.
    """
    from verso.engine.anchoring import (
        anchoring_center,
        set_center_position_along_axis,
    )

    ap_axis = 1
    good = sorted(
        (s for s in project.sections if s.id in section_ids),
        key=lambda s: s.slice_index,
    )
    if len(good) < 2:
        return
    ap_centers = [float(anchoring_center(s.alignment.current_anchoring)[ap_axis]) for s in good]
    # VERSO convention: AP decreases with slice_index unless reverse_axis.
    ascending = ap_centers[-1] > ap_centers[0]
    if ascending == bool(reverse_axis):
        return  # already matches the built-in proposal direction
    for section, ap in zip(good, reversed(ap_centers), strict=False):
        flipped = set_center_position_along_axis(section.alignment.current_anchoring, ap, ap_axis)
        section.alignment.current_anchoring = flipped


def _interpolate_bad_sections(
    project: Project,
    atlas_shape: tuple[int, int, int],
    applied_section_ids: set[str],
    bad_ids: set[str],
) -> set[str]:
    """Fill bad-section anchorings by interpolating from the good DeepSlice ones.

    Returns the ids of the sections that were filled.
    """
    from verso.engine.anchoring import propagate_series_anchorings

    usable: list[tuple[Section, int, int]] = [
        (section, *section.resolution_thumbnail_wh) for section in project.sections
    ]
    if not usable:
        return set()

    stored = [
        section.alignment.current_anchoring if section.id in applied_section_ids else None
        for section, _, _ in usable
    ]
    if not any(a is not None for a in stored):
        return set()

    # DeepSlice is coronal-only, so this interpolation always runs along AP.
    propagated = propagate_series_anchorings(
        image_sizes=[(w, h) for _, w, h in usable],
        slice_indices=[s.slice_index for s, _, _ in usable],
        atlas_shape=atlas_shape,
        interpolation_axis=1,
        stored_anchorings=stored,
        center_proposals=False,
    )

    filled: set[str] = set()
    for (section, _, _), anchoring, st in zip(usable, propagated, stored, strict=False):
        if st is not None or section.id not in bad_ids:
            continue
        section.alignment.set_auto_proposal(anchoring, source="deepslice_bad_interpolated")
        section.warp.reset()
        filled.add(section.id)
    return filled


def _copy_registration_images(
    sections: list[Section],
    input_dir: Path,
    working_scale: float,
    reverse_section_order: bool = False,
    channels=None,
    gamma: float = 1.0,
) -> list[tuple[Path, str]]:
    """Write DeepSlice-ready PNGs and return (dst_path, section_id) pairs.

    DeepSlice predicts anchorings in the pixel space of the files it receives.
    VERSO registers against the working-resolution image, so the staged PNGs
    must be produced from that same working copy instead of opportunistically
    copying an original-resolution source.  The ``_s{serial}`` suffix is what
    DeepSlice's ``number_sections`` regex parses to recover ``nr``.

    When *channels* is supplied (the project's :class:`ChannelSpec` list), only
    the ``visible`` flags are read — see :func:`_format_deepslice_image` for why
    the rest of the display settings are deliberately ignored.
    """
    from PIL import Image

    from verso.engine.io.image_io import ensure_working_copy

    copied: list[tuple[Path, str]] = []
    slice_indices = [section.slice_index for section in sections]
    for section in sections:
        original_thumbnail_path = section.thumbnail_path
        try:
            img = ensure_working_copy(section, working_scale)
        finally:
            section.thumbnail_path = original_thumbnail_path
        if img is None:
            continue
        img = _format_deepslice_image(
            img,
            flip_horizontal=section.preprocessing.flip_horizontal,
            flip_vertical=section.preprocessing.flip_vertical,
            channels=channels,
            gamma=gamma,
        )
        img = _resize_for_deepslice(img)
        deepslice_nr = _deepslice_section_number(
            section.slice_index, slice_indices, reverse_section_order
        )
        # Filename layout: ``{slice_index}_s{deepslice_nr}.png``.
        # - Prefix is the user's true (possibly non-contiguous) slice index so
        #   they recognise it in the staged folder.
        # - Suffix is the only ``_s\d+`` token in the name, which is what
        #   DeepSlice's ``number_sections`` regex picks up.  Reverse maps the
        #   suffix to ``min+max-index`` so the reflection is visible.
        dst = input_dir / f"{section.slice_index:03d}_s{deepslice_nr:03d}.png"
        if dst.exists():
            dst = input_dir / f"{section.slice_index:03d}-{len(copied)}_s{deepslice_nr:03d}.png"
        Image.fromarray(img).save(dst)
        copied.append((dst, section.id))

    if not copied:
        raise DeepSliceError("No readable registration images are available for DeepSlice")
    return copied


def _deepslice_section_number(
    slice_index: int,
    slice_indices: list[int],
    reverse_section_order: bool,
) -> int:
    """Return the section number encoded in the staged DeepSlice filename.

    When the physical series is ordered posterior-to-anterior, DeepSlice still
    needs numbers increasing in its expected direction.  Reflecting the slice
    index around the first/last index preserves gaps from missing sections
    (e.g. 10, 20, 40 -> 40, 30, 10).
    """
    if not reverse_section_order or len(slice_indices) < 2:
        return slice_index
    return min(slice_indices) + max(slice_indices) - slice_index


def _resize_for_deepslice(image):
    """Area-resize a staged image to the square DeepSlice feeds its network.

    Deliberately square and aspect-distorting: that is what DeepSlice's own
    loader does (``target_size=(299, 299)`` with ``keep_aspect_ratio=False``),
    and what its training images went through — *"All input images were
    grayscaled and downsampled to 299 × 299 pixels"* (Carey et al., Nat Commun
    2023).  Preserving the aspect ratio here would move the data *away* from the
    training distribution, not toward it.

    ``INTER_AREA`` is the one deviation: DeepSlice resizes with nearest, which at
    these reduction factors is close to random point-sampling.  The paper does
    not state the interpolation used at training time, so this is a considered
    bet rather than a proven match — see :const:`_DEEPSLICE_INPUT_SIDE`.
    """
    import cv2
    import numpy as np

    side = _DEEPSLICE_INPUT_SIDE
    arr = np.asarray(image)
    if arr.shape[:2] == (side, side):
        return arr
    resized = cv2.resize(arr, (side, side), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def _format_deepslice_image(
    image,
    *,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    channels=None,
    gamma: float = 1.0,
):
    """Convert VERSO's working HWC image into a greyscale uint8 PNG payload.

    DeepSlice greyscales everything it is given before inference — its generator
    applies ``skimage.color.rgb2gray`` (``0.2125·R + 0.7154·G + 0.0721·B``) and
    feeds the result to Xception as three identical planes.  Training used the
    same preprocessing: *"All input images were grayscaled and downsampled to
    299 × 299 pixels"* (Carey et al., Nat Commun 2023).  Emitting greyscale here
    is therefore lossless with respect to what the network can see, and — since
    those three weights sum to 1.0 — makes ``rgb2gray`` an identity, so a channel's
    display colour can no longer act as a hidden per-channel weight.

    Only ``ChannelSpec.visible`` is honoured.  ``color``, ``scale`` and ``gamma``
    are display-only: letting them through made the predicted alignment depend on
    wherever the user last left a brightness slider.  Dropping ``scale`` costs no
    contrast protection either, because :func:`~verso.engine.io.image_io.ensure_working_copy`
    has already percentile-stretched every stored plane (1st–99.8th) and clipped
    its outliers.

    Visible planes are combined with a per-pixel ``max`` (Fiji-style, and what
    ``load_filmstrip_thumbnail`` / ``detect_foreground`` already do).  Note that
    the upstream stretch is *per channel*, so a sparse signal channel is scaled to
    the same full range as a dense anatomical one and its dots will reach white.
    That is in-distribution — the Allen Connectivity training images are bright
    sparse reporter over dim autofluorescence — but hiding signal channels so
    DeepSlice sees anatomy alone is closer still to the synthetic sections cut
    from single scalar volumes that dominate the training set.

    Every section takes this one path, including JPEG/PNG imports whose planes
    were once a photograph's R/G/B.  There is no photograph left to preserve by
    then: ``ensure_working_copy`` percentile-stretches each plane *independently*,
    so the relationship between the guns is gone before this function ever sees
    them, and a branch keyed on the original file's extension would be reading a
    fact about a file VERSO stopped opening at import.  Reducing everything the
    same way is also what makes ``visible`` mean the same thing everywhere — for
    an RGB import, hiding two guns is a real (and for a saturated stain, useful)
    lever rather than a no-op.

    *gamma* < 1 lifts midtones and compresses highlights, which makes a few-hot-
    cells fluorescence image look more like a 2pst/Nissl autofluorescence frame
    (closer to what DeepSlice was trained on).  It survives DeepSlice's own
    ``x /= std(x)`` normalisation, which cancels only linear scaling.
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        out = arr[:, :, 0]
    elif arr.ndim == 3:
        out = _visible_max_projection(arr, channels)
    else:
        raise DeepSliceError(f"Unsupported image shape for DeepSlice: {arr.shape}")

    out = out.astype(np.uint8, copy=False)
    if gamma is not None and gamma > 0 and gamma != 1.0:
        # Apply gamma in [0, 1] space, then map back to uint8.
        normalized = out.astype(np.float32) / 255.0
        out = (np.power(normalized, float(gamma)) * 255.0).clip(0, 255).astype(np.uint8)

    if flip_horizontal:
        out = np.flip(out, axis=1)
    if flip_vertical:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def _visible_max_projection(arr, channels):
    """Max-project the planes of ``arr`` whose :class:`ChannelSpec` is visible.

    Falls back to projecting every plane when no specs are supplied or when the
    user has hidden all of them — an all-hidden project used to stage pure black
    PNGs and produce silently meaningless predictions.

    Specs are matched to planes positionally, the same convention
    :func:`~verso.engine.preprocessing.composite_channels` uses; surplus planes
    and surplus specs are both ignored.
    """
    specs = list(channels or [])
    n = min(arr.shape[2], len(specs))
    keep = [i for i in range(n) if getattr(specs[i], "visible", True)]
    if not keep:
        return arr.max(axis=2)
    return arr[:, :, keep].max(axis=2)


def _runner_script() -> str:
    return textwrap.dedent(
        """
        import json
        import sys

        from DeepSlice import DSModel

        payload = json.loads(sys.argv[1])
        model = DSModel(payload["species"])

        model.predict(
            payload["folder"],
            ensemble=payload["ensemble"],
            section_numbers=payload["section_numbers"],
        )
        if payload["bad_filenames"]:
            model.set_bad_sections(payload["bad_filenames"])
        if payload["propagate_angles"]:
            model.propagate_angles()
        if payload["enforce_index_order"]:
            model.enforce_index_order()
        elif payload["section_thickness"] is not None:
            model.enforce_index_spacing(section_thickness=payload["section_thickness"])
        model.save_predictions(payload["output_base"])
        """
    )


def _find_deepslice_json(tmp_dir: Path, output_base: Path) -> Path:
    candidates = [
        output_base,
        output_base.with_suffix(".json"),
        output_base.parent / f"{output_base.name}.json",
    ]
    candidates.extend(sorted(tmp_dir.rglob("*.json")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise DeepSliceError("DeepSlice did not write a JSON predictions file")


def _load_suggestions(path: Path) -> list[DeepSliceSectionSuggestion]:
    """Read raw QuickNII-convention anchorings from the DeepSlice output JSON.

    Values are returned as-is (QuickNII convention).  Conversion to the
    BrainGlobe convention used by VERSO happens in
    :func:`apply_deepslice_suggestions_with_atlas` once the atlas shape is
    known.  Reading directly from the JSON avoids the ``load_quicknii`` atlas-
    name lookup which silently skips conversion for DeepSlice target strings
    like ``"ABA_Mouse_CCFv3_2017_25um.cutlas"``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_sections = raw.get("slices") or raw.get("sections", [])
    suggestions: list[DeepSliceSectionSuggestion] = []

    for i, raw_section in enumerate(raw_sections):
        anchoring = [float(x) for x in raw_section.get("anchoring", [])]
        if len(anchoring) < 9 or not is_anchored(anchoring):
            continue
        confidence = raw_section.get("confidence")
        suggestions.append(
            DeepSliceSectionSuggestion(
                filename=raw_section.get("filename", ""),
                slice_index=int(raw_section.get("nr", i + 1)),
                anchoring=anchoring,
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return suggestions
