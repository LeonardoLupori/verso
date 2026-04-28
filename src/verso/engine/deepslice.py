"""Optional DeepSlice proposal generation.

This module keeps DeepSlice out of VERSO's import path.  DeepSlice is run in a
separate Python environment and its QuickNII-compatible JSON output is parsed
back into lightweight suggestions that callers can apply atomically.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from verso.engine.io.quint_io import load_deepslice
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Project, Section


@dataclass
class DeepSliceOptions:
    """Options passed to the external DeepSlice runner."""

    species: str = "mouse"
    ensemble: bool = True
    section_numbers: bool = True
    propagate_angles: bool = True
    enforce_index_order: bool = True
    reverse_section_order: bool = False
    section_thickness: float | None = None
    timeout_seconds: int | None = None


@dataclass
class DeepSliceSectionSuggestion:
    """One affine suggestion returned by DeepSlice."""

    filename: str
    serial_number: int
    anchoring: list[float]
    confidence: float | None = None


@dataclass
class DeepSliceRunResult:
    """Suggestions and diagnostics from one DeepSlice run."""

    run_id: str
    suggestions: list[DeepSliceSectionSuggestion]
    stdout: str = ""
    stderr: str = ""
    output_json: str | None = None


class DeepSliceError(RuntimeError):
    """Raised when the optional DeepSlice runner cannot produce suggestions."""


def run_deepslice_suggestions(
    project: Project,
    python_executable: str | Path,
    options: DeepSliceOptions | None = None,
) -> DeepSliceRunResult:
    """Run DeepSlice in a separate Python environment and return suggestions.

    The input project is not modified.  Call :func:`apply_deepslice_suggestions`
    only after this function succeeds.
    """
    opts = options or DeepSliceOptions()
    run_id = uuid.uuid4().hex
    executable = str(python_executable)

    with TemporaryDirectory(prefix="verso-deepslice-") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "images"
        output_base = tmp_dir / "deepslice_predictions"
        input_dir.mkdir()

        _copy_registration_images(
            project.sections,
            input_dir,
            reverse_section_order=opts.reverse_section_order,
        )
        script = _runner_script()
        payload = {
            "folder": str(input_dir),
            "output_base": str(output_base),
            "species": opts.species,
            "ensemble": opts.ensemble,
            "section_numbers": opts.section_numbers,
            "propagate_angles": opts.propagate_angles,
            "enforce_index_order": opts.enforce_index_order,
            "section_thickness": opts.section_thickness,
        }

        try:
            completed = subprocess.run(
                [executable, "-c", script, json.dumps(payload)],
                check=False,
                capture_output=True,
                text=True,
                timeout=opts.timeout_seconds,
            )
        except OSError as exc:
            raise DeepSliceError(f"Cannot start DeepSlice Python executable: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise DeepSliceError("DeepSlice run timed out") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise DeepSliceError(
                f"DeepSlice failed with exit code {completed.returncode}: {detail}"
            )

        output_json = _find_deepslice_json(tmp_dir, output_base)
        suggestions = _load_suggestions(output_json)
        if not suggestions:
            raise DeepSliceError("DeepSlice completed but produced no usable suggestions")

        return DeepSliceRunResult(
            run_id=run_id,
            suggestions=suggestions,
            stdout=completed.stdout,
            stderr=completed.stderr,
            output_json=str(output_json),
        )


def apply_deepslice_suggestions(project: Project, result: DeepSliceRunResult) -> int:
    """Apply DeepSlice suggestions to matching project sections.

    Matching prefers copied filename stem, then serial number.  Matching
    sections become editable ``IN_PROGRESS`` alignments with DeepSlice metadata.
    """
    return apply_deepslice_suggestions_with_atlas(project, result, atlas_shape=None)


def apply_deepslice_suggestions_with_atlas(
    project: Project,
    result: DeepSliceRunResult,
    atlas_shape: tuple[int, int, int] | None,
) -> int:
    """Apply DeepSlice suggestions, converting from QuickNII convention if possible."""
    by_stem = {Path(s.thumbnail_path or s.original_path).stem: s for s in project.sections}
    by_original_stem = {Path(s.original_path).stem: s for s in project.sections}
    by_serial = {s.serial_number: s for s in project.sections}
    applied = 0

    for suggestion in result.suggestions:
        stem = Path(suggestion.filename).stem
        section = by_stem.get(stem) or by_original_stem.get(stem)
        if section is None:
            section = next(
                (s for s in project.sections if stem.startswith(f"{s.id}_")),
                None,
            )
        if section is None:
            section = by_serial.get(suggestion.serial_number)
        if section is None:
            continue
        anchoring = list(suggestion.anchoring)
        if atlas_shape is not None:
            from verso.engine.io.quint_io import _to_quicknii_convention

            anchoring = _to_quicknii_convention(anchoring, atlas_shape)
        section.alignment.anchoring = anchoring
        section.alignment.status = AlignmentStatus.IN_PROGRESS
        section.alignment.source = "deepslice"
        section.alignment.proposal_anchoring = list(anchoring)
        section.alignment.proposal_confidence = suggestion.confidence
        section.alignment.proposal_run_id = result.run_id
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED
        applied += 1

    return applied


def reset_in_progress_to_default_proposals(
    sections: list[Section],
    atlas_shape: tuple[int, int, int],
    reverse_ap: bool = False,
) -> int:
    """Clear editable suggestions and regenerate QuickNII-style default proposals."""
    from verso.engine.io.image_io import registration_dimensions
    from verso.engine.registration import quicknii_coronal_series_anchorings

    usable: list[tuple[Section, int, int]] = []
    for section in sections:
        try:
            w, h = registration_dimensions(section)
        except Exception:
            continue
        if w > 0 and h > 0:
            usable.append((section, w, h))

    if not usable:
        return 0

    stored_anchorings = [
        section.alignment.anchoring
        if section.alignment.status == AlignmentStatus.COMPLETE
        and section.alignment.anchoring
        and any(v != 0.0 for v in section.alignment.anchoring)
        else None
        for section, _, _ in usable
    ]
    propagated = quicknii_coronal_series_anchorings(
        image_sizes=[(w, h) for _, w, h in usable],
        serial_numbers=[section.serial_number for section, _, _ in usable],
        atlas_shape=atlas_shape,
        stored_anchorings=stored_anchorings,
        reverse_ap=reverse_ap,
    )

    changed = 0
    for (section, _, _), anchoring, stored in zip(usable, propagated, stored_anchorings):
        if stored is not None:
            continue
        section.alignment.anchoring = anchoring
        section.alignment.ap_position_mm = None
        section.alignment.status = AlignmentStatus.IN_PROGRESS
        section.alignment.source = "quicknii_default"
        section.alignment.proposal_anchoring = None
        section.alignment.proposal_confidence = None
        section.alignment.proposal_run_id = None
        section.warp.control_points.clear()
        section.warp.status = AlignmentStatus.NOT_STARTED
        changed += 1

    return changed


def _copy_registration_images(
    sections: list[Section],
    input_dir: Path,
    reverse_section_order: bool = False,
) -> None:
    copied = 0
    serials = sorted({section.serial_number for section in sections})
    reversed_serial_by_serial = {
        serial: reversed_serial
        for serial, reversed_serial in zip(serials, reversed(serials))
    }
    for section in sections:
        src = Path(section.thumbnail_path)
        if not src.exists():
            src = Path(section.original_path)
        if not src.exists():
            continue
        # DeepSlice's section-number mode expects filenames containing "_sXXX".
        # VERSO keeps user/cache filenames intact, so give DeepSlice stable
        # temporary names based on the parsed serial number and map results back
        # by that serial number after prediction.
        suffix = src.suffix or ".png"
        serial = (
            reversed_serial_by_serial[section.serial_number]
            if reverse_section_order
            else section.serial_number
        )
        dst = input_dir / f"{section.id}_s{serial:03d}{suffix}"
        if dst.exists():
            dst = input_dir / f"{section.id}_{copied}_s{serial:03d}{suffix}"
        shutil.copy2(src, dst)
        copied += 1
    if copied == 0:
        raise DeepSliceError("No readable registration images are available for DeepSlice")


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
        if payload["propagate_angles"]:
            model.propagate_angles()
        if payload["enforce_index_order"]:
            model.enforce_index_order()
        else:
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
    project = load_deepslice(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_sections = raw.get("slices") or raw.get("sections", [])
    suggestions: list[DeepSliceSectionSuggestion] = []

    for section, raw_section in zip(project.sections, raw_sections):
        anchoring = section.alignment.anchoring
        if not anchoring or not any(v != 0.0 for v in anchoring):
            continue
        confidence = raw_section.get("confidence")
        suggestions.append(
            DeepSliceSectionSuggestion(
                filename=section.original_path,
                serial_number=section.serial_number,
                anchoring=list(anchoring),
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return suggestions
