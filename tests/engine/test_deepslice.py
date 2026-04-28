import json
import subprocess
from pathlib import Path

from PIL import Image

from verso.engine.deepslice import (
    DeepSliceRunResult,
    DeepSliceSectionSuggestion,
    apply_deepslice_suggestions,
    apply_deepslice_suggestions_with_atlas,
    reset_in_progress_to_default_proposals,
    run_deepslice_suggestions,
)
from verso.engine.model.alignment import AlignmentStatus, ControlPoint
from verso.engine.model.project import AtlasRef, Project, Section


def _make_project(tmp_path: Path) -> Project:
    img1 = tmp_path / "s001.png"
    img2 = tmp_path / "s002.png"
    Image.new("RGB", (100, 80)).save(img1)
    Image.new("RGB", (100, 80)).save(img2)
    return Project(
        name="deep",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section("s001", 1, str(img1), str(img1)),
            Section("s002", 2, str(img2), str(img2)),
        ],
    )


def test_run_deepslice_suggestions_reads_generated_json(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_run(args, check, capture_output, text, timeout):
        payload = json.loads(args[3])
        copied = sorted(Path(payload["folder"]).glob("*.png"))
        assert copied[0].name == "s001_s001.png"
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps({
                "name": "out",
                "target": "allen_mouse_25um",
                "sections": [
                    {
                        "filename": copied[0].name,
                        "nr": 1,
                        "anchoring": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        "confidence": 0.8,
                    }
                ],
            })
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_deepslice_suggestions(project, "python")

    assert len(result.suggestions) == 1
    assert result.suggestions[0].filename == "s001_s001.png"
    assert result.suggestions[0].confidence == 0.8
    assert result.stdout == "ok"


def test_run_deepslice_failure_does_not_mutate_project(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)
    before = project.to_dict()

    def fake_run(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        run_deepslice_suggestions(project, "missing-python")
    except Exception:
        pass

    assert project.to_dict() == before


def test_apply_deepslice_suggestions_marks_editable_and_clears_warp(tmp_path: Path):
    project = _make_project(tmp_path)
    project.sections[0].warp.control_points.append(ControlPoint(0.1, 0.2, 0.3, 0.4))
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001.png",
                serial_number=1,
                anchoring=[1.0] * 9,
                confidence=0.5,
            )
        ],
    )

    applied = apply_deepslice_suggestions(project, result)

    s0 = project.sections[0]
    assert applied == 1
    assert s0.alignment.status == AlignmentStatus.IN_PROGRESS
    assert s0.alignment.source == "deepslice"
    assert s0.alignment.proposal_anchoring == [1.0] * 9
    assert s0.alignment.proposal_confidence == 0.5
    assert s0.alignment.proposal_run_id == "run-1"
    assert s0.warp.control_points == []


def test_apply_deepslice_suggestions_matches_temporary_id_filename(tmp_path: Path):
    project = _make_project(tmp_path)
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s999.png",
                serial_number=999,
                anchoring=[1.0] * 9,
            )
        ],
    )

    applied = apply_deepslice_suggestions(project, result)

    assert applied == 1
    assert project.sections[0].alignment.source == "deepslice"


def test_apply_deepslice_suggestions_converts_quicknii_convention(tmp_path: Path):
    project = _make_project(tmp_path)
    quicknii = [
        10.0, 428.0, 280.0,
        20.0, -3.0, -4.0,
        5.0, -6.0, -7.0,
    ]
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s001.png",
                serial_number=1,
                anchoring=quicknii,
            )
        ],
    )

    apply_deepslice_suggestions_with_atlas(
        project,
        result,
        atlas_shape=(528, 320, 456),
    )

    assert project.sections[0].alignment.anchoring == [
        10.0, 100.0, 40.0,
        20.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
    ]


def test_run_deepslice_can_reverse_temporary_section_numbers(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_run(args, check, capture_output, text, timeout):
        payload = json.loads(args[3])
        copied = sorted(path.name for path in Path(payload["folder"]).glob("*.png"))
        assert copied == ["s001_s002.png", "s002_s001.png"]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps({
                "name": "out",
                "target": "allen_mouse_25um",
                "sections": [
                    {"filename": copied[0], "nr": 2, "anchoring": [1.0] * 9},
                    {"filename": copied[1], "nr": 1, "anchoring": [2.0] * 9},
                ],
            })
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from verso.engine.deepslice import DeepSliceOptions

    result = run_deepslice_suggestions(
        project,
        "python",
        DeepSliceOptions(reverse_section_order=True),
    )

    assert [s.filename for s in result.suggestions] == ["s001_s002.png", "s002_s001.png"]


def test_reset_in_progress_to_default_proposals_clears_deepslice_metadata(tmp_path: Path):
    project = _make_project(tmp_path)
    for section in project.sections:
        section.alignment.anchoring = [1.0] * 9
        section.alignment.status = AlignmentStatus.IN_PROGRESS
        section.alignment.source = "deepslice"
        section.alignment.proposal_anchoring = [1.0] * 9
        section.alignment.proposal_confidence = 0.5
        section.warp.control_points.append(ControlPoint(0.1, 0.2, 0.3, 0.4))

    changed = reset_in_progress_to_default_proposals(
        project.sections,
        atlas_shape=(528, 320, 456),
    )

    assert changed == 2
    for section in project.sections:
        assert section.alignment.status == AlignmentStatus.IN_PROGRESS
        assert section.alignment.source == "quicknii_default"
        assert section.alignment.proposal_anchoring is None
        assert section.alignment.proposal_confidence is None
        assert section.warp.control_points == []
