import contextlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from verso.engine.anchoring import (
    pack_series_anchoring,
    reset_in_progress_to_default_proposals,
    unpack_series_anchoring,
)
from verso.engine.deepslice import (
    DeepSliceOptions,
    DeepSliceRunResult,
    DeepSliceSectionSuggestion,
    apply_deepslice_suggestions,
    apply_deepslice_suggestions_with_atlas,
    run_deepslice_suggestions,
)
from verso.engine.model.alignment import AlignmentStatus, ControlPoint
from verso.engine.model.project import AtlasRef, ChannelSpec, Preprocessing, Project, Section


def _make_project(tmp_path: Path) -> Project:
    img1 = tmp_path / "s001.png"
    img2 = tmp_path / "s002.png"
    Image.new("RGB", (100, 80)).save(img1)
    Image.new("RGB", (100, 80)).save(img2)
    return Project(
        name="deep",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section("s001", 1, str(img1), str(img1), resolution_thumbnail_wh=(100, 80)),
            Section("s002", 2, str(img2), str(img2), resolution_thumbnail_wh=(100, 80)),
        ],
    )


def test_run_deepslice_suggestions_reads_generated_json(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(Path(payload["folder"]).glob("*.png"))
        assert payload["section_numbers"] is True
        assert payload["propagate_angles"] is True
        assert payload["enforce_index_order"] is True
        assert copied[0].name == "001_s001.png"
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [
                        {
                            "filename": copied[0].name,
                            "nr": 1,
                            "anchoring": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            "confidence": 0.8,
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_deepslice_suggestions(project)

    assert len(result.suggestions) == 1
    assert result.suggestions[0].anchoring == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert result.suggestions[0].confidence == 0.8
    assert result.stdout == "ok"


def test_run_deepslice_failure_does_not_mutate_project(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)
    before = project.to_dict()

    def fake_run(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with contextlib.suppress(Exception):
        run_deepslice_suggestions(project)

    assert project.to_dict() == before


def test_apply_deepslice_suggestions_marks_editable_and_clears_warp(tmp_path: Path):
    project = _make_project(tmp_path)
    project.sections[0].warp.control_points.append(ControlPoint(0.1, 0.2, 0.3, 0.4))
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001.png",
                slice_index=1,
                anchoring=[1.0] * 9,
                confidence=0.5,
            )
        ],
    )

    touched = apply_deepslice_suggestions(project, result)

    s0 = project.sections[0]
    assert touched == {s0.id}
    assert s0.alignment.status == AlignmentStatus.IN_PROGRESS
    assert s0.alignment.source == "deepslice"
    assert s0.warp.control_points == []


def test_apply_deepslice_suggestions_matches_temporary_id_filename(tmp_path: Path):
    project = _make_project(tmp_path)
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s999.png",
                slice_index=999,
                anchoring=[1.0] * 9,
            )
        ],
    )

    touched = apply_deepslice_suggestions(project, result)

    assert len(touched) == 1
    assert project.sections[0].alignment.source == "deepslice"


def test_apply_deepslice_suggestions_converts_quicknii_convention(tmp_path: Path):
    project = _make_project(tmp_path)
    # QuickNII convention: origin flips about N-1 (527-100=427, 319-40=279).
    quicknii = [
        10.0,
        427.0,
        279.0,
        20.0,
        -3.0,
        -4.0,
        5.0,
        -6.0,
        -7.0,
    ]
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s001.png",
                slice_index=1,
                anchoring=quicknii,
            )
        ],
    )

    apply_deepslice_suggestions_with_atlas(
        project,
        result,
        atlas_shape=(528, 320, 456),
    )

    assert project.sections[0].alignment.current_anchoring == [
        10.0,
        100.0,
        40.0,
        20.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]


def test_apply_deepslice_suggestions_rescales_for_non_25um_atlas(tmp_path: Path):
    """DeepSlice always predicts in the Allen 25 µm grid.

    Applying it to a 10 µm project must rescale the voxel coordinates by the
    per-axis extent ratio (×2.5) *before* the QuickNII→BrainGlobe convention
    flip, or the plane lands 2.5× too small and at the wrong depth. Without the
    rescale a 25 µm ``oz`` was flipped against the 10 µm ``dv_max``, placing the
    section hundreds of voxels away from where it belongs.
    """
    project = _make_project(tmp_path)
    project.atlas = AtlasRef(name="allen_mouse_10um")
    # Raw DeepSlice (25 µm QuickNII) anchoring.
    raw = [10.0, 400.0, 100.0, 20.0, -3.0, -4.0, 5.0, -6.0, -7.0]
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s001.png",
                slice_index=1,
                anchoring=raw,
            )
        ],
    )

    # 10 µm atlas shape (AP, DV, LR) — exactly 2.5× the 25 µm reference.
    apply_deepslice_suggestions_with_atlas(
        project,
        result,
        atlas_shape=(1320, 800, 1140),
    )

    # rescale ×2.5 → [25, 1000, 250, 50, -7.5, -10, 12.5, -15, -17.5], then flip
    # AP/DV about N-1: oy = 1319 - 1000 = 319, oz = 799 - 250 = 549.
    assert project.sections[0].alignment.current_anchoring == [
        25.0,
        319.0,
        549.0,
        50.0,
        7.5,
        10.0,
        12.5,
        15.0,
        17.5,
    ]


def test_apply_deepslice_suggestions_does_not_mirror_for_reversed_section_order(
    tmp_path: Path,
):
    """DeepSlice predicts each image's AP from pixel content, and
    ``enforce_section_ordering`` re-sorts predictions monotonically by ``nr``.
    Both happen regardless of how VERSO renames filenames for the reversed
    case, so the QuickNII→BG convention conversion is the only adjustment
    needed on apply.
    """
    project = _make_project(tmp_path)
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=[
            DeepSliceSectionSuggestion(
                filename="s001_s002.png",
                slice_index=2,
                anchoring=[10.0, 427.0, 279.0, 20.0, -3.0, -4.0, 5.0, -6.0, -7.0],
            )
        ],
    )

    apply_deepslice_suggestions_with_atlas(
        project,
        result,
        atlas_shape=(528, 320, 456),
    )

    assert project.sections[0].alignment.current_anchoring == [
        10.0,
        100.0,
        40.0,
        20.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]


def test_run_deepslice_can_reverse_temporary_section_numbers(tmp_path: Path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(path.name for path in Path(payload["folder"]).glob("*.png"))
        assert copied == ["001_s002.png", "002_s001.png"]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [
                        {"filename": copied[0], "nr": 2, "anchoring": [1.0] * 9},
                        {"filename": copied[1], "nr": 1, "anchoring": [2.0] * 9},
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from verso.engine.deepslice import DeepSliceOptions

    result = run_deepslice_suggestions(
        project,
        DeepSliceOptions(reverse_section_order=True),
    )

    assert [s.filename for s in result.suggestions] == ["001_s002.png", "002_s001.png"]


def test_run_deepslice_reverses_section_numbers_by_reflection_preserving_gaps(
    tmp_path: Path,
    monkeypatch,
):
    image_paths = []
    for serial in (10, 20, 40):
        path = tmp_path / f"s{serial:03d}.png"
        Image.new("RGB", (100, 80)).save(path)
        image_paths.append(path)
    project = Project(
        name="deep",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section("s010", 10, str(image_paths[0]), str(image_paths[0])),
            Section("s020", 20, str(image_paths[1]), str(image_paths[1])),
            Section("s040", 40, str(image_paths[2]), str(image_paths[2])),
        ],
    )

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(path.name for path in Path(payload["folder"]).glob("*.png"))
        assert copied == ["010_s040.png", "020_s030.png", "040_s010.png"]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [
                        {"filename": name, "nr": i + 1, "anchoring": [1.0] * 9}
                        for i, name in enumerate(copied)
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from verso.engine.deepslice import DeepSliceOptions

    run_deepslice_suggestions(
        project,
        DeepSliceOptions(reverse_section_order=True),
    )


def test_apply_deepslice_discards_bad_predictions_and_interpolates(tmp_path: Path):
    """Bad sections keep no DeepSlice prediction and instead get an
    anchoring interpolated from the neighbouring good sections."""
    image_paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1:03d}.png"
        Image.new("RGB", (100, 80)).save(path)
        image_paths.append(path)
    project = Project(
        name="bad-interp",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                "s001",
                1,
                str(image_paths[0]),
                str(image_paths[0]),
                resolution_thumbnail_wh=(100, 80),
            ),
            Section(
                "s002",
                2,
                str(image_paths[1]),
                str(image_paths[1]),
                resolution_thumbnail_wh=(100, 80),
            ),
            Section(
                "s003",
                3,
                str(image_paths[2]),
                str(image_paths[2]),
                resolution_thumbnail_wh=(100, 80),
            ),
        ],
    )

    # Suggestions for all three sections; the middle one is flagged as bad so
    # its predicted (extreme) anchoring should be discarded.
    suggestions = [
        DeepSliceSectionSuggestion(
            filename="001_s001.png",
            slice_index=1,
            anchoring=[228.0, 100.0, 160.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0],
        ),
        DeepSliceSectionSuggestion(
            filename="002_s002.png",
            slice_index=2,
            anchoring=[0.0, 999.0, 0.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
        ),
        DeepSliceSectionSuggestion(
            filename="003_s003.png",
            slice_index=3,
            anchoring=[228.0, 400.0, 160.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0],
        ),
    ]
    result = DeepSliceRunResult(
        run_id="run-1",
        suggestions=suggestions,
        bad_section_ids=["s002"],
    )

    touched = apply_deepslice_suggestions_with_atlas(
        project,
        result,
        atlas_shape=(528, 320, 456),
    )
    assert len(touched) == 3  # 2 from DeepSlice, 1 interpolated.

    s2 = project.sections[1]
    assert s2.alignment.source == "deepslice_bad_interpolated"
    # The bogus 999 values must not have leaked through.
    assert all(abs(v) < 600 for v in s2.alignment.current_anchoring)
    # AP should land between the two neighbours' AP values after the QN->BG
    # convention flip applied to the good predictions.
    s1_ap = project.sections[0].alignment.current_anchoring[1]
    s3_ap = project.sections[2].alignment.current_anchoring[1]
    s2_ap = s2.alignment.current_anchoring[1]
    assert min(s1_ap, s3_ap) <= s2_ap <= max(s1_ap, s3_ap)


def test_apply_deepslice_orients_series_to_verso_convention(tmp_path: Path):
    """DeepSlice's AP order is realigned to VERSO's default-proposal direction.

    DeepSlice 1.2.7 auto-detects indexing direction, so it can return a series
    that scrolls opposite to VERSO's own default proposals.  Apply re-orients
    it: with ``reverse_axis=False`` the AP centre must *decrease* with
    ``slice_index`` (VERSO's default), and with ``reverse_axis=True`` it must
    *increase* — regardless of how DeepSlice happened to order the input.
    """
    from verso.engine.anchoring import anchoring_center

    def _build() -> Project:
        paths = []
        for i in range(3):
            p = tmp_path / f"s{i + 1:03d}.png"
            Image.new("RGB", (100, 80)).save(p)
            paths.append(p)
        return Project(
            name="orient",
            atlas=AtlasRef(name="allen_mouse_25um"),
            sections=[
                Section("s001", 1, str(paths[0]), str(paths[0])),
                Section("s002", 2, str(paths[1]), str(paths[1])),
                Section("s003", 3, str(paths[2]), str(paths[2])),
            ],
        )

    # Raw QuickNII oy DECREASING with slice_index. After the QN→BG flip this
    # yields BG AP INCREASING with slice_index — i.e. opposite to VERSO's
    # default proposal direction, so reverse_axis=False must reorder it.
    suggestions = [
        DeepSliceSectionSuggestion(
            filename=f"{i + 1:03d}_s{i + 1:03d}.png",
            slice_index=i + 1,
            anchoring=[10.0, 250.0 - 100.0 * i, 20.0, 100.0, 0.0, 0.0, 0.0, 0.0, 80.0],
        )
        for i in range(3)
    ]
    atlas_shape = (528, 320, 456)

    forward = _build()
    apply_deepslice_suggestions_with_atlas(
        forward, DeepSliceRunResult("r", suggestions), atlas_shape, reverse_axis=False
    )
    reversed_ = _build()
    apply_deepslice_suggestions_with_atlas(
        reversed_, DeepSliceRunResult("r", suggestions), atlas_shape, reverse_axis=True
    )

    fwd_ap = [anchoring_center(s.alignment.current_anchoring)[1] for s in forward.sections]
    rev_ap = [anchoring_center(s.alignment.current_anchoring)[1] for s in reversed_.sections]

    # reverse_axis=False → AP decreases with slice_index (VERSO default).
    assert fwd_ap[0] > fwd_ap[1] > fwd_ap[2]
    # reverse_axis=True → AP increases with slice_index.
    assert rev_ap[0] < rev_ap[1] < rev_ap[2]
    # Same set of predicted planes, only the order differs.
    assert sorted(fwd_ap) == sorted(rev_ap)


def test_run_deepslice_uses_user_serial_as_filename_prefix(tmp_path: Path, monkeypatch):
    """Staged PNG names lead with the user's true serial number (which may be
    non-contiguous), not VERSO's internal sequential ``section.id``."""
    image_paths = []
    for serial in (10, 30, 50):
        path = tmp_path / f"s{serial:03d}.png"
        Image.new("RGB", (50, 40)).save(path)
        image_paths.append(path)
    project = Project(
        name="non-contig",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section("s001", 10, str(image_paths[0]), str(image_paths[0])),
            Section("s002", 30, str(image_paths[1]), str(image_paths[1])),
            Section("s003", 50, str(image_paths[2]), str(image_paths[2])),
        ],
    )

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(p.name for p in Path(payload["folder"]).glob("*.png"))
        # Prefix is the user's serial, not s001/s002/s003.
        assert copied == ["010_s010.png", "030_s030.png", "050_s050.png"]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [
                        {"filename": copied[0], "nr": 10, "anchoring": [1.0] * 9},
                        {"filename": copied[1], "nr": 30, "anchoring": [2.0] * 9},
                        {"filename": copied[2], "nr": 50, "anchoring": [3.0] * 9},
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_deepslice_suggestions(project)
    touched = apply_deepslice_suggestions(project, result)
    assert len(touched) == 3
    # Each section keeps its own anchoring — matched via the serial prefix.
    assert project.sections[0].alignment.current_anchoring == [1.0] * 9  # serial 10
    assert project.sections[2].alignment.current_anchoring == [3.0] * 9  # serial 50


def test_run_deepslice_reverse_visible_in_filenames_for_non_contiguous_serials(
    tmp_path: Path,
    monkeypatch,
):
    """When ``reverse_section_order`` is set, the staged-name suffix differs
    from the prefix so the reflection is visible in the folder."""
    image_paths = []
    for serial in (10, 30, 50):
        path = tmp_path / f"s{serial:03d}.png"
        Image.new("RGB", (50, 40)).save(path)
        image_paths.append(path)
    project = Project(
        name="non-contig-rev",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section("s001", 10, str(image_paths[0]), str(image_paths[0])),
            Section("s002", 30, str(image_paths[1]), str(image_paths[1])),
            Section("s003", 50, str(image_paths[2]), str(image_paths[2])),
        ],
    )

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(p.name for p in Path(payload["folder"]).glob("*.png"))
        # Prefix = user serial, suffix = reflected DeepSlice nr.
        assert copied == ["010_s050.png", "030_s030.png", "050_s010.png"]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [
                        {
                            "filename": name,
                            "nr": int(name.split("_s")[-1].split(".")[0]),
                            "anchoring": [1.0] * 9,
                        }
                        for name in copied
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from verso.engine.deepslice import DeepSliceOptions

    run_deepslice_suggestions(
        project,
        DeepSliceOptions(reverse_section_order=True),
    )


def test_run_deepslice_applies_gamma_to_staged_png(tmp_path: Path, monkeypatch):
    """Gamma < 1 brightens midtones so peaky fluorescence flattens out.

    A flat mid-gray (128) image with gamma=0.5 should map to
    ``round(255 * (128/255)**0.5)`` = ~181, well above 128.
    """
    img_path = tmp_path / "s001.png"
    Image.new("RGB", (50, 40), color=(128, 128, 128)).save(img_path)
    project = Project(
        name="gamma",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[Section("s001", 1, str(img_path), str(img_path))],
    )

    captured: dict[str, int] = {}

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(Path(payload["folder"]).glob("*.png"))
        with Image.open(copied[0]) as staged:
            captured["mean"] = int(np.asarray(staged).mean())
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [{"filename": copied[0].name, "nr": 1, "anchoring": [1.0] * 9}],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from verso.engine.deepslice import DeepSliceOptions

    run_deepslice_suggestions(project, DeepSliceOptions(gamma=0.5))
    # 128^0.5 in normalized space → ~181 in uint8; allow a tolerance.
    assert 175 <= captured["mean"] <= 185


def _make_multichannel_project(tmp_path: Path, planes: np.ndarray, channels) -> Project:
    """Project whose single section has *planes* ``(H, W, C)`` as its working copy.

    Written as an OME-TIFF so ``ensure_working_copy`` loads it verbatim.
    """
    from verso.engine.io.image_io import _save_ome_tiff

    tmp_path.mkdir(parents=True, exist_ok=True)
    original = tmp_path / "s001.tif"
    thumb = tmp_path / "s001-thumb.ome.tif"
    _save_ome_tiff(planes, thumb, channel_names=[f"Ch {i}" for i in range(planes.shape[2])])
    _save_ome_tiff(planes, original, channel_names=[f"Ch {i}" for i in range(planes.shape[2])])
    return Project(
        name="multich",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[Section("s001", 1, str(original), str(thumb))],
        channels=list(channels),
    )


def _stage_and_capture(project: Project, monkeypatch) -> np.ndarray:
    """Run DeepSlice with a stubbed subprocess and return the staged PNG array."""
    captured: dict[str, object] = {}

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(Path(payload["folder"]).glob("*.png"))
        with Image.open(copied[0]) as staged:
            captured["mode"] = staged.mode
            captured["array"] = np.asarray(staged).copy()
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [{"filename": copied[0].name, "nr": 1, "anchoring": [1.0] * 9}],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_deepslice_suggestions(project, DeepSliceOptions(gamma=1.0))
    return captured["mode"], captured["array"]


def _three_planes() -> np.ndarray:
    """(20, 20, 3) uint8 with a distinct constant value per plane."""
    planes = np.zeros((20, 20, 3), dtype=np.uint8)
    planes[:, :, 0] = 40
    planes[:, :, 1] = 90
    planes[:, :, 2] = 200
    return planes


def test_staged_png_is_grayscale_max_of_visible_planes(tmp_path: Path, monkeypatch):
    """Multichannel sections stage as mode-L max-projection, not an RGB composite."""
    channels = [ChannelSpec(name=f"Ch {i}") for i in range(3)]
    project = _make_multichannel_project(tmp_path, _three_planes(), channels)

    mode, arr = _stage_and_capture(project, monkeypatch)

    assert mode == "L"
    assert arr.ndim == 2
    assert np.all(arr == 200)  # max(40, 90, 200)


def test_staged_png_ignores_channel_color(tmp_path: Path, monkeypatch):
    """Recolouring a channel must not change a single byte DeepSlice sees.

    Colour used to act as a hidden per-channel weight via ``rgb2gray``.
    """
    planes = _three_planes()
    white = [ChannelSpec(name=f"Ch {i}") for i in range(3)]
    tinted = [
        ChannelSpec(name="Ch 0", color=(255, 0, 0)),
        ChannelSpec(name="Ch 1", color=(0, 255, 0)),
        ChannelSpec(name="Ch 2", color=(0, 0, 255)),
    ]

    _, baseline = _stage_and_capture(
        _make_multichannel_project(tmp_path, planes, white), monkeypatch
    )
    _, recoloured = _stage_and_capture(
        _make_multichannel_project(tmp_path / "b", planes, tinted), monkeypatch
    )

    assert np.array_equal(baseline, recoloured)


def test_staged_png_ignores_channel_scale_and_gamma(tmp_path: Path, monkeypatch):
    """Brightness/gamma sliders are display-only and must not move the prediction."""
    planes = _three_planes()
    plain = [ChannelSpec(name=f"Ch {i}") for i in range(3)]
    adjusted = [ChannelSpec(name=f"Ch {i}", scale=0.2, gamma=2.5) for i in range(3)]

    _, baseline = _stage_and_capture(
        _make_multichannel_project(tmp_path, planes, plain), monkeypatch
    )
    _, tweaked = _stage_and_capture(
        _make_multichannel_project(tmp_path / "b", planes, adjusted), monkeypatch
    )

    assert np.array_equal(baseline, tweaked)


def test_hiding_a_channel_changes_the_staged_png(tmp_path: Path, monkeypatch):
    """Visibility is the one display control that is *meant* to be an input."""
    planes = _three_planes()
    all_visible = [ChannelSpec(name=f"Ch {i}") for i in range(3)]
    brightest_hidden = [
        ChannelSpec(name="Ch 0"),
        ChannelSpec(name="Ch 1"),
        ChannelSpec(name="Ch 2", visible=False),
    ]

    _, everything = _stage_and_capture(
        _make_multichannel_project(tmp_path, planes, all_visible), monkeypatch
    )
    _, reduced = _stage_and_capture(
        _make_multichannel_project(tmp_path / "b", planes, brightest_hidden), monkeypatch
    )

    assert np.all(everything == 200)
    assert np.all(reduced == 90)  # max(40, 90), the 200 plane excluded


def test_all_channels_hidden_falls_back_to_every_plane(tmp_path: Path, monkeypatch):
    """An all-hidden project used to stage pure black PNGs and predict nonsense."""
    channels = [ChannelSpec(name=f"Ch {i}", visible=False) for i in range(3)]
    project = _make_multichannel_project(tmp_path, _three_planes(), channels)

    mode, arr = _stage_and_capture(project, monkeypatch)

    assert mode == "L"
    assert np.all(arr == 200)
    assert arr.max() > 0


def _png_source_project(tmp_path: Path, channels) -> Project:
    """Single-section project whose original is a flat ``(40, 90, 200)`` PNG.

    The flat colour survives ``_stretch_per_channel`` untouched (a zero-range
    uint8 plane is returned verbatim), so each gun arrives at DeepSlice staging
    as its literal value.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    img_path = tmp_path / "s001.png"
    Image.new("RGB", (50, 40), color=(40, 90, 200)).save(img_path)
    return Project(
        name="photo",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[Section("s001", 1, str(img_path), str(img_path))],
        channels=list(channels),
    )


def test_png_source_is_greyscaled_like_any_other_section(tmp_path: Path, monkeypatch):
    """A JPEG/PNG import takes the same staging path as an OME-TIFF one.

    Its planes were a photograph's R/G/B once, but ``ensure_working_copy``
    percentile-stretches each of them *independently* at import, so there is no
    photograph left to hand to DeepSlice's own luminance conversion.
    """
    project = _png_source_project(tmp_path, [ChannelSpec(name=f"Ch {i}") for i in range(3)])

    mode, arr = _stage_and_capture(project, monkeypatch)

    assert mode == "L"
    assert arr.shape == (299, 299)
    assert np.all(arr == 200)  # max(40, 90, 200)


def test_hiding_a_channel_works_for_a_png_source(tmp_path: Path, monkeypatch):
    """Visibility used to be inert for true-colour imports.

    That was not a corner case: the QUINT import path produces ``-thumb.png``
    working images, so the dialog's "Using visible channels…" hint was false for
    every QuickNII/VisuAlign project.
    """
    channels = [
        ChannelSpec(name="Ch 0"),
        ChannelSpec(name="Ch 1"),
        ChannelSpec(name="Ch 2", visible=False),
    ]

    _, arr = _stage_and_capture(_png_source_project(tmp_path, channels), monkeypatch)

    assert np.all(arr == 90)  # max(40, 90), the 200 gun excluded


def test_run_deepslice_stages_working_resolution_png_with_display_flips(
    tmp_path: Path,
    monkeypatch,
):
    original = tmp_path / "s001.png"
    image = Image.new("RGB", (500, 400), color=(0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(original)
    project = Project(
        name="deep",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                "s001",
                1,
                str(original),
                str(tmp_path / "missing-thumb.ome.tif"),
                preprocessing=Preprocessing(flip_horizontal=True, flip_vertical=True),
            )
        ],
    )

    def fake_run(args, check, capture_output, text, **kwargs):
        payload = json.loads(args[3])
        copied = sorted(Path(payload["folder"]).glob("*.png"))
        assert len(copied) == 1
        with Image.open(copied[0]) as staged:
            # Staged at DeepSlice's own network input size so its loader's
            # nearest-neighbour resize becomes a no-op.
            assert staged.size == (299, 299)
            assert staged.mode == "L"
            staged_arr = np.asarray(staged)
        assert staged_arr[-1, -1] > staged_arr[0, 0]
        output = Path(payload["output_base"]).with_suffix(".json")
        output.write_text(
            json.dumps(
                {
                    "name": "out",
                    "target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                    "slices": [{"filename": copied[0].name, "nr": 1, "anchoring": [1.0] * 9}],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_deepslice_suggestions(project)


def test_reset_in_progress_to_default_proposals_clears_deepslice_metadata(tmp_path: Path):
    project = _make_project(tmp_path)
    for section in project.sections:
        section.alignment.current_anchoring = [1.0] * 9
        section.alignment.status = AlignmentStatus.IN_PROGRESS
        section.alignment.source = "deepslice"
        section.warp.control_points.append(ControlPoint(0.1, 0.2, 0.3, 0.4))

    changed = reset_in_progress_to_default_proposals(
        project.sections,
        atlas_shape=(528, 320, 456),
    )

    assert changed == 2
    for section in project.sections:
        assert section.alignment.status == AlignmentStatus.IN_PROGRESS
        assert section.alignment.source == "series_interpolation"
        assert section.warp.control_points == []


def test_reset_default_proposals_interpolates_flipped_keyframe_in_display_space(
    tmp_path: Path,
):
    atlas_shape = (528, 320, 456)
    image_paths = []
    for i in range(3):
        path = tmp_path / f"s{i + 1:03d}.png"
        Image.new("RGB", (1000, 800)).save(path)
        image_paths.append(path)

    angle = math.radians(18.0)
    unpacked_left = [
        228.0,
        500.0,
        160.0,
        math.cos(angle),
        math.sin(angle),
        0.0,
        0.0,
        0.0,
        1.0,
        0.456,
        0.4,
    ]
    unpacked_right = list(unpacked_left)
    unpacked_right[1] = 100.0
    left = pack_series_anchoring(unpacked_left, 1000, 800)
    # Section 3 is flipped to restore coherent orientation. After flipping,
    # the user aligns the atlas in display space — u points the same anatomical
    # direction as the unflipped keyframe. Both keyframes have the same u vector
    # in display space, so interpolation propagates the angle correctly.
    right = pack_series_anchoring(unpacked_right, 1000, 800)

    project = Project(
        name="deep",
        atlas=AtlasRef(name="allen_mouse_25um"),
        sections=[
            Section(
                "s001",
                1,
                str(image_paths[0]),
                str(image_paths[0]),
                resolution_thumbnail_wh=(1000, 800),
            ),
            Section(
                "s002",
                2,
                str(image_paths[1]),
                str(image_paths[1]),
                resolution_thumbnail_wh=(1000, 800),
            ),
            Section(
                "s003",
                3,
                str(image_paths[2]),
                str(image_paths[2]),
                preprocessing=Preprocessing(flip_horizontal=True),
                resolution_thumbnail_wh=(1000, 800),
            ),
        ],
    )
    project.sections[0].alignment.current_anchoring = left
    project.sections[0].alignment.stored_anchoring = list(left)
    project.sections[0].alignment.status = AlignmentStatus.COMPLETE
    project.sections[2].alignment.current_anchoring = right
    project.sections[2].alignment.stored_anchoring = list(right)
    project.sections[2].alignment.status = AlignmentStatus.COMPLETE

    changed = reset_in_progress_to_default_proposals(
        project.sections,
        atlas_shape=atlas_shape,
    )

    assert changed == 1
    middle = unpack_series_anchoring(project.sections[1].alignment.current_anchoring, 1000, 800)
    np.testing.assert_allclose(middle[4], math.sin(angle), atol=1e-9)
    np.testing.assert_allclose(middle[1], 300.0, atol=1e-9)


def test_reset_to_default_can_clear_complete_alignments(tmp_path: Path):
    project = _make_project(tmp_path)
    for section in project.sections:
        section.alignment.current_anchoring = [1.0] * 9
        section.alignment.status = AlignmentStatus.COMPLETE
        section.alignment.source = "manual"
        section.warp.control_points.append(ControlPoint(0.1, 0.2, 0.3, 0.4))

    changed = reset_in_progress_to_default_proposals(
        project.sections,
        atlas_shape=(528, 320, 456),
        include_complete=True,
    )

    assert changed == 2
    for section in project.sections:
        assert section.alignment.status == AlignmentStatus.IN_PROGRESS
        assert section.alignment.source == "series_interpolation"
        assert section.alignment.current_anchoring != [1.0] * 9
        assert section.warp.control_points == []
