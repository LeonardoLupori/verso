"""Tests for engine/io/quint_import.py — building a project from QuickNII/VisuAlign."""

import json
from pathlib import Path

from PIL import Image

from verso.engine.io.quint_import import (
    build_quint_project,
    filenames_are_thumbnails,
    match_originals_by_similarity,
    match_registration_images,
)
from verso.engine.model.alignment import AlignmentStatus
from verso.engine.model.project import Project


def _png(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size).save(path)


def _visualign_json(target: str = "allen_mouse_25um") -> dict:
    """A 2-slice VisuAlign file: slice 1 has markers, slice 2 is affine-only."""
    return {
        "name": "imported",
        "target": target,
        "slices": [
            {
                "filename": "IMG_0001.png",
                "nr": 1,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
                "markers": [[500.0, 240.0, 520.0, 232.0], [100.0, 640.0, 50.0, 664.0]],
            },
            {
                "filename": "IMG_0002.png",
                "nr": 2,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0, 160.0, 250.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
            },
        ],
    }


# ---------------------------------------------------------------------------
# match_registration_images
# ---------------------------------------------------------------------------


def test_match_is_extension_tolerant_and_reports_unmatched(tmp_path: Path):
    json_path = tmp_path / "va.json"
    json_path.write_text(json.dumps(_visualign_json()))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    # JSON says IMG_0001.png; on disk it is a .tif. IMG_0002 is missing entirely.
    _png(imgs / "IMG_0001.tif", (1000, 800))

    matched, unmatched = match_registration_images(json_path, imgs)

    assert set(matched) == {0}
    assert matched[0].name == "IMG_0001.tif"
    assert unmatched == [(1, "IMG_0002.png")]


def test_filenames_are_thumbnails_detects_quint_convention():
    assert filenames_are_thumbnails(["thumbnails/IMG_0001-thumb.png", "thumbnails/IMG_2-thumb.png"])
    assert filenames_are_thumbnails(["IMG_0001-thumb.png"])  # -thumb stem, no folder
    assert not filenames_are_thumbnails(["IMG_0001.png", "IMG_0002.png"])  # bare QuickNII
    assert not filenames_are_thumbnails(["/abs/path/IMG_0001.png"])  # DeepSlice absolute
    assert not filenames_are_thumbnails([])


def test_match_originals_by_similarity_pairs_thumbnails_to_originals():
    """Thumbnail reference names map to their most similar original, one-to-one."""
    refs = ["thumbnails/AL1A_002-thumb.png", "thumbnails/AL1A_004-thumb.png"]
    # Deliberately shuffled and differently-typed originals with matching stems.
    candidates = [Path("scans/AL1A_004.tif"), Path("scans/AL1A_002.tif")]

    matched = match_originals_by_similarity(refs, candidates)

    assert matched[0].name == "AL1A_002.tif"  # -thumb stripped, stems align
    assert matched[1].name == "AL1A_004.tif"


def test_match_originals_by_similarity_is_unique_and_thresholded():
    refs = ["S1-thumb.png", "S2-thumb.png", "S3-thumb.png"]
    # Two clearly-related files and one unrelated: S3 stays unassigned, no reuse.
    candidates = [Path("S1.tif"), Path("S2.tif"), Path("totally_unrelated_xyz.tif")]

    matched = match_originals_by_similarity(refs, candidates)

    assert matched[0].name == "S1.tif"
    assert matched[1].name == "S2.tif"
    assert 2 not in matched  # no candidate above the similarity floor
    assert len({str(p) for p in matched.values()}) == len(matched)  # unique files


def test_match_originals_by_similarity_empty_inputs():
    assert match_originals_by_similarity([], [Path("a.tif")]) == {}
    assert match_originals_by_similarity(["a-thumb.png"], []) == {}


def test_match_searches_subfolders_and_strips_thumb_suffix(tmp_path: Path):
    data = {
        "name": "x",
        "target": "allen_mouse_25um",
        "slices": [
            {
                "filename": "thumbnails/IMG_0001-thumb.png",
                "nr": 1,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0] * 9,
            }
        ],
    }
    json_path = tmp_path / "va.json"
    json_path.write_text(json.dumps(data))
    root = tmp_path / "imgs"
    sub = root / "scans"
    sub.mkdir(parents=True)
    _png(sub / "IMG_0001.png", (1000, 800))

    matched, unmatched = match_registration_images(json_path, root)

    assert not unmatched
    assert matched[0].name == "IMG_0001.png"


# ---------------------------------------------------------------------------
# build_quint_project
# ---------------------------------------------------------------------------


def _matched(tmp_path: Path, size: tuple[int, int] = (1000, 800)) -> tuple[Path, dict[int, Path]]:
    json_path = tmp_path / "va.json"
    json_path.write_text(json.dumps(_visualign_json()))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _png(imgs / "IMG_0001.png", size)
    _png(imgs / "IMG_0002.png", size)
    matched, unmatched = match_registration_images(json_path, imgs)
    assert not unmatched
    return json_path, matched


def test_build_reuse_originals_populates_dims_and_scale(tmp_path: Path):
    json_path, matched = _matched(tmp_path)
    project = build_quint_project(json_path, tmp_path / "proj", registration_paths=matched)

    assert len(project.sections) == 2
    assert project.working_scale == 1.0  # 1000px <= THUMBNAIL_MAX_SIDE
    for section in project.sections:
        assert min(section.resolution_original_wh) > 0
        assert min(section.resolution_thumbnail_wh) > 0  # unblocks registration
        assert Path(section.original_path).name.startswith("IMG_")
        assert "thumbnails" in section.thumbnail_path

    # Slice nr=1 sorts first; its markers pass through unchanged (factor == 1).
    cps = project.sections[0].warp.control_points
    assert len(cps) == 2
    assert abs(cps[0].src_x - 500.0) < 1e-6
    assert abs(cps[0].src_y - 240.0) < 1e-6
    assert abs(cps[0].dst_x - 520.0) < 1e-6
    assert abs(cps[0].dst_y - 232.0) < 1e-6


def test_build_separate_originals_rescales_control_points(tmp_path: Path):
    json_path = tmp_path / "va.json"
    json_path.write_text(json.dumps(_visualign_json()))
    reg = tmp_path / "reg"
    reg.mkdir()
    _png(reg / "IMG_0001.png", (1000, 800))
    _png(reg / "IMG_0002.png", (1000, 800))
    orig = tmp_path / "orig"
    orig.mkdir()
    _png(orig / "IMG_0001.png", (4000, 3200))
    _png(orig / "IMG_0002.png", (4000, 3200))
    reg_m, _ = match_registration_images(json_path, reg)
    orig_m, _ = match_registration_images(json_path, orig)

    project = build_quint_project(
        json_path, tmp_path / "proj", registration_paths=reg_m, original_paths=orig_m
    )

    assert project.working_scale == 0.5  # 2000 / 4000
    section = project.sections[0]
    assert section.resolution_original_wh == (4000, 3200)
    assert section.resolution_thumbnail_wh == (2000, 1600)
    # Working dims / registration dims = 2000 / 1000 = 2 → control points doubled.
    cp = section.warp.control_points[0]
    assert abs(cp.src_x - 1000.0) < 1e-6
    assert abs(cp.src_y - 480.0) < 1e-6
    assert abs(cp.dst_x - 1040.0) < 1e-6
    assert abs(cp.dst_y - 464.0) < 1e-6


def test_build_resolves_cutlas_target(tmp_path: Path):
    """A real VisuAlign file names the atlas by its .cutlas bundle."""
    data = _visualign_json(target="ABA_Mouse_CCFv3_2017_25um.cutlas")
    json_path = tmp_path / "va.json"
    json_path.write_text(json.dumps(data))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _png(imgs / "IMG_0001.png", (1000, 800))
    _png(imgs / "IMG_0002.png", (1000, 800))
    matched, _ = match_registration_images(json_path, imgs)

    project = build_quint_project(json_path, tmp_path / "proj", registration_paths=matched)

    assert project.atlas.name == "allen_mouse_25um"
    first = project.sections[0].alignment
    assert first.status == AlignmentStatus.COMPLETE
    assert any(first.current_anchoring)  # convention conversion ran → non-zero plane


def test_build_sets_and_overrides_interpolation_axis(tmp_path: Path):
    """The imported project carries a slicing axis: inferred by default, overridable."""
    json_path, matched = _matched(tmp_path)

    inferred = build_quint_project(json_path, tmp_path / "p1", registration_paths=matched)
    assert inferred.interpolation_axis in ("AP", "ML", "DV")

    forced = build_quint_project(
        json_path, tmp_path / "p2", registration_paths=matched, interpolation_axis="ML"
    )
    assert forced.interpolation_axis == "ML"

    # An unknown override is ignored, keeping the inferred axis.
    ignored = build_quint_project(
        json_path, tmp_path / "p3", registration_paths=matched, interpolation_axis="nonsense"
    )
    assert ignored.interpolation_axis == inferred.interpolation_axis


def test_build_saves_and_reloads(tmp_path: Path):
    json_path, matched = _matched(tmp_path)
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    project = build_quint_project(json_path, proj_dir, registration_paths=matched)

    out = proj_dir / "p_verso.json"
    project.save(out)
    reloaded = Project.load(out)

    assert len(reloaded.sections) == 2
    section = reloaded.sections[0]
    assert min(section.resolution_thumbnail_wh) > 0
    assert len(section.warp.control_points) == 2

    saved = project.sections[0].alignment.stored_anchoring
    loaded = section.alignment.stored_anchoring
    assert saved is not None and loaded is not None
    for a, b in zip(saved, loaded, strict=True):
        assert abs(a - b) < 1e-6
