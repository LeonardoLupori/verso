"""Unit tests for add/remove section helpers in ``engine.sections``."""

from __future__ import annotations

from pathlib import Path

from verso.engine.model.alignment import Alignment, WarpState
from verso.engine.model.project import Preprocessing, Section
from verso.engine.sections import (
    make_added_sections,
    next_section_ids,
    removed_section_artifacts,
)


def _section(sid: str, slice_index: int, original: str, thumbs: Path) -> Section:
    return Section(
        id=sid,
        slice_index=slice_index,
        original_path=original,
        thumbnail_path=str(thumbs / f"{Path(original).stem}-thumb.ome.tif"),
        preprocessing=Preprocessing(),
        alignment=Alignment(),
        warp=WarpState(),
    )


# ---------------------------------------------------------------------------
# next_section_ids
# ---------------------------------------------------------------------------


def test_next_section_ids_continues_from_max():
    assert next_section_ids(["s001", "s002", "s003"], 2) == ["s004", "s005"]


def test_next_section_ids_avoids_collisions_with_gaps_and_oddballs():
    ids = next_section_ids(["s001", "s003", "x", "weird"], 2)
    assert ids == ["s004", "s005"]
    assert not set(ids) & {"s001", "s003"}


def test_next_section_ids_empty_project():
    assert next_section_ids([], 3) == ["s001", "s002", "s003"]


# ---------------------------------------------------------------------------
# make_added_sections
# ---------------------------------------------------------------------------


def test_make_added_sections_appends_after_series(tmp_path):
    thumbs = tmp_path / "thumbnails"
    existing = [
        _section("s001", 5, "a/img_01.tif", thumbs),
        _section("s002", 6, "a/img_02.tif", thumbs),
    ]
    new, skipped = make_added_sections(existing, ["b/img_10.tif", "b/img_09.tif"], thumbs)
    assert skipped == []
    # Provisional indices continue past the current max (6), in natural order.
    assert [s.slice_index for s in new] == [7, 8]
    assert [Path(s.original_path).stem for s in new] == ["img_09", "img_10"]
    # Fresh, non-colliding ids.
    assert [s.id for s in new] == ["s003", "s004"]
    assert all(s.thumbnail_path.startswith(str(thumbs)) for s in new)


def test_make_added_sections_skips_duplicate_original(tmp_path):
    thumbs = tmp_path / "thumbnails"
    existing = [_section("s001", 1, "a/img_01.tif", thumbs)]
    new, skipped = make_added_sections(existing, ["a/img_01.tif"], thumbs)
    assert new == []
    assert skipped == ["a/img_01.tif"]


def test_make_added_sections_skips_stem_collision(tmp_path):
    # Same stem in a different folder maps to the same thumbnail filename.
    thumbs = tmp_path / "thumbnails"
    existing = [_section("s001", 1, "a/img_01.tif", thumbs)]
    new, skipped = make_added_sections(existing, ["b/img_01.tif"], thumbs)
    assert new == []
    assert skipped == ["b/img_01.tif"]


def test_make_added_sections_skips_intra_batch_stem_collision(tmp_path):
    thumbs = tmp_path / "thumbnails"
    new, skipped = make_added_sections([], ["a/img_01.tif", "b/img_01.tif"], thumbs)
    assert len(new) == 1
    assert skipped == ["b/img_01.tif"]


# ---------------------------------------------------------------------------
# removed_section_artifacts
# ---------------------------------------------------------------------------


def test_removed_section_artifacts_lists_own_files(tmp_path):
    thumbs = tmp_path / "thumbnails"
    section = _section("s001", 1, "a/img_01.tif", thumbs)
    others = [_section("s002", 2, "a/img_02.tif", thumbs)]
    paths = removed_section_artifacts(section, others)
    stems = {p.name for p in paths}
    assert "img_01-thumb.ome.tif" in stems
    assert "img_01-slice-mask.png" in stems
    assert "img_01_lr.png" in stems
    # Nothing belonging to the surviving section.
    assert not any("img_02" in p.name for p in paths)


def test_removed_section_artifacts_excludes_shared_files(tmp_path):
    # A surviving section sharing the same stem must keep its files.
    thumbs = tmp_path / "thumbnails"
    section = _section("s001", 1, "a/img_01.tif", thumbs)
    survivor = _section("s002", 2, "b/img_01.tif", thumbs)  # same stem
    paths = removed_section_artifacts(section, [survivor])
    assert paths == []
