"""Tests for AnnotationController's create/edit/delete/save/load flow.

Uses lightweight fakes for the window/state/page/view so the controller logic is
exercised without constructing real widgets. Interactive dialogs (delete confirm,
save-without-path notice) are monkeypatched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QMessageBox

from verso.engine.io.annotation_io import load_annotations
from verso.engine.model.annotation import AnnotationPoint
from verso.gui.controllers.annotation_controller import AnnotationController


class _FakePage:
    def __init__(self) -> None:
        self.annotations: list = []
        self.active = -1
        self.dirty = False

    def set_annotations(self, annotations, active_index) -> None:
        self.annotations = annotations
        self.active = active_index

    def set_dirty(self, dirty) -> None:
        self.dirty = dirty


class _FakeView:
    def __init__(self) -> None:
        self.annotations: list = []
        self.active = -1

    def set_annotations(self, annotations, active_index) -> None:
        self.annotations = annotations
        self.active = active_index


def _make_controller(tmp_path: Path, section_name: str | None = None) -> AnnotationController:
    section = None
    if section_name is not None:
        section = SimpleNamespace(original_path=str(tmp_path / section_name))
    state = SimpleNamespace(
        project=object(),
        project_path=tmp_path / "project-verso.json",
        current_section=section,
        show_status=lambda _msg: None,
    )
    window = SimpleNamespace(
        _state=state,
        _props=SimpleNamespace(annotate=_FakePage()),
        _annotate=_FakeView(),
    )
    return AnnotationController(window)  # type: ignore[arg-type]


def test_new_annotation_marks_dirty_and_refreshes(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    assert ctrl.is_dirty()
    page = ctrl._window._props.annotate  # type: ignore[attr-defined]
    assert len(page.annotations) == 1
    assert page.active == 0
    assert page.dirty is True
    # View got the same list pushed for rendering.
    assert len(ctrl._window._annotate.annotations) == 1  # type: ignore[attr-defined]


def test_new_annotations_get_distinct_titles_and_colors(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    ctrl.new_annotation()
    titles = [a.title for a in ctrl._annotations]
    colors = [a.color for a in ctrl._annotations]
    assert len(set(titles)) == 2
    assert colors[0] != colors[1]


def test_edit_active_updates_model(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    ctrl.set_color((1, 2, 3))
    ctrl.set_opacity(0.25)
    ctrl.set_visibility(False)
    ctrl.rename_active("my cells")
    ann = ctrl._annotations[0]
    assert ann.color == (1, 2, 3)
    assert ann.opacity == 0.25
    assert ann.visible is False
    assert ann.title == "my cells"


def test_rename_dedupes_against_other_titles(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()  # "annotation"
    ctrl.new_annotation()  # "annotation 2", active
    ctrl.rename_active("annotation")
    assert ctrl._annotations[1].title != "annotation"


def test_delete_active_removes_and_prompts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    ctrl.new_annotation()
    ctrl.delete_active()
    assert len(ctrl._annotations) == 1


def test_set_active_does_not_dirty(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    ctrl.save()  # clears dirty (path exists)
    assert not ctrl.is_dirty()
    ctrl.new_annotation()
    ctrl.save()
    ctrl.set_active(0)
    assert not ctrl.is_dirty()


def test_save_and_reload_round_trip(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.new_annotation()
    ctrl.rename_active("cells")
    ctrl.set_color((10, 20, 30))
    assert ctrl.save() is True
    assert not ctrl.is_dirty()

    # On disk under annotations/, and re-loadable.
    on_disk = load_annotations(tmp_path)
    assert [a.title for a in on_disk] == ["cells"]
    assert on_disk[0].color == (10, 20, 30)

    # A fresh controller loads the same set.
    ctrl2 = _make_controller(tmp_path)
    ctrl2.load_for_project()
    assert [a.title for a in ctrl2._annotations] == ["cells"]
    assert not ctrl2.is_dirty()


def test_save_without_project_path_is_noop(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    ctrl = _make_controller(tmp_path)
    ctrl._state.project_path = None  # type: ignore[attr-defined]
    ctrl.new_annotation()
    assert ctrl.save() is False
    # Still dirty (nothing was written).
    assert ctrl.is_dirty()


def test_load_for_project_clears_when_no_annotations(tmp_path: Path):
    ctrl = _make_controller(tmp_path)
    ctrl.load_for_project()
    assert ctrl._annotations == []
    assert not ctrl.is_dirty()


# ---------------------------------------------------------------------------
# Point editing (add / lasso-remove / undo)
# ---------------------------------------------------------------------------

_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_add_point_appends_to_active_with_section_filename(tmp_path: Path):
    ctrl = _make_controller(tmp_path, "sec.tif")
    ctrl.new_annotation()
    ctrl.add_point(100.0, 200.0)
    pts = ctrl._annotations[0].points
    assert len(pts) == 1
    assert (pts[0].x, pts[0].y, pts[0].image) == (100.0, 200.0, "sec.tif")


def test_add_point_without_active_is_noop(tmp_path: Path):
    ctrl = _make_controller(tmp_path, "sec.tif")
    ctrl.add_point(1.0, 2.0)
    assert ctrl._annotations == []


def test_remove_in_polygon_removes_enclosed(tmp_path: Path):
    ctrl = _make_controller(tmp_path, "sec.tif")
    ctrl.new_annotation()
    ctrl.add_point(5.0, 5.0)  # inside the square
    ctrl.add_point(50.0, 50.0)  # outside
    ctrl.remove_in_polygon(_SQUARE)
    pts = ctrl._annotations[0].points
    assert [(p.x, p.y) for p in pts] == [(50.0, 50.0)]


def test_remove_only_affects_current_section(tmp_path: Path):
    ctrl = _make_controller(tmp_path, "sec.tif")
    ctrl.new_annotation()
    # A point on a *different* image, inside the polygon, must survive.
    ctrl._annotations[0].points.append(AnnotationPoint(5.0, 5.0, "other.tif"))
    ctrl.add_point(5.0, 5.0)  # on sec.tif, inside
    ctrl.remove_in_polygon(_SQUARE)
    assert [p.image for p in ctrl._annotations[0].points] == ["other.tif"]


def test_undo_restores_previous_points(tmp_path: Path):
    ctrl = _make_controller(tmp_path, "sec.tif")
    ctrl.new_annotation()
    ctrl.add_point(1.0, 1.0)
    ctrl.add_point(2.0, 2.0)
    ctrl.undo()
    assert [(p.x, p.y) for p in ctrl._annotations[0].points] == [(1.0, 1.0)]
    ctrl.undo()
    assert ctrl._annotations[0].points == []
