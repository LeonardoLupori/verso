"""The Open Project dialog starts in the folder the last project was opened from.

Drives ``ProjectController.open_project`` / ``open_project_path`` with the file
dialog and project loader patched out, so the test covers the wiring — which
directory is handed to the dialog, and when the folder is recorded — without a
real project on disk.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from verso.engine.settings import get_last_project_dir, set_last_project_dir
from verso.gui import file_dialogs
from verso.gui.controllers import project_controller as pc
from verso.gui.controllers.project_controller import ProjectController


@pytest.fixture(autouse=True)
def _config_dir(tmp_path, monkeypatch):
    """Keep every test off the real per-user settings file."""
    monkeypatch.setenv("VERSO_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture
def controller(monkeypatch):
    """A ProjectController with just enough of a MainWindow to run these paths."""
    state = SimpleNamespace(project=None, project_path=None, loaded=[])
    state.load_project = lambda project, path: state.loaded.append((project, path))
    window = SimpleNamespace(
        _state=state,
        confirm_discard_active_draft=lambda: True,
    )
    ctl = ProjectController.__new__(ProjectController)
    ctl._window = window
    ctl._state = state
    return ctl


@pytest.fixture
def dialog_calls(monkeypatch):
    """Record the directory handed to getOpenFileName; return no selection."""
    calls = []

    def fake_open(parent, title, directory, file_filter):
        calls.append(directory)
        return "", ""

    monkeypatch.setattr(file_dialogs.QFileDialog, "getOpenFileName", staticmethod(fake_open))
    return calls


def test_first_ever_open_uses_the_default_directory(controller, dialog_calls):
    """With nothing remembered, behave exactly as before: an empty start dir."""
    controller.open_project()
    assert dialog_calls == [""]


def test_open_starts_in_the_remembered_folder(controller, dialog_calls, tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    set_last_project_dir(folder)

    controller.open_project()
    assert dialog_calls == [str(folder.resolve())]


def test_open_falls_back_when_the_remembered_folder_is_gone(controller, dialog_calls, tmp_path):
    folder = tmp_path / "removable"
    folder.mkdir()
    set_last_project_dir(folder)
    folder.rmdir()

    controller.open_project()
    assert dialog_calls == [""]


def test_a_successful_load_records_the_folder(controller, monkeypatch, tmp_path):
    project_file = tmp_path / "experiment" / "project-verso.json"
    project_file.parent.mkdir()
    project_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pc.Project, "load", staticmethod(lambda path: object()))

    controller.open_project_path(project_file)

    assert len(controller._state.loaded) == 1
    assert get_last_project_dir() == project_file.parent.resolve()


def test_a_failed_load_records_nothing(controller, monkeypatch, tmp_path):
    """A broken project must not point the next Open dialog at its folder."""
    shown = []
    monkeypatch.setattr(
        pc.QMessageBox,
        "critical",
        classmethod(lambda cls, *args, **kwargs: shown.append(args)),
    )

    def boom(path):
        raise ValueError("bad project")

    monkeypatch.setattr(pc.Project, "load", staticmethod(boom))

    controller.open_project_path(tmp_path / "broken" / "project-verso.json")

    assert len(shown) == 1
    assert get_last_project_dir() is None
    assert controller._state.loaded == []


def test_reopening_updates_the_remembered_folder(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(pc.Project, "load", staticmethod(lambda path: object()))
    first = tmp_path / "one"
    second = tmp_path / "two"
    for folder in (first, second):
        folder.mkdir()

    controller.open_project_path(first / "project-verso.json")
    assert get_last_project_dir() == first.resolve()

    controller.open_project_path(second / "project-verso.json")
    assert get_last_project_dir() == second.resolve()


def test_the_chosen_file_is_opened(controller, monkeypatch, tmp_path):
    """The dialog's selection is what gets loaded."""
    chosen = tmp_path / "picked" / "project-verso.json"
    chosen.parent.mkdir()
    monkeypatch.setattr(
        file_dialogs.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(chosen), "")),
    )
    opened = []
    monkeypatch.setattr(
        ProjectController, "open_project_path", lambda self, path: opened.append(path)
    )

    controller.open_project()
    assert opened == [Path(chosen)]
