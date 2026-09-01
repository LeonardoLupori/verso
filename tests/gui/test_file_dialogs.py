"""Tests for :mod:`verso.gui.file_dialogs`.

The Qt static helpers are patched out, so these assert the wrapper's contract:
which directory a dialog starts in, what gets remembered afterwards, and that
the kinds stay independent of each other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verso.engine.settings import (
    DIR_ANNOTATION,
    DIR_EXPORT,
    DIR_IMAGES,
    DIR_IMPORT,
    DIR_PROJECT,
    get_last_dir,
    set_last_dir,
)
from verso.gui import file_dialogs


@pytest.fixture(autouse=True)
def _config_dir(tmp_path, monkeypatch):
    """Keep every test off the real per-user settings file."""
    monkeypatch.setenv("VERSO_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def spy(monkeypatch):
    """Patch all four Qt helpers; record the start dir, return a chosen path."""

    class Spy:
        def __init__(self):
            self.start_dirs: list[str] = []
            self.result = ""

        def set(self, result):
            self.result = result

    spy = Spy()

    def fake_open(parent, title, directory, file_filter):
        spy.start_dirs.append(directory)
        return spy.result, ""

    def fake_open_many(parent, title, directory, file_filter):
        spy.start_dirs.append(directory)
        return ([spy.result] if spy.result else []), ""

    def fake_save(parent, title, suggested, file_filter):
        spy.start_dirs.append(suggested)
        return spy.result, ""

    def fake_dir(parent, title, directory):
        spy.start_dirs.append(directory)
        return spy.result

    monkeypatch.setattr(file_dialogs.QFileDialog, "getOpenFileName", staticmethod(fake_open))
    monkeypatch.setattr(file_dialogs.QFileDialog, "getOpenFileNames", staticmethod(fake_open_many))
    monkeypatch.setattr(file_dialogs.QFileDialog, "getSaveFileName", staticmethod(fake_save))
    monkeypatch.setattr(file_dialogs.QFileDialog, "getExistingDirectory", staticmethod(fake_dir))
    return spy


# ----------------------------------------------------------------------
# start_dir
# ----------------------------------------------------------------------


def test_start_dir_falls_back_when_nothing_is_remembered():
    assert file_dialogs.start_dir(DIR_PROJECT) == ""
    assert file_dialogs.start_dir(DIR_PROJECT, fallback="/home/me") == "/home/me"


def test_start_dir_prefers_the_remembered_folder(folder):
    set_last_dir(folder, DIR_PROJECT)
    assert file_dialogs.start_dir(DIR_PROJECT, fallback="/home/me") == str(folder.resolve())


def test_start_dir_falls_back_when_the_folder_is_gone(folder):
    set_last_dir(folder, DIR_PROJECT)
    folder.rmdir()
    assert file_dialogs.start_dir(DIR_PROJECT, fallback="/home/me") == "/home/me"


# ----------------------------------------------------------------------
# open_file / open_files
# ----------------------------------------------------------------------


def test_open_file_starts_at_the_fallback_then_remembers(spy, folder):
    chosen = folder / "thing.json"
    spy.set(str(chosen))

    assert file_dialogs.open_file(None, "t", "*", kind=DIR_PROJECT) == str(chosen)
    assert spy.start_dirs == [""]
    assert get_last_dir(DIR_PROJECT) == folder.resolve()

    # A second call now opens where the first one ended.
    file_dialogs.open_file(None, "t", "*", kind=DIR_PROJECT)
    assert spy.start_dirs[-1] == str(folder.resolve())


def test_cancelling_remembers_nothing(spy):
    spy.set("")
    assert file_dialogs.open_file(None, "t", "*", kind=DIR_PROJECT) == ""
    assert get_last_dir(DIR_PROJECT) is None


def test_open_files_remembers_the_first_selection(spy, folder):
    spy.set(str(folder / "a.tif"))
    assert file_dialogs.open_files(None, "t", "*", kind=DIR_IMAGES) == [str(folder / "a.tif")]
    assert get_last_dir(DIR_IMAGES) == folder.resolve()


def test_open_files_cancelled_remembers_nothing(spy):
    spy.set("")
    assert file_dialogs.open_files(None, "t", "*", kind=DIR_IMAGES) == []
    assert get_last_dir(DIR_IMAGES) is None


# ----------------------------------------------------------------------
# save_file
# ----------------------------------------------------------------------


def test_save_file_anchors_a_bare_name_to_the_remembered_folder(spy, folder):
    set_last_dir(folder, DIR_EXPORT)
    spy.set(str(folder / "out.json"))

    file_dialogs.save_file(None, "t", "out.json", "*", kind=DIR_EXPORT)
    assert spy.start_dirs == [str(folder.resolve() / "out.json")]


def test_save_file_leaves_an_absolute_suggestion_alone(spy, folder, tmp_path):
    set_last_dir(folder, DIR_EXPORT)
    absolute = tmp_path / "elsewhere" / "project.json"
    spy.set(str(absolute))

    file_dialogs.save_file(None, "t", str(absolute), "*", kind=DIR_EXPORT)
    assert spy.start_dirs == [str(absolute)]


def test_save_file_with_no_memory_keeps_the_bare_name(spy, folder):
    spy.set(str(folder / "out.json"))
    file_dialogs.save_file(None, "t", "out.json", "*", kind=DIR_EXPORT)
    assert spy.start_dirs == ["out.json"]


def test_save_file_remembers_where_the_file_went(spy, folder):
    spy.set(str(folder / "out.json"))
    file_dialogs.save_file(None, "t", "out.json", "*", kind=DIR_EXPORT)
    assert get_last_dir(DIR_EXPORT) == folder.resolve()


# ----------------------------------------------------------------------
# choose_dir
# ----------------------------------------------------------------------


def test_choose_dir_prefers_current_over_memory(spy, folder, tmp_path):
    set_last_dir(folder, DIR_PROJECT)
    spy.set(str(folder))
    file_dialogs.choose_dir(None, "t", kind=DIR_PROJECT, fallback="/home/me", current=str(tmp_path))
    assert spy.start_dirs == [str(tmp_path)]


def test_choose_dir_uses_memory_then_fallback(spy, folder):
    spy.set(str(folder))
    file_dialogs.choose_dir(None, "t", kind=DIR_PROJECT, fallback="/home/me")
    assert spy.start_dirs == ["/home/me"]

    file_dialogs.choose_dir(None, "t", kind=DIR_PROJECT, fallback="/home/me")
    assert spy.start_dirs[-1] == str(folder.resolve())


def test_choose_dir_remembers_the_directory_itself(spy, folder):
    spy.set(str(folder))
    file_dialogs.choose_dir(None, "t", kind=DIR_IMAGES)
    assert get_last_dir(DIR_IMAGES) == folder.resolve()


# ----------------------------------------------------------------------
# Kinds are independent
# ----------------------------------------------------------------------


def test_each_kind_is_remembered_separately(spy, tmp_path):
    """Choosing an export destination must not move where Open Project starts."""
    folders = {}
    for kind in (DIR_PROJECT, DIR_IMAGES, DIR_EXPORT, DIR_IMPORT, DIR_ANNOTATION):
        target = tmp_path / kind
        target.mkdir()
        folders[kind] = target.resolve()
        spy.set(str(target / "file.ext"))
        file_dialogs.open_file(None, "t", "*", kind=kind)

    for kind, expected in folders.items():
        assert get_last_dir(kind) == expected


def test_one_kind_does_not_leak_into_another(spy, folder):
    spy.set(str(folder / "a.tif"))
    file_dialogs.open_file(None, "t", "*", kind=DIR_IMAGES)

    assert get_last_dir(DIR_IMAGES) == folder.resolve()
    assert get_last_dir(DIR_PROJECT) is None
    assert file_dialogs.start_dir(DIR_PROJECT) == ""


def test_a_chosen_path_that_is_a_directory_is_stored_as_is(spy, folder):
    spy.set(str(folder))
    file_dialogs.open_file(None, "t", "*", kind=DIR_IMPORT)
    assert get_last_dir(DIR_IMPORT) == folder.resolve()


def test_returned_path_is_untouched(spy, folder):
    """The wrapper must not rewrite what the user picked."""
    chosen = folder / "weird name (1).json"
    spy.set(str(chosen))
    assert file_dialogs.open_file(None, "t", "*", kind=DIR_PROJECT) == str(chosen)
    assert Path(file_dialogs.open_file(None, "t", "*", kind=DIR_PROJECT)) == chosen
