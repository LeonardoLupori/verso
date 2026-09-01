"""Tests for :mod:`verso.engine.settings`.

``VERSO_CONFIG_DIR`` is pointed at a tmp_path in every test, so nothing here
touches the real per-user settings file.
"""

from __future__ import annotations

import json

import pytest

from verso.engine.settings import (
    LAST_PROJECT_DIR,
    SETTINGS_FILENAME,
    default_config_dir,
    get_last_project_dir,
    load_settings,
    save_settings,
    set_last_project_dir,
    settings_path,
    update_setting,
)


@pytest.fixture(autouse=True)
def _config_dir(tmp_path, monkeypatch):
    """Redirect the settings file into a throwaway directory."""
    target = tmp_path / "config"
    monkeypatch.setenv("VERSO_CONFIG_DIR", str(target))
    return target


def test_config_dir_honours_the_override_and_is_created(_config_dir):
    assert default_config_dir() == _config_dir
    assert _config_dir.is_dir()
    assert settings_path() == _config_dir / SETTINGS_FILENAME


def test_missing_file_reads_as_empty():
    assert load_settings() == {}


def test_settings_round_trip():
    save_settings({"a": 1, "b": "two"})
    assert load_settings() == {"a": 1, "b": "two"}


def test_update_setting_preserves_other_keys():
    save_settings({"keep": "me"})
    update_setting("added", 5)
    assert load_settings() == {"keep": "me", "added": 5}


def test_corrupt_file_reads_as_empty_instead_of_raising():
    settings_path().write_text("{not json at all", encoding="utf-8")
    assert load_settings() == {}


def test_non_object_json_reads_as_empty():
    settings_path().write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings() == {}


def test_save_leaves_no_temporary_file_behind(_config_dir):
    save_settings({"a": 1})
    assert [p.name for p in _config_dir.iterdir()] == [SETTINGS_FILENAME]


def test_save_writes_readable_json(_config_dir):
    save_settings({"a": 1})
    assert json.loads(settings_path().read_text(encoding="utf-8")) == {"a": 1}


# ----------------------------------------------------------------------
# Last project folder
# ----------------------------------------------------------------------


def test_no_remembered_folder_by_default():
    assert get_last_project_dir() is None


def test_remembers_the_folder_holding_a_project_file(tmp_path):
    project = tmp_path / "experiment" / "project-verso.json"
    project.parent.mkdir()
    project.write_text("{}", encoding="utf-8")

    set_last_project_dir(project)
    assert get_last_project_dir() == project.parent.resolve()


def test_a_directory_may_be_passed_directly(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    set_last_project_dir(folder)
    assert get_last_project_dir() == folder.resolve()


def test_a_folder_that_has_gone_away_is_ignored(tmp_path):
    """An unplugged drive must not point the Open dialog at a dead path."""
    folder = tmp_path / "removable"
    folder.mkdir()
    set_last_project_dir(folder)
    assert get_last_project_dir() is not None

    folder.rmdir()
    assert get_last_project_dir() is None


def test_a_junk_value_is_ignored():
    for junk in (123, "", None, ["a"]):
        save_settings({LAST_PROJECT_DIR: junk})
        assert get_last_project_dir() is None


def test_remembering_a_folder_keeps_unrelated_settings():
    save_settings({"other": "value"})
    set_last_project_dir(default_config_dir())
    stored = load_settings()
    assert stored["other"] == "value"
    assert LAST_PROJECT_DIR in stored
