"""Tests for image IO helpers."""

from verso.engine.io.image_io import parse_section_serial_number, thumbnail_filename


def test_parse_section_serial_number_from_mouse_name():
    assert parse_section_serial_number("MOUSE_0042_CODEs.tif", fallback=1) == 42


def test_parse_section_serial_number_falls_back_to_list_order():
    assert parse_section_serial_number("section_without_number.tif", fallback=7) == 7


def test_thumbnail_filename_uses_source_stem():
    assert thumbnail_filename("MOUSE_0042_CODEs.tif") == "MOUSE_0042_CODEs-thumb.png"
