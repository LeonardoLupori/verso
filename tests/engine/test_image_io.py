"""Tests for image IO helpers."""

from verso.engine.io.image_io import parse_section_serial_number


def test_parse_section_serial_number_from_mouse_name():
    assert parse_section_serial_number("MOUSE_0042_CODEs.tif", fallback=1) == 42


def test_parse_section_serial_number_falls_back_to_list_order():
    assert parse_section_serial_number("section_without_number.tif", fallback=7) == 7
