"""Tests for image IO helpers."""

from pathlib import Path

import numpy as np

from verso.engine.io.image_io import (
    _save_ome_tiff,
    guess_slice_indices,
    load_image,
    probe_channels,
    thumbnail_filename,
    to_multichannel,
)


def test_guess_slice_indices_picks_most_discriminating_field():
    # Constant mouse id + constant channel; only the section field varies.
    paths = [
        "M1_slide03_section012_DAPI.tif",
        "M1_slide03_section045_DAPI.tif",
        "M1_slide03_section007_DAPI.tif",
    ]
    assert guess_slice_indices(paths) == [12, 45, 7]


def test_guess_slice_indices_preserves_gaps_and_duplicates():
    paths = [
        "img_001.tif",
        "img_002.tif",
        "img_018.tif",  # gap before this one
        "img_018.tif",  # duplicate index (broken slice)
    ]
    assert guess_slice_indices(paths) == [1, 2, 18, 18]


def test_guess_slice_indices_falls_back_to_sequential_when_no_numbers():
    paths = ["anterior.tif", "middle.tif", "posterior.tif"]
    # No numeric field → 1..N assigned by natural-sorted name order.
    assert guess_slice_indices(paths) == [1, 2, 3]


def test_guess_slice_indices_ignores_partial_number_field():
    # The numeric field is absent from the first file, so no token position has
    # full coverage → sequential fallback assigned by name order.
    paths = ["plain.tif", "plain_2.tif", "plain_3.tif"]
    assert guess_slice_indices(paths) == [1, 2, 3]


def test_guess_slice_indices_empty():
    assert guess_slice_indices([]) == []


def test_thumbnail_filename_is_ome_tiff():
    assert thumbnail_filename("MOUSE_0042_CODEs.tif") == "MOUSE_0042_CODEs-thumb.ome.tif"


# ---------------------------------------------------------------------------
# to_multichannel
# ---------------------------------------------------------------------------

def test_to_multichannel_2d_grayscale_becomes_single_plane():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = to_multichannel(gray)
    assert out.shape == (2, 2, 1)
    assert out.dtype == np.uint8


def test_to_multichannel_rgb_kept_as_three_channels():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # bright red
    out = to_multichannel(rgb)
    assert out.shape == (100, 100, 3)


def test_to_multichannel_rgba_drops_alpha():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    out = to_multichannel(rgba)
    assert out.shape == (100, 100, 3)


def test_to_multichannel_channels_first_is_transposed():
    # OME-TIFF layout: (C, H, W) with C small, H/W large
    chw = np.zeros((4, 200, 150), dtype=np.uint8)
    chw[0] = 200
    chw[1] = 100
    out = to_multichannel(chw)
    assert out.shape == (200, 150, 4)


# ---------------------------------------------------------------------------
# OME-TIFF roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_ome_tiff_preserves_channels(tmp_path: Path):
    arr = np.zeros((50, 60, 3), dtype=np.uint8)
    arr[..., 0] = 100
    arr[..., 1] = 150
    arr[..., 2] = 200
    path = tmp_path / "test-thumb.ome.tif"
    _save_ome_tiff(arr, path, channel_names=["DAPI", "GFP", "RFP"])

    loaded = to_multichannel(load_image(path))
    assert loaded.shape == (50, 60, 3)


def test_probe_channels_reads_ome_names(tmp_path: Path):
    arr = np.zeros((20, 30, 2), dtype=np.uint8)
    path = tmp_path / "with-names.ome.tif"
    _save_ome_tiff(arr, path, channel_names=["DAPI", "GFP"])

    names = probe_channels(path)
    assert names == ["DAPI", "GFP"]


def test_probe_channels_falls_back_to_generic_names(tmp_path: Path):
    # Plain TIFF without OME metadata.
    import tifffile
    arr = np.zeros((20, 30), dtype=np.uint8)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), arr)

    names = probe_channels(path)
    assert len(names) == 1
    assert names[0].startswith("Ch ")
