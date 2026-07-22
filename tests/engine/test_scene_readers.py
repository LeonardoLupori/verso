"""Tests for the scene-aware container readers.

The CZI/LIF vendor-library paths need real sample files, so these tests focus on
the pure-Python core that VERSO owns: the dimension-labelled flatten
(:func:`reduce_to_hwc`), the LIF mosaic stitch, the extension registry, and the
error surface. The vendor readers are exercised end-to-end via a real file in
the GUI smoke test (see the plan's verification section).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verso.engine.io import scene_readers
from verso.engine.io.scene_readers import (
    CONTAINER_EXTENSIONS,
    SceneInfo,
    is_container,
    reduce_to_hwc,
)


class TestReduceToHwc:
    def test_channels_first_cyx(self):
        arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        out = reduce_to_hwc(arr, "CYX")
        assert out.shape == (4, 5, 3)
        # channel 1 plane preserved
        assert np.array_equal(out[:, :, 1], arr[1])

    def test_max_projects_z_and_takes_first_t(self):
        arr = np.random.randint(0, 100, size=(2, 6, 3, 8, 7)).astype(np.uint16)
        out = reduce_to_hwc(arr, "TZCYX")
        assert out.shape == (8, 7, 3)
        # MIP over Z at T=0 for one channel/pixel
        assert out[4, 5, 1] == arr[0, :, 1, 4, 5].max()

    def test_single_channel_zstack(self):
        arr = np.random.randint(0, 255, size=(4, 10, 12)).astype(np.uint8)
        out = reduce_to_hwc(arr, "ZYX")
        assert out.shape == (10, 12, 1)
        assert np.array_equal(out[:, :, 0], arr.max(axis=0))

    def test_rgb_samples_preserved(self):
        arr = np.random.randint(0, 255, size=(10, 12, 3)).astype(np.uint8)
        out = reduce_to_hwc(arr, "YXS")
        assert out.shape == (10, 12, 3)
        assert np.array_equal(out, arr)

    def test_plain_2d_gets_channel_axis(self):
        out = reduce_to_hwc(np.zeros((5, 6), np.uint8), "YX")
        assert out.shape == (5, 6, 1)

    def test_dtype_preserved(self):
        arr = np.ones((2, 4, 5), np.uint16) * 300
        assert reduce_to_hwc(arr, "CYX").dtype == np.uint16

    def test_axes_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            reduce_to_hwc(np.zeros((3, 4)), "CYX")

    def test_missing_spatial_axis_raises(self):
        with pytest.raises(ValueError):
            reduce_to_hwc(np.zeros((3, 4)), "CT")


def _fake_tilescan(field_y, field_x, *, swap_xy=False, flip_x=False, flip_y=False):
    tiles = np.zeros(
        len(field_y),
        dtype=[
            ("field_y", "i4"),
            ("field_x", "i4"),
            ("pos_z", "f8"),
            ("pos_y", "f8"),
            ("pos_x", "f8"),
        ],
    )
    tiles["field_y"] = field_y
    tiles["field_x"] = field_x

    class _TS:
        def __init__(self):
            self.tiles = tiles
            self.swap_xy = swap_xy
            self.flip_x = flip_x
            self.flip_y = flip_y

        def __len__(self):
            return len(tiles)

    return _TS()


class TestLifMosaicStitch:
    def test_physical_position_placement_painters_overlap(self):
        # Two 4-wide tiles, pixel size 1.0, tile1 shifted +3 -> 1px overlap at x=3.
        # Painter's algorithm (matching libCZI): the later tile (m=1) wins overlap.
        ts = _fake_tilescan([0, 0], [0, 1])  # field indices deliberately wrong/unused
        ts.tiles["pos_x"] = [0.0, 3.0]
        ts.tiles["pos_y"] = [0.0, 0.0]
        img = SimpleNamespace(tilescan=ts, coords={"X": np.arange(4, dtype=float)})
        arr = np.zeros((2, 1, 4, 4), np.uint8)  # MCYX
        arr[0, 0] = 20  # tile 0 is the *brighter* one, drawn first
        arr[1, 0] = 10  # tile 1 drawn last -> overwrites the overlap
        out, axes = scene_readers._lif_stitch_mosaic(arr, "MCYX", img)
        assert axes == "CYX"
        assert out.shape == (1, 4, 7)  # width = max(ox)+tile_w = 3+4
        assert out[0, 0, 0] == 20  # tile 0 only
        assert out[0, 0, 3] == 10  # overlap -> last tile wins (not max)
        assert out[0, 0, 6] == 10  # tile 1 only

    def test_anisotropic_pixel_size(self):
        # Different X/Y pixel size must be applied per axis.
        ts = _fake_tilescan([0, 0], [0, 0])
        ts.tiles["pos_x"] = [0.0, 10.0]  # 10 metres / 2 (px_x) = 5 px offset
        ts.tiles["pos_y"] = [0.0, 0.0]
        img = SimpleNamespace(
            tilescan=ts,
            coords={"X": np.arange(4) * 2.0, "Y": np.arange(4) * 1.0},
        )
        ox, oy, _w, _h = scene_readers._lif_tile_offsets(img, 4, 4)
        assert list(ox) == [0, 5]  # 10 / px_x(=2)
        assert list(oy) == [0, 0]

    def test_grid_fallback_when_no_pixel_size(self):
        tile_h, tile_w = 5, 6
        ts = _fake_tilescan([0, 0, 1, 1], [0, 1, 0, 1])
        img = SimpleNamespace(tilescan=ts)
        # dims MCYX: 4 tiles, 1 channel
        arr = np.zeros((4, 1, tile_h, tile_w), np.uint16)
        for m in range(4):
            arr[m, 0] = m + 1
        out, axes = scene_readers._lif_stitch_mosaic(arr, "MCYX", img)
        assert axes == "CYX"
        assert out.shape == (1, 2 * tile_h, 2 * tile_w)
        # tile placement: TL=1, TR=2, BL=3, BR=4
        assert out[0, 0, 0] == 1
        assert out[0, 0, tile_w] == 2
        assert out[0, tile_h, 0] == 3
        assert out[0, tile_h, tile_w] == 4

    def test_no_mosaic_axis_is_noop(self):
        arr = np.zeros((2, 4, 5), np.uint8)
        out, axes = scene_readers._lif_stitch_mosaic(arr, "CYX", SimpleNamespace(tilescan=None))
        assert axes == "CYX"
        assert out.shape == arr.shape

    def test_missing_geometry_falls_back_to_first_tile(self):
        arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)  # MYX
        out, axes = scene_readers._lif_stitch_mosaic(arr, "MYX", SimpleNamespace(tilescan=None))
        assert axes == "YX"
        assert np.array_equal(out, arr[0])


class TestRegistryAndDispatch:
    def test_container_extensions(self):
        assert CONTAINER_EXTENSIONS == (".czi", ".lif")

    def test_is_container_case_insensitive(self):
        assert is_container("x.CZI")
        assert is_container("y.lif")
        assert not is_container("z.tif")

    def test_reader_for_unknown_extension_raises(self):
        with pytest.raises(ValueError):
            scene_readers._reader_for("foo.tif")

    def test_enumerate_scenes_wraps_errors(self, tmp_path):
        bogus = tmp_path / "broken.czi"
        bogus.write_bytes(b"not a real czi")
        with pytest.raises(RuntimeError, match="Cannot read scenes"):
            scene_readers.enumerate_scenes(bogus)

    def test_read_scene_wraps_errors(self, tmp_path):
        bogus = tmp_path / "broken.lif"
        bogus.write_bytes(b"not a real lif")
        with pytest.raises(RuntimeError, match="Cannot read scene"):
            scene_readers.read_scene(bogus, 0)


def test_scene_info_fields():
    info = SceneInfo(2, "Scene 2", 100, 80)
    assert (info.scene_index, info.name, info.width, info.height) == (2, "Scene 2", 100, 80)
