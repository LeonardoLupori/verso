"""Tests for the scene-aware container readers.

The CZI vendor-library path needs a real sample file, so these tests focus on
the pure-Python core that VERSO owns: the dimension-labelled flatten
(:func:`reduce_to_hwc`), the extension registry, and the error surface. The CZI
reader is exercised end-to-end via a real file in the GUI smoke test (see the
plan's verification section).
"""

from __future__ import annotations

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


class TestRegistryAndDispatch:
    def test_container_extensions(self):
        assert CONTAINER_EXTENSIONS == (".czi",)

    def test_is_container_case_insensitive(self):
        assert is_container("x.CZI")
        assert not is_container("y.lif")
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
        bogus = tmp_path / "broken.czi"
        bogus.write_bytes(b"not a real czi")
        with pytest.raises(RuntimeError, match="Cannot read scene"):
            scene_readers.read_scene(bogus, 0)


def test_scene_info_fields():
    info = SceneInfo(2, "Scene 2", 100, 80)
    assert (info.scene_index, info.name, info.width, info.height) == (2, "Scene 2", 100, 80)
