"""
Tests for base pixelator exports

Copyright (c) 2022 Pixelgen Technologies AB.
"""
# pylint: disable=redefined-outer-name
import pixelator as mpx
from pixelator.pixeldataset import PixelDataset


def test_read(setup_basic_pixel_dataset, tmp_path):
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))
    res = mpx.read(str(file_target))
    assert isinstance(res, PixelDataset)
