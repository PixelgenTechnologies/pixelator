"""
Tests for base pixelator exports

Copyright © 2022 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name
import pixelator.mpx as mpx
from pixelator.mpx.pixeldataset import PixelDataset


def test_read(setup_basic_pixel_dataset, tmp_path):
    """Verify read.

    Args:
        setup_basic_pixel_dataset: setup basic pixel dataset.
        tmp_path: tmp path.
    """
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))
    res = mpx.read(str(file_target))
    assert isinstance(res, PixelDataset)
