"""Tests for the PixelatorWorkdir class.

Copyright © 2023 Pixelgen Technologies AB.
"""

import os
import shutil

import pytest

from pixelator.pna.report.common.workdir import (
    PixelatorPNAWorkdir,
    WorkdirOutputNotFound,
)


def test_workdir_samples(
    pixelator_workdir: PixelatorPNAWorkdir, all_stages_all_reports_and_meta
):
    """Test the workdir samples method."""

    assert set(pixelator_workdir.samples(cache=False)) == {
        "PNA055_Sample07_filtered_S7",
    }

    # Check that the cache is used and that the workdir is not re-read
    shutil.rmtree(os.fspath(pixelator_workdir.basedir))
    assert set(pixelator_workdir.samples(cache=True)) == {
        "PNA055_Sample07_filtered_S7",
    }


def test_workdir_metadata(
    pixelator_workdir: PixelatorPNAWorkdir, all_stages_all_reports_and_meta
):
    f = pixelator_workdir.metadata_files()
    assert len(f) == 17


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_workdir_filtered_dataset(
    pixelator_workdir, all_stages_all_reports_and_meta, sample_id
):
    res = pixelator_workdir.filtered_dataset(sample_id, cache=False)
    assert res.name == f"{sample_id}.graph.pxl"

    with pytest.raises(WorkdirOutputNotFound):
        pixelator_workdir.filtered_dataset("blah", cache=False)
