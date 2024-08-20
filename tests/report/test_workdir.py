"""Tests for the PixelatorWorkdir class.

Copyright © 2023 Pixelgen Technologies AB.
"""

import os
import shutil

import pytest

from pixelator.report import PixelatorWorkdir
from pixelator.report.common import WorkdirOutputNotFound


def test_workdir_samples(pixelator_workdir: PixelatorWorkdir, all_stages_all_reports):
    """Test the workdir samples method."""

    assert set(pixelator_workdir.samples(cache=False)) == {
        "pbmcs_unstimulated",
        "uropod_control",
    }

    # Check that the cache is used and that the workdir is not re-read
    shutil.rmtree(os.fspath(pixelator_workdir.basedir))
    assert set(pixelator_workdir.samples(cache=True)) == {
        "pbmcs_unstimulated",
        "uropod_control",
    }


def test_workdir_metadata(
    pixelator_workdir: PixelatorWorkdir, all_stages_all_reports_and_meta
):
    f = pixelator_workdir.metadata_files()

    # 9 stages with 2 samples each
    assert len(f) == 18


def test_workdir_filtered_dataset(
    pixelator_workdir, all_stages_all_reports_and_meta, filtered_dataset_pxl_workdir
):
    res = pixelator_workdir.filtered_dataset("uropod_control", cache=False)
    assert res.name == "uropod_control.annotate.dataset.pxl"

    with pytest.raises(WorkdirOutputNotFound):
        pixelator_workdir.filtered_dataset("blah", cache=False)


def test_workdir_raw_component_metrics(
    pixelator_workdir, all_stages_all_reports_and_meta, raw_component_metrics_workdir
):
    res = pixelator_workdir.raw_component_metrics("uropod_control", cache=False)
    assert res.name == "uropod_control.raw_components_metrics.csv.gz"

    with pytest.raises(WorkdirOutputNotFound):
        pixelator_workdir.raw_component_metrics("blah", cache=False)
