"""Copyright © 2023 Pixelgen Technologies AB."""

import pytest

from pixelator.report import PixelatorWorkdir
from pixelator.report.common import WorkdirOutputNotFound


def test_workdir_samples(pixelator_workdir: PixelatorWorkdir, all_stages_all_reports):
    """Test the workdir samples method."""

    assert set(pixelator_workdir.samples(cache=False)) == {
        "pbmcs_unstimulated",
        "uropod_control",
    }
    assert set(pixelator_workdir.samples(cache=True)) == {
        "pbmcs_unstimulated",
        "uropod_control",
    }


def test_workdir_metadata(
    pixelator_workdir: PixelatorWorkdir, all_stages_all_reports_and_meta
):
    f = pixelator_workdir.metadata_files()

    # 8 stages with 2 samples each
    assert len(f) == 16


def test_workdir_filtered_dataset(
    pixelator_workdir, all_stages_all_reports_and_meta, filtered_datasets
):
    res = pixelator_workdir.filtered_dataset("uropod_control_300k_S1_001", cache=False)
    assert res.name == "uropod_control_300k_S1_001.annotate.dataset.pxl"

    with pytest.raises(WorkdirOutputNotFound):
        pixelator_workdir.filtered_dataset("blah", cache=False)


def test_workdir_raw_component_metrics(
    pixelator_workdir, all_stages_all_reports_and_meta, raw_component_metrics
):
    res = pixelator_workdir.raw_component_metrics(
        "uropod_control_300k_S1_001", cache=False
    )
    assert res.name == "uropod_control_300k_S1_001.raw_components_metrics.csv.gz"

    with pytest.raises(WorkdirOutputNotFound):
        pixelator_workdir.raw_component_metrics("blah", cache=False)