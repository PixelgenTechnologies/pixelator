import pytest

from pixelator.report import PixelatorWorkdir


def test_workdir_samples(pixelator_workdir: PixelatorWorkdir, all_stages_all_reports):
    """Test the workdir samples method."""

    assert set(pixelator_workdir.samples()) == {"pbmcs_unstimulated", "uropod_control"}


def test_get_filtered_datasets(
    pixelator_workdir: PixelatorWorkdir, all_stages_all_reports
):
    """Test the workdir samples method."""

    assert set(pixelator_workdir.samples()) == {"pbmcs_unstimulated", "uropod_control"}
