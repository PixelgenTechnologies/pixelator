"""Tests for PixelatorReporting related to the adapterqc stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir


@pytest.fixture()
def adapterqc_summary_input(
    pixelator_workdir, adapterqc_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_adapterqc_metrics_lookup(adapterqc_summary_input, sample_name, snapshot):
    reporting = PixelatorReporting(adapterqc_summary_input)
    r = reporting.adapterqc_metrics(sample_name)

    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_adapterqc_metrics.json")


def test_adapterqc_summary(adapterqc_summary_input, snapshot):
    reporting = PixelatorReporting(adapterqc_summary_input)
    result = reporting.adapterqc_summary()

    snapshot.assert_match(result.to_csv(), "adapterqc_summary.csv")
