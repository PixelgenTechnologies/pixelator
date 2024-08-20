"""Tests for PixelatorReporting related to the collapse stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir


@pytest.fixture()
def collapse_summary_input(
    pixelator_workdir, collapse_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_collapse_metrics_lookup(collapse_summary_input, sample_name, snapshot):
    reporting = PixelatorReporting(collapse_summary_input)
    r = reporting.collapse_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_collapse_metrics.json")


def test_collapse_summary(collapse_summary_input, snapshot):
    reporting = PixelatorReporting(collapse_summary_input)
    result = reporting.collapse_summary()

    snapshot.assert_match(result.to_csv(), "collapse_summary.json")
