"""Tests for PixelatorReporting related to the layout stage.

Copyright © 2024 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.layout import LayoutSampleReport


@pytest.fixture()
def layout_summary_input(
    pixelator_workdir, layout_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


expected = [
    LayoutSampleReport(
        sample_id="pbmcs_unstimulated",
    ),
    LayoutSampleReport(
        sample_id="uropod_control",
    ),
]


@pytest.mark.parametrize("sample_name,expected", [(r.sample_id, r) for r in expected])
def test_layout_metrics_lookup(layout_summary_input, sample_name, expected):
    reporting = PixelatorReporting(layout_summary_input)
    r = reporting.layout_metrics(sample_name)
    assert r == expected


def test_layout_summary(layout_summary_input, snapshot):
    reporting = PixelatorReporting(layout_summary_input)
    result = reporting.layout_summary()
    snapshot.assert_match(result.to_csv(), "layout_summary.csv")
