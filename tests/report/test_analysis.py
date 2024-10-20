"""Tests for PixelatorReporting related to the analysis stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.analysis import (
    AnalysisSampleReport,
    ColocalizationReport,
    PolarizationReport,
)


@pytest.fixture()
def analysis_summary_input(
    pixelator_workdir, analysis_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


expected = [
    AnalysisSampleReport(
        sample_id="pbmcs_unstimulated",
        polarization=PolarizationReport(),
        colocalization=ColocalizationReport(),
    ),
    AnalysisSampleReport(
        sample_id="uropod_control",
        polarization=PolarizationReport(),
        colocalization=ColocalizationReport(),
    ),
]


@pytest.mark.parametrize("sample_name,expected", [(r.sample_id, r) for r in expected])
def test_analysis_metrics_lookup(analysis_summary_input, sample_name, expected):
    reporting = PixelatorReporting(analysis_summary_input)
    r = reporting.analysis_metrics(sample_name)
    assert r == expected


def test_analysis_summary(analysis_summary_input, snapshot):
    reporting = PixelatorReporting(analysis_summary_input)
    result = reporting.analysis_summary()
    snapshot.assert_match(result.to_csv(), "analysis_summary.csv")
