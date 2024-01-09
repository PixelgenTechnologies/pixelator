import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.analysis import AnalysisStageReport


@pytest.fixture()
def analysis_summary_input(
    pixelator_workdir, analysis_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


def test_analysis_stage_model(analysis_stage_report_pbmcs):
    report = AnalysisStageReport.from_json(analysis_stage_report_pbmcs)
    assert report.antibody_control == ["mIgG2a", "mIgG2b", "mIgG1"]


def test_analysis_summary(analysis_summary_input):
    reporting = PixelatorReporting(analysis_summary_input)
    analysis_summary = reporting.analysis_summary()

    assert analysis_summary.loc["pbmcs_unstimulated", :].antibody_control == [
        "mIgG2a",
        "mIgG2b",
        "mIgG1",
    ]
    assert analysis_summary.loc["uropod_control", :].antibody_control == [
        "mIgG2a",
        "mIgG2b",
        "mIgG1",
    ]
