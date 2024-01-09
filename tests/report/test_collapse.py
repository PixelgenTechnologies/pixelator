import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.collapse import CollapseStageReport


@pytest.fixture()
def collapse_summary_input(
    pixelator_workdir, collapse_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


def test_collapse_stage_model(collapse_stage_report_pbmcs):
    report = CollapseStageReport.from_json(collapse_stage_report_pbmcs)

    assert report.output_read_count == 7001
    assert report.edge_count == 3463


def test_collapse_summary(collapse_summary_input):
    reporting = PixelatorReporting(collapse_summary_input)
    collapse_summary = reporting.collapse_summary()

    assert collapse_summary.loc["pbmcs_unstimulated", :].output_read_count == 7001
    assert collapse_summary.loc["uropod_control", :].output_read_count == 9171
