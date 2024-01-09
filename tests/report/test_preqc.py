import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.preqc import PreQCStageReport


@pytest.fixture()
def preqc_summary_input(pixelator_workdir, preqc_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


def test_preqc_stage_model(preqc_stage_report_pbmcs):
    report = PreQCStageReport.from_json(preqc_stage_report_pbmcs)

    assert report.total_read_count == 200000


def test_preqc_summary(preqc_summary_input):
    reporting = PixelatorReporting(preqc_summary_input)
    preqc_summary = reporting.preqc_summary()

    assert preqc_summary.loc["pbmcs_unstimulated", :].total_read_count == 200000
    assert preqc_summary.loc["uropod_control", :].total_read_count == 300000
