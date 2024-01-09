import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.adapterqc import AdapterQCStageReport


@pytest.fixture()
def adapterqc_summary_input(
    pixelator_workdir, adapterqc_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


def test_adapterqc_stage_model(adapterqc_stage_report_pbmcs):
    report = AdapterQCStageReport.from_json(adapterqc_stage_report_pbmcs)

    assert report.total_read_count == 199390


def test_preqc_summary(adapterqc_summary_input):
    reporting = PixelatorReporting(adapterqc_summary_input)
    preqc_summary = reporting.adapterqc_summary()

    assert preqc_summary.loc["pbmcs_unstimulated"].total_read_count == 199390
    assert preqc_summary.loc["uropod_control"].total_read_count == 298988
