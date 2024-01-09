import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.amplicon import AmpliconStageReport


@pytest.fixture()
def amplicon_summary_input(
    pixelator_workdir, amplicon_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


def test_amplicon_stage_model(amplicon_stage_report_pbmcs):
    report = AmpliconStageReport.from_json(amplicon_stage_report_pbmcs)

    assert report.fraction_q30 == 0.9606927317923855


def test_amplicon_summary(amplicon_summary_input):
    reporting = PixelatorReporting(amplicon_summary_input)
    amplicon_summary = reporting.amplicon_summary()

    assert (
        amplicon_summary.loc["pbmcs_unstimulated", :].fraction_q30 == 0.9606927317923855
    )
    assert amplicon_summary.loc["uropod_control", :].fraction_q30 == 0.9602150898560277
