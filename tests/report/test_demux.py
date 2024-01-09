import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.demux import DemuxStageReport


@pytest.fixture()
def demux_summary_input(pixelator_workdir, demux_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


def test_demux_stage_model(demux_stage_report_pbmcs):
    report = DemuxStageReport.from_json(demux_stage_report_pbmcs)

    assert report.input_read_count == 199390
    assert report.output_read_count == 189009
    assert report.per_antibody_read_counts["ACTB"] == 55


def test_demux_summary(demux_summary_input):
    reporting = PixelatorReporting(demux_summary_input)
    demux_summary = reporting.demux_summary()

    assert demux_summary.loc["pbmcs_unstimulated", :].input_read_count == 199390
    assert demux_summary.loc["uropod_control", :].input_read_count == 298988
