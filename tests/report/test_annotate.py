import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.annotate import AnnotateStageReport


@pytest.fixture()
def annotate_summary_input(
    pixelator_workdir, annotate_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


def test_annotate_stage_model(annotate_stage_report_pbmcs):
    report = AnnotateStageReport.from_json(annotate_stage_report_pbmcs)

    assert report.edge_count == 3463
    assert report.marker_count == 71


def test_annotate_summary(annotate_summary_input):
    reporting = PixelatorReporting(annotate_summary_input)
    annotate_summary = reporting.annotate_summary()

    assert annotate_summary.loc["pbmcs_unstimulated", :].total_upia == 3453
    assert annotate_summary.loc["uropod_control", :].total_upia == 4519
