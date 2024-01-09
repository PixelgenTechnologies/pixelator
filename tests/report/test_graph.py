import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.graph import GraphStageReport


@pytest.fixture()
def graph_summary_input(pixelator_workdir, graph_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


def test_graph_stage_model(graph_stage_report_pbmcs):
    report = GraphStageReport.from_json(graph_stage_report_pbmcs)

    assert report.edge_count == 3463
    assert report.marker_count == 71


def test_graph_summary(graph_summary_input):
    reporting = PixelatorReporting(graph_summary_input)
    graph_summary = reporting.graph_summary()

    assert graph_summary.loc["pbmcs_unstimulated", :].total_upia == 3453
    assert graph_summary.loc["uropod_control", :].total_upia == 4519
