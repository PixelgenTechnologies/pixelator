"""Tests for PixelatorReporting related to the graph stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir


@pytest.fixture()
def graph_summary_input(pixelator_workdir, graph_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_graph_metrics_lookup(graph_summary_input, sample_name, snapshot):
    reporting = PixelatorReporting(graph_summary_input)
    r = reporting.graph_metrics(sample_name)

    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_graph_metrics.json")


def test_graph_summary(graph_summary_input, snapshot):
    reporting = PixelatorReporting(graph_summary_input)
    result = reporting.graph_summary()
    snapshot.assert_match(result.to_csv(), "graph_summary.csv")
