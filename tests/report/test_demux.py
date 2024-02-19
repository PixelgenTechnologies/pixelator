"""Copyright © 2023 Pixelgen Technologies AB."""

import dataclasses

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir


@pytest.fixture()
def demux_summary_input(pixelator_workdir, demux_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


@dataclasses.dataclass
class Test:
    sample_id: str


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_collapse_metrics_lookup(demux_summary_input, sample_name, snapshot):
    reporting = PixelatorReporting(demux_summary_input)
    r = reporting.demux_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_demux_metrics.json")


def test_demux_summary(demux_summary_input, snapshot):
    reporting = PixelatorReporting(demux_summary_input)
    result = reporting.demux_summary()
    snapshot.assert_match(result.to_json(indent=4), "demux_summary.json")
