"""Tests for the PixelatorReporting class not related to any specific pixelator stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.common.reporting import _ordered_pixelator_commands


def test_reporting_plain_dir_constructor(pixelator_workdir):
    # Construct from Path instance
    reporting = PixelatorReporting(pixelator_workdir.basedir)
    assert isinstance(reporting.workdir, PixelatorWorkdir)


def test_reporting_samples(pixelator_workdir, all_stages_all_reports_and_meta):
    # Construct from PixelatorWorkdir instance
    reporting = PixelatorReporting(pixelator_workdir)
    assert reporting.samples() == {
        "pbmcs_unstimulated",
        "uropod_control",
    }


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_reporting_reads_flow(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot, sample_name
):
    reporting = PixelatorReporting(pixelator_workdir)
    reads_flow = reporting.reads_flow(sample_name)

    snapshot.assert_match(
        reads_flow.to_json(indent=4), f"{sample_name}_reads_flow.json"
    )


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_reporting_molecules_flow(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot, sample_name
):
    reporting = PixelatorReporting(pixelator_workdir)
    molecules_flow = reporting.molecules_flow(sample_name)

    snapshot.assert_match(
        molecules_flow.to_json(indent=4), f"{sample_name}_molecules_flow.json"
    )


def test_reporting_reads_flow_summary(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorReporting(pixelator_workdir)
    reads_flow = reporting.reads_flow_summary()

    snapshot.assert_match(reads_flow.to_csv(), "reads_flow_summary.csv")


def test_reporting_molecules_flow_summary(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorReporting(pixelator_workdir)
    result = reporting.molecules_flow_summary()

    snapshot.assert_match(result.to_csv(), "molecules_flow_summary.csv")


def test_sorted_pixelator_commands():
    commands = _ordered_pixelator_commands()

    assert len(commands) >= 10
    assert commands[:10] == [
        "pixelator single-cell amplicon",
        "pixelator single-cell preqc",
        "pixelator single-cell adapterqc",
        "pixelator single-cell demux",
        "pixelator single-cell collapse",
        "pixelator single-cell graph",
        "pixelator single-cell annotate",
        "pixelator single-cell layout",
        "pixelator single-cell analysis",
        "pixelator single-cell report",
    ]
