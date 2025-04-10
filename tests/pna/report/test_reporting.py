"""Tests for the PixelatorPNAReporting class not related to any specific pixelator stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import shutil
import uuid
from pathlib import Path

import pytest

from pixelator.mpx.report.common.reporting import _ordered_pixelator_commands
from pixelator.pna.report.common import (
    PixelatorPNAReporting,
    PixelatorPNAWorkdir,
    SingleCellPNAStage,
)


@pytest.fixture()
def pixelator_rundir(tmpdir_factory) -> Path:
    tmpdir = tmpdir_factory.mktemp(f"pixelator-{uuid.uuid4().hex}")
    return Path(tmpdir)


@pytest.fixture()
def pixelator_workdir(pixelator_rundir) -> PixelatorPNAWorkdir:
    return PixelatorPNAWorkdir(pixelator_rundir)


@pytest.fixture()
def all_stages_all_reports_and_meta(pixelator_workdir, full_run_dir):
    shutil.rmtree(pixelator_workdir.basedir, ignore_errors=True)
    res = shutil.copytree(full_run_dir, pixelator_workdir.basedir, dirs_exist_ok=True)
    return pixelator_workdir


@pytest.fixture()
def all_stages_all_reports_and_meta_independent(pixelator_workdir, full_run_dir):
    shutil.rmtree(pixelator_workdir.basedir, ignore_errors=True)
    res = shutil.copytree(full_run_dir, pixelator_workdir.basedir, dirs_exist_ok=True)
    shutil.rmtree(pixelator_workdir.basedir / "collapse", ignore_errors=True)
    shutil.move(
        pixelator_workdir.basedir / "collapse-independent",
        pixelator_workdir.basedir / "collapse",
    )
    return pixelator_workdir


def test_reporting_plain_dir_constructor(pixelator_workdir):
    # Construct from Path instance
    reporting = PixelatorPNAReporting(pixelator_workdir.basedir)
    assert isinstance(reporting.workdir, PixelatorPNAWorkdir)


def test_reporting_samples(all_stages_all_reports_and_meta):
    # Construct from PixelatorWorkdir instance
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    assert reporting.samples() == {
        "PNA055_Sample07_filtered_S7",
    }


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_amplicon_metrics_lookup(
    sample_name, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.amplicon_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_amplicon_metrics.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_demux_metrics_lookup(sample_name, all_stages_all_reports_and_meta, snapshot):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.demux_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_demux_metrics.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_collapse_metrics_lookup(
    sample_name, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.collapse_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_collapse_metrics.json")


# @pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
# def test_collapse_metrics_lookup_independent(
#    sample_name, all_stages_all_reports_and_meta_independent, snapshot
# ):
#    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta_independent)
#    r = reporting.collapse_metrics(sample_name)
#    assert r.report_type == "collapse-independent"
#    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_collapse_metrics.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_graph_metrics_lookup(sample_name, all_stages_all_reports_and_meta, snapshot):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.graph_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_graph_metrics.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_analysis_metrics_lookup(
    sample_name, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.analysis_metrics(sample_name)
    snapshot.assert_match(r.to_json(indent=4), f"{sample_name}_analysis_metrics.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_reporting_reads_flow(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot, sample_name
):
    reporting = PixelatorPNAReporting(pixelator_workdir)
    reads_flow = reporting.reads_flow(sample_name)

    snapshot.assert_match(
        reads_flow.to_json(indent=4), f"{sample_name}_reads_flow.json"
    )


def test_sorted_pixelator_commands():
    commands = _ordered_pixelator_commands()

    assert len(commands) >= 10

    pna_commands = {
        "pixelator single-cell-pna amplicon",
        "pixelator single-cell-pna demux",
        "pixelator single-cell-pna collapse",
        "pixelator single-cell-pna graph",
        "pixelator single-cell-pna analysis",
        "pixelator single-cell-pna layout",
        "pixelator single-cell-pna report",
        "pixelator single-cell-pna combine-collapse",
    }

    assert pna_commands.issubset(commands)


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_cli_invocation_info(sample_name, all_stages_all_reports_and_meta, snapshot):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.cli_invocation_info(sample_name)

    # Full cache key
    amplicon_cmd1 = r.get_stage("pixelator single-cell-pna amplicon")
    amplicon_cmd2 = r.get_stage(SingleCellPNAStage.AMPLICON)
    amplicon_cmd3 = r.get_stage("amplicon")

    assert amplicon_cmd1 == amplicon_cmd2 == amplicon_cmd3

    snapshot.assert_match(r.to_json(), f"{sample_name}_cli_invocation_info.json")


@pytest.mark.parametrize("sample_name", ["PNA055_Sample07_filtered_S7"])
def test_cli_invocation_info_get_option(sample_name, all_stages_all_reports_and_meta):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    r = reporting.cli_invocation_info(sample_name)

    # Stage name
    panel_opt = r.get_option("collapse", "--panel")
    assert panel_opt.name == "--panel"
    assert panel_opt.value == "proxiome-immuno-155"

    # Stage Enum
    panel_opt = r.get_option(SingleCellPNAStage.COLLAPSE, "--panel")
    assert panel_opt.name == "--panel"
    assert panel_opt.value == "proxiome-immuno-155"

    # Stage Enum
    panel_opt = r.get_option("pixelator single-cell-pna collapse", "--panel")
    assert panel_opt.name == "--panel"
    assert panel_opt.value == "proxiome-immuno-155"
