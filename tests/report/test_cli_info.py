"""Tests for the PixelatorReporting class and the CommandInfo class.

Copyright © 2023 Pixelgen Technologies AB.
"""

import shutil

import pytest

from pixelator.report import PixelatorReporting, SingleCellStage
from pixelator.report.common import WorkdirOutputNotFound
from pixelator.report.models import CommandInfo


def test_command_info_loading_from_json(
    pixelator_workdir, all_stages_all_reports_and_meta
):
    meta_files = pixelator_workdir.metadata_files()
    m = meta_files[0]

    for m in meta_files:
        data = CommandInfo.from_json(m)
        assert data.command.startswith("pixelator single-cell")


@pytest.fixture()
def uropod_cli_invocation_data(pixelator_workdir, all_stages_all_reports_and_meta):
    reporting = PixelatorReporting(pixelator_workdir)
    data = reporting.cli_invocation_info("uropod_control")
    return data


@pytest.fixture()
def uropod_cli_invocation_data_no_amplicon(
    pixelator_workdir, all_stages_all_reports_and_meta
):
    shutil.rmtree(pixelator_workdir.basedir / "amplicon")
    reporting = PixelatorReporting(pixelator_workdir)
    data = reporting.cli_invocation_info("uropod_control", cache=False)
    return data


def test_cli_info_init(uropod_cli_invocation_data):
    data = uropod_cli_invocation_data
    assert data.sample_id == "uropod_control"
    assert len(data) == 9


def test_cli_info_bad_sample(pixelator_workdir, all_stages_all_reports_and_meta):
    reporting = PixelatorReporting(pixelator_workdir)
    with pytest.raises(
        WorkdirOutputNotFound,
        match='No command line metadata found for sample: "doesnotexist"',
    ):
        reporting.cli_invocation_info("doesnotexist")


def test_cli_info_iter(uropod_cli_invocation_data):
    for idx, command in enumerate(uropod_cli_invocation_data):
        assert uropod_cli_invocation_data[idx] == command


def test_cli_info_missing_stage(uropod_cli_invocation_data_no_amplicon):
    data = uropod_cli_invocation_data_no_amplicon
    assert data.get_stage(SingleCellStage.AMPLICON) is None


@pytest.mark.parametrize(
    "stage_selector,expected_command",
    [
        (SingleCellStage.PREQC, "pixelator single-cell preqc"),
        ("preqc", "pixelator single-cell preqc"),
        (SingleCellStage.ADAPTERQC, "pixelator single-cell adapterqc"),
        ("adapterqc", "pixelator single-cell adapterqc"),
        (SingleCellStage.COLLAPSE, "pixelator single-cell collapse"),
        ("collapse", "pixelator single-cell collapse"),
        (SingleCellStage.DEMUX, "pixelator single-cell demux"),
        ("demux", "pixelator single-cell demux"),
        (SingleCellStage.GRAPH, "pixelator single-cell graph"),
        ("graph", "pixelator single-cell graph"),
        (SingleCellStage.ANNOTATE, "pixelator single-cell annotate"),
        ("annotate", "pixelator single-cell annotate"),
        (SingleCellStage.ANALYSIS, "pixelator single-cell analysis"),
        ("analysis", "pixelator single-cell analysis"),
        (SingleCellStage.LAYOUT, "pixelator single-cell layout"),
        ("layout", "pixelator single-cell layout"),
        (SingleCellStage.AMPLICON, "pixelator single-cell amplicon"),
        ("amplicon", "pixelator single-cell amplicon"),
    ],
)
def test_cli_info_get_stage(
    uropod_cli_invocation_data, stage_selector, expected_command
):
    data = uropod_cli_invocation_data
    cli_info = data.get_stage(stage_selector)
    assert cli_info.command == expected_command


def test_cli_info_get_option(uropod_cli_invocation_data):
    data = uropod_cli_invocation_data

    out_dir_option = data.get_option(SingleCellStage.COLLAPSE, "--output")
    assert out_dir_option.name == "--output"
    assert out_dir_option.value == "."


def test_cli_info_get_option_missing_stage(uropod_cli_invocation_data_no_amplicon):
    data = uropod_cli_invocation_data_no_amplicon

    with pytest.raises(
        KeyError,
        match="No commandline metadata found for stage: pixelator single-cell-mpx amplicon",
    ):
        data.get_option(SingleCellStage.AMPLICON, "--output")
