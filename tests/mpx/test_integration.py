"""Integration tests for the pixelator CLI.

Copyright © 2022 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name

import pytest
from click.testing import CliRunner

from pixelator.cli.main import main_cli, single_cell_mpx

pytestmark = pytest.mark.integration_test


def test_console_script_loads():
    """Test that the console script loads."""
    runner = CliRunner()
    res = runner.invoke(main_cli, ["--help"])
    assert res.exit_code == 0
    assert "Usage: pixelator [OPTIONS] COMMAND [ARGS]..." in res.stdout
    assert "single-cell" in res.stdout


def test_command_line_interface():
    """Test the CLI commands work by just invoking help."""
    runner = CliRunner()
    # main
    result = runner.invoke(main_cli)
    # assert result.exit_code == 0

    result = runner.invoke(main_cli, ["--help"])
    assert result.exit_code == 0

    # Single cell commands

    # amplicon
    result = runner.invoke(single_cell_mpx, ["amplicon", "--help"])
    assert result.exit_code == 0

    # preqc
    result = runner.invoke(single_cell_mpx, ["preqc", "--help"])
    assert result.exit_code == 0

    # adapterqc
    result = runner.invoke(single_cell_mpx, ["adapterqc", "--help"])
    assert result.exit_code == 0

    # demux
    result = runner.invoke(single_cell_mpx, ["demux", "--help"])
    assert result.exit_code == 0

    # collapse
    result = runner.invoke(single_cell_mpx, ["collapse", "--help"])
    assert result.exit_code == 0

    # graph
    result = runner.invoke(single_cell_mpx, ["graph", "--help"])
    assert result.exit_code == 0

    # annotate
    result = runner.invoke(single_cell_mpx, ["annotate", "--help"])
    assert result.exit_code == 0

    # analysis
    result = runner.invoke(single_cell_mpx, ["analysis", "--help"])
    assert result.exit_code == 0

    # report
    result = runner.invoke(single_cell_mpx, ["report", "--help"])
    assert result.exit_code == 0


def test_single_cell_list_designs():
    """Test that the list designs command works."""
    runner = CliRunner()
    result = runner.invoke(main_cli, ["single-cell-mpx", "--list-designs"])
    assert result.exit_code == 0
    assert "D21" in result.stdout


def test_single_cell_list_panels():
    """Test that the list panels command works."""
    runner = CliRunner()
    result = runner.invoke(main_cli, ["single-cell-mpx", "--list-panels"])
    assert result.exit_code == 0
    assert "human-sc-immunology-spatial-proteomics" in result.stdout
