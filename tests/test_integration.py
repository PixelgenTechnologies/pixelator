"""Integration tests for the pixelator CLI.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
# pylint: disable=redefined-outer-name

import pytest
from click.testing import CliRunner

from pixelator import cli

pytestmark = pytest.mark.integration_test


def test_console_script_loads():
    """Test that the console script loads."""
    runner = CliRunner()
    res = runner.invoke(cli.main_cli, ["--help"])
    assert res.exit_code == 0
    assert "Usage: pixelator [OPTIONS] COMMAND [ARGS]..." in res.stdout
    assert "single-cell" in res.stdout


def test_command_line_interface():
    """Test the CLI commands work by just invoking help."""
    runner = CliRunner()
    # main
    result = runner.invoke(cli.main_cli)
    assert result.exit_code == 0

    result = runner.invoke(cli.main_cli, ["--help"])
    assert result.exit_code == 0

    # Single cell commands

    # amplicon
    result = runner.invoke(cli.single_cell, ["amplicon", "--help"])
    assert result.exit_code == 0

    # preqc
    result = runner.invoke(cli.single_cell, ["preqc", "--help"])
    assert result.exit_code == 0

    # adapterqc
    result = runner.invoke(cli.single_cell, ["adapterqc", "--help"])
    assert result.exit_code == 0

    # demux
    result = runner.invoke(cli.single_cell, ["demux", "--help"])
    assert result.exit_code == 0

    # collapse
    result = runner.invoke(cli.single_cell, ["collapse", "--help"])
    assert result.exit_code == 0

    # graph
    result = runner.invoke(cli.single_cell, ["graph", "--help"])
    assert result.exit_code == 0

    # annotate
    result = runner.invoke(cli.single_cell, ["annotate", "--help"])
    assert result.exit_code == 0

    # analysis
    result = runner.invoke(cli.single_cell, ["analysis", "--help"])
    assert result.exit_code == 0

    # report
    result = runner.invoke(cli.single_cell, ["report", "--help"])
    assert result.exit_code == 0


def test_single_cell_list_designs():
    """Test that the list designs command works."""
    runner = CliRunner()
    result = runner.invoke(cli.main_cli, ["single-cell", "--list-designs"])
    assert result.exit_code == 0
    assert "D21" in result.stdout


def test_single_cell_list_panels():
    """Test that the list panels command works."""
    runner = CliRunner()
    result = runner.invoke(cli.main_cli, ["single-cell", "--list-panels"])
    assert result.exit_code == 0
    assert "human-sc-immunology-spatial-proteomics" in result.stdout
