"""Integration tests for the pixelator CLI.

Copyright © 2022 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name

import pytest
from click.testing import CliRunner

from pixelator.cli.main import main_cli, single_cell_pna

pytestmark = pytest.mark.integration_test


def test_console_script_loads():
    """Test that the console script loads."""
    runner = CliRunner()
    res = runner.invoke(main_cli, ["--help"])
    assert res.exit_code == 0
    assert "Usage: pixelator [OPTIONS] COMMAND [ARGS]..." in res.stdout
    assert "single-cell-pna" in res.stdout


def test_command_line_interface():
    """Test the CLI commands work by just invoking help."""
    runner = CliRunner()
    # main
    result = runner.invoke(main_cli)
    # assert result.exit_code == 0

    result = runner.invoke(main_cli, ["--help"])
    assert result.exit_code == 0

    # Single cell PNA commands
    for command in [
        "amplicon",
        "demux",
        "collapse",
        "graph",
        "graph_legacy",
        "sample-calling",
        "denoise",
        "analysis",
        "layout",
        "combine-collapse",
    ]:
        result = runner.invoke(single_cell_pna, [command, "--help"])
        assert result.exit_code == 0, command


def test_single_cell_list_designs():
    """Test that the list designs command works."""
    runner = CliRunner()
    result = runner.invoke(main_cli, ["single-cell-pna", "--list-designs"])
    assert result.exit_code == 0
    assert "proxiome-v1" in result.stdout


def test_single_cell_list_panels():
    """Test that the list panels command works."""
    runner = CliRunner()
    result = runner.invoke(main_cli, ["single-cell-pna", "--list-panels"])
    assert result.exit_code == 0
    assert "proxiome-v1-immuno" in result.stdout
