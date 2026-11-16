"""Tests for the exit codes of the top-level CLI group.

Copyright © 2026 Pixelgen Technologies AB.
"""

import click
import duckdb
import pytest
from click.testing import CliRunner

from pixelator.cli.common import EXIT_CODE_RESOURCE_EXHAUSTED, AliasedOrderedGroup


@pytest.mark.parametrize(
    "error",
    [duckdb.OutOfMemoryException("out of memory"), MemoryError("out of memory")],
)
def test_group_maps_out_of_memory_to_resource_exit_code(error):
    """Out-of-memory errors from any subcommand exit with the dedicated code."""

    @click.group(cls=AliasedOrderedGroup)
    def group():
        pass

    @group.command()
    def fail():
        raise error

    result = CliRunner().invoke(group, ["fail"])

    assert EXIT_CODE_RESOURCE_EXHAUSTED == 137
    assert result.exit_code == EXIT_CODE_RESOURCE_EXHAUSTED


def test_group_keeps_exit_code_1_for_other_errors():
    """Non out-of-memory failures are left untouched."""

    @click.group(cls=AliasedOrderedGroup)
    def group():
        pass

    @group.command()
    def fail():
        raise click.ClickException("bad input")

    @group.command()
    def crash():
        raise duckdb.InvalidInputException("corrupt file")

    runner = CliRunner()
    assert runner.invoke(group, ["fail"]).exit_code == 1
    # Unhandled exceptions are captured by CliRunner with exit code 1.
    assert runner.invoke(group, ["crash"]).exit_code == 1
