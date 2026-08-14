"""Copyright © 2023 Pixelgen Technologies AB."""

import json
import os

import click
import pytest
from click.testing import CliRunner

from pixelator.common.utils import (
    flatten,
    get_available_cpu_count,
    write_parameters_file,
)


@pytest.mark.parametrize(
    "input,expected",
    (
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([1, 2, ["test"]], [1, 2, "test"]),
        ([1, 2, ("test", 3)], [1, 2, "test", 3]),
    ),
)
def test_flatten(input, expected):
    """Verify flatten.

    Args:
        input: input.
        expected: expected.
    """
    assert list(flatten(input)) == expected


def test_write_parameters_file_includes_multi_file_arguments(tmp_path):
    """Multi-file Click arguments are listed under cli.arguments."""
    input_a = tmp_path / "a.parquet"
    input_b = tmp_path / "b.parquet"
    input_a.write_text("a")
    input_b.write_text("b")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    meta_file = tmp_path / "meta.json"

    @click.command()
    @click.argument(
        "input_files",
        nargs=-1,
        required=True,
        type=click.Path(exists=True),
    )
    @click.option("--output", required=True, type=click.Path())
    @click.pass_context
    def fake_command(ctx, input_files, output):
        write_parameters_file(ctx, meta_file)

    runner = CliRunner()
    result = runner.invoke(
        fake_command,
        [str(input_a), str(input_b), "--output", str(output_dir)],
    )
    assert result.exit_code == 0, result.output

    data = json.loads(meta_file.read_text())
    assert data["cli"]["options"]["--output"] == str(output_dir.resolve())
    assert data["cli"]["arguments"]["input_files"] == [
        str(input_a.resolve()),
        str(input_b.resolve()),
    ]


def test_write_parameters_file_includes_single_file_argument(tmp_path):
    """Single-file Click arguments are listed under cli.arguments."""
    input_file = tmp_path / "input.pxl"
    input_file.write_text("x")
    meta_file = tmp_path / "meta.json"

    @click.command()
    @click.argument("pxl_file", nargs=1, required=True, type=click.Path(exists=True))
    @click.option("--flag", is_flag=True, default=False)
    @click.pass_context
    def fake_command(ctx, pxl_file, flag):
        write_parameters_file(ctx, meta_file)

    runner = CliRunner()
    result = runner.invoke(fake_command, [str(input_file), "--flag"])
    assert result.exit_code == 0, result.output

    data = json.loads(meta_file.read_text())
    assert data["cli"]["options"]["--flag"] is True
    assert data["cli"]["arguments"]["pxl_file"] == str(input_file.resolve())


def test_get_available_cpu_count_prefers_process_cpu_count(monkeypatch):
    """process_cpu_count (affinity-aware, Python 3.13+) is used when available."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert get_available_cpu_count() == 4


def test_get_available_cpu_count_falls_back_to_sched_getaffinity(monkeypatch):
    """sched_getaffinity is used when process_cpu_count is unavailable or None."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert get_available_cpu_count() == 3


def test_get_available_cpu_count_falls_back_to_cpu_count(monkeypatch):
    """os.cpu_count is used when no affinity-aware API is available (e.g. Windows)."""
    monkeypatch.setattr(os, "process_cpu_count", None, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert get_available_cpu_count() == 8


def test_get_available_cpu_count_is_always_at_least_one(monkeypatch):
    """The count never drops below 1, even when every source returns None."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert get_available_cpu_count() == 1
