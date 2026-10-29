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
from pixelator.common.utils import parallel as common_parallel
from pixelator.common.utils.parallel import _read_cgroup_cpu_quota


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


@pytest.fixture
def no_cgroup_quota(monkeypatch):
    """Neutralize the cgroup CPU quota so only the affinity path is exercised."""
    monkeypatch.setattr(common_parallel, "_read_cgroup_cpu_quota", lambda: None)


def test_get_available_cpu_count_prefers_process_cpu_count(
    monkeypatch, no_cgroup_quota
):
    """process_cpu_count (affinity-aware, Python 3.13+) is used when available."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert get_available_cpu_count() == 4


def test_get_available_cpu_count_falls_back_to_sched_getaffinity(
    monkeypatch, no_cgroup_quota
):
    """sched_getaffinity is used when process_cpu_count is unavailable or None."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert get_available_cpu_count() == 3


def test_get_available_cpu_count_falls_back_to_cpu_count(monkeypatch, no_cgroup_quota):
    """os.cpu_count is used when no affinity-aware API is available (e.g. Windows)."""
    monkeypatch.setattr(os, "process_cpu_count", None, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert get_available_cpu_count() == 8


def test_get_available_cpu_count_is_always_at_least_one(monkeypatch, no_cgroup_quota):
    """The count never drops below 1, even when every source returns None."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert get_available_cpu_count() == 1


def _point_cgroup_paths(
    monkeypatch, tmp_path, *, v2=None, v1_quota=None, v1_period=None
):
    """Point the cgroup path constants at temp files, writing contents when given."""
    v2_path = tmp_path / "cpu.max"
    quota_path = tmp_path / "cpu.cfs_quota_us"
    period_path = tmp_path / "cpu.cfs_period_us"
    if v2 is not None:
        v2_path.write_text(v2)
    if v1_quota is not None:
        quota_path.write_text(v1_quota)
    if v1_period is not None:
        period_path.write_text(v1_period)
    monkeypatch.setattr(common_parallel, "_CGROUP_V2_CPU_MAX", str(v2_path))
    monkeypatch.setattr(common_parallel, "_CGROUP_V1_CFS_QUOTA", str(quota_path))
    monkeypatch.setattr(common_parallel, "_CGROUP_V1_CFS_PERIOD", str(period_path))


def test_read_cgroup_cpu_quota_cgroup_v2(monkeypatch, tmp_path):
    """A cgroup v2 quota of 2 CPUs is reported."""
    _point_cgroup_paths(monkeypatch, tmp_path, v2="200000 100000")
    assert _read_cgroup_cpu_quota() == 2


def test_read_cgroup_cpu_quota_cgroup_v2_unlimited(monkeypatch, tmp_path):
    """A cgroup v2 'max' quota means no limit."""
    _point_cgroup_paths(monkeypatch, tmp_path, v2="max 100000")
    assert _read_cgroup_cpu_quota() is None


def test_read_cgroup_cpu_quota_rounds_fractional_up(monkeypatch, tmp_path):
    """A fractional quota (1.5 CPUs) is rounded up to 2."""
    _point_cgroup_paths(monkeypatch, tmp_path, v2="150000 100000")
    assert _read_cgroup_cpu_quota() == 2


def test_read_cgroup_cpu_quota_cgroup_v1(monkeypatch, tmp_path):
    """The cgroup v1 quota/period files are used when the v2 file is absent."""
    _point_cgroup_paths(monkeypatch, tmp_path, v1_quota="200000", v1_period="100000")
    assert _read_cgroup_cpu_quota() == 2


def test_read_cgroup_cpu_quota_cgroup_v1_unlimited(monkeypatch, tmp_path):
    """A negative cgroup v1 quota means no limit."""
    _point_cgroup_paths(monkeypatch, tmp_path, v1_quota="-1", v1_period="100000")
    assert _read_cgroup_cpu_quota() is None


def test_read_cgroup_cpu_quota_malformed_is_ignored(monkeypatch, tmp_path):
    """Unexpected cgroup file contents are ignored rather than raising."""
    _point_cgroup_paths(monkeypatch, tmp_path, v2="not a number")
    assert _read_cgroup_cpu_quota() is None


def test_read_cgroup_cpu_quota_no_files(monkeypatch, tmp_path):
    """No quota is reported when no cgroup CPU files exist (e.g. on Windows)."""
    _point_cgroup_paths(monkeypatch, tmp_path)
    assert _read_cgroup_cpu_quota() is None


def test_get_available_cpu_count_quota_narrower_than_affinity(monkeypatch):
    """The cgroup quota wins when it is more restrictive than CPU affinity."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
    monkeypatch.setattr(common_parallel, "_read_cgroup_cpu_quota", lambda: 2)

    assert get_available_cpu_count() == 2


def test_get_available_cpu_count_affinity_narrower_than_quota(monkeypatch):
    """The CPU affinity (cpuset) wins when it is more restrictive than the quota."""
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2, raising=False)
    monkeypatch.setattr(common_parallel, "_read_cgroup_cpu_quota", lambda: 8)

    assert get_available_cpu_count() == 2
