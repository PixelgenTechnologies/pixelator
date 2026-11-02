"""End-to-end tests for the `aggregate` CLI command.

Copyright © 2026 Pixelgen Technologies AB.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.aggregate.report import AggregateSampleReport


def _write_edgelist(path: Path) -> None:
    pl.DataFrame(
        {
            "marker_1": ["CD3", "CD3", "CD8"],
            "marker_2": ["CD4", "CD4", "CD19"],
            "umi1": pl.Series([1, 1, 5], dtype=pl.UInt64),
            "umi2": pl.Series([2, 2, 6], dtype=pl.UInt64),
        }
    ).write_parquet(path)


def test_aggregate_cli_run():
    """Aggregate writes parquet, meta.json, and report.json under the stage dir."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_path = d / "sample.edgelist.parquet"
        _write_edgelist(input_path)

        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "aggregate",
            str(input_path),
            "--output",
            str(d),
        ]
        result = runner.invoke(cli.main_cli, args)

        assert result.exit_code == 0, result.output

        out_dir = d / "aggregate"
        assert (out_dir / "sample.parquet").is_file()
        assert (out_dir / "sample.meta.json").is_file()
        assert (out_dir / "sample.report.json").is_file()

        df = pl.read_parquet(out_dir / "sample.parquet")
        assert df.columns == ["marker_1", "marker_2", "umi1", "umi2", "read_count"]
        assert df.height == 2
        assert df["read_count"].sum() == 3

        report = AggregateSampleReport.from_json(out_dir / "sample.report.json")
        assert report.report_type == "aggregate"
        assert report.sample_id == "sample"
        assert report.input_reads == 3
        assert report.output_reads == 3
        assert report.output_molecules == 2

        with open(out_dir / "sample.meta.json") as fh:
            meta = json.load(fh)
        assert meta["cli"]["arguments"]["edgelist"] == [str(input_path.resolve())]


def test_aggregate_cli_multiple_inputs():
    """Multiple edgelist inputs merge into one output; meta lists all inputs."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_a = d / "sample.part_001.edgelist.parquet"
        input_b = d / "sample.part_002.edgelist.parquet"
        _write_edgelist(input_a)
        pl.DataFrame(
            {
                "marker_1": ["CD3", "CD3"],
                "marker_2": ["CD4", "CD8"],
                "umi1": pl.Series([1, 9], dtype=pl.UInt64),
                "umi2": pl.Series([2, 10], dtype=pl.UInt64),
            }
        ).write_parquet(input_b)

        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "aggregate",
            str(input_a),
            str(input_b),
            "--output",
            str(d),
        ]
        result = runner.invoke(cli.main_cli, args)

        assert result.exit_code == 0, result.output

        out_dir = d / "aggregate"
        assert (out_dir / "sample.parquet").is_file()
        assert (out_dir / "sample.meta.json").is_file()
        assert (out_dir / "sample.report.json").is_file()

        df = pl.read_parquet(out_dir / "sample.parquet")
        assert df.height == 3
        assert df["read_count"].sum() == 5

        with open(out_dir / "sample.meta.json") as fh:
            meta = json.load(fh)
        assert meta["cli"]["arguments"]["edgelist"] == [
            str(input_a.resolve()),
            str(input_b.resolve()),
        ]

        report = AggregateSampleReport.from_json(out_dir / "sample.report.json")
        assert report.input_reads == 5
        assert report.output_molecules == 3


def test_aggregate_cli_sample_name_override():
    """--sample-name overrides the output basename."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_path = d / "sample.edgelist.parquet"
        _write_edgelist(input_path)

        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "aggregate",
            str(input_path),
            "--output",
            str(d),
            "--sample-name",
            "custom",
        ]
        result = runner.invoke(cli.main_cli, args)

        assert result.exit_code == 0, result.output

        out_dir = d / "aggregate"
        assert (out_dir / "custom.parquet").is_file()
        assert (out_dir / "custom.meta.json").is_file()
        assert (out_dir / "custom.report.json").is_file()
        assert not (out_dir / "sample.parquet").exists()


def test_aggregate_cli_rejects_non_parquet():
    """Non-parquet input is rejected."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_path = d / "sample.txt"
        input_path.write_text("not a parquet\n")

        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "aggregate",
            str(input_path),
            "--output",
            str(d),
        ]
        result = runner.invoke(cli.main_cli, args)

        assert result.exit_code != 0


def test_aggregate_cli_reports_invalid_parquet_schema():
    """Invalid edgelist schemas produce a clear CLI error."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_path = d / "invalid.parquet"
        pl.DataFrame(
            {
                "marker_1": ["CD3"],
                "marker_2": ["CD4"],
                "umi1": pl.Series([1], dtype=pl.UInt64),
            }
        ).write_parquet(input_path)

        result = runner.invoke(
            cli.main_cli,
            [
                "single-cell-pna",
                "aggregate",
                str(input_path),
                "--output",
                str(d),
            ],
        )

        assert result.exit_code != 0
        assert "Invalid edgelist schema" in result.output
        assert "missing columns: umi2" in result.output
