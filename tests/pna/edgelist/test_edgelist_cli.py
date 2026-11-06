"""End-to-end tests for the native `edgelist` CLI command.

Copyright © 2026 Pixelgen Technologies AB.
"""

import gzip
import json
import random
import tempfile
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from pixelator import cli


@pytest.fixture
def random_fastq(tmp_path) -> tuple[Path, Path]:
    """Paired gzipped FASTQ files filled with random DNA sequences."""
    rng = random.Random(0)
    n_reads = 32
    read_length = 80
    quality = "I" * read_length
    r1 = tmp_path / "random_S1_R1_001.fastq.gz"
    r2 = tmp_path / "random_S1_R2_001.fastq.gz"
    for path in (r1, r2):
        with gzip.open(path, "wt") as fh:
            for i in range(n_reads):
                seq = "".join(rng.choice("ACGT") for _ in range(read_length))
                fh.write(f"@read_{i}\n{seq}\n+\n{quality}\n")
    return r1, r2


@pytest.fixture(scope="module")
def edgelist_cli_run(testdata_300k):
    """Run the edgelist CLI once on the 300k fixture; yield stage outputs."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "edgelist",
            "--read-forward",
            str(testdata_300k[0]),
            "--read-reverse",
            str(testdata_300k[1]),
            "--output",
            str(d),
            "--design",
            "proxiome-v1",
            "--panel",
            "proxiome-v1-immuno-155-v1.0",
        ]
        result = runner.invoke(cli.main_cli, args)
        assert result.exit_code == 0, result.output

        stage_dir = Path(d) / "edgelist"
        parquets = list(stage_dir.glob("*.parquet"))
        reports = list(stage_dir.glob("*.report.json"))
        assert len(parquets) == 1, parquets
        assert len(reports) == 1, reports

        payload = json.loads(reports[0].read_text())
        df = pl.read_parquet(parquets[0])
        yield stage_dir, payload, df


def test_edgelist_paired_run_writes_parquet(edgelist_cli_run):
    """Paired raw FASTQ reads produce a single edgelist parquet."""
    _, _, df = edgelist_cli_run
    assert df.columns == ["marker_1", "marker_2", "umi1", "umi2"]
    assert df.height > 0


def test_edgelist_output_names_follow_the_input_fastq(edgelist_cli_run):
    """Paired outputs keep the FASTQ basename with the R1/R2 suffix removed."""
    stage_dir, payload, _ = edgelist_cli_run
    stem = "PNA055_Sample07_300k_S7_001"

    assert (stage_dir / f"{stem}.parquet").is_file()
    assert (stage_dir / f"{stem}.meta.json").is_file()
    assert (stage_dir / f"{stem}.report.json").is_file()
    assert payload["sample_id"] == stem


def test_edgelist_paired_run_writes_a_sample_report(edgelist_cli_run):
    """The run writes `<sample>.report.json` beside the parquet.

    Asserted against the raw JSON rather than the pydantic model, because the
    file is what ExperimentSummary consumes.
    """
    _, payload, _ = edgelist_cli_run

    assert payload["product_id"] == "single-cell-pna"
    assert payload["report_type"] == "edgelist"
    assert payload["sample_id"]

    assert payload["output_reads"] > 0
    assert payload["input_reads"] >= payload["output_reads"]

    q30 = payload["q30_statistics"]
    assert set(q30) == {"total", "umi1", "pid1", "pid2", "umi2"}
    assert all(0.0 < value <= 1.0 for value in q30.values()), q30

    dropped_by = payload["dropped_by"]
    assert sum(dropped_by.values()) == payload["input_reads"] - payload["output_reads"]
    counts = list(dropped_by.values())
    assert counts == sorted(counts, reverse=True), dropped_by

    for side in ("1", "2"):
        per_marker = payload[f"reads_per_marker_{side}"]
        assert sum(per_marker.values()) == payload["output_reads"]
        counts = list(per_marker.values())
        assert counts == sorted(counts, reverse=True), per_marker


def test_edgelist_report_row_count_matches_the_parquet(edgelist_cli_run):
    """`output_reads` is the number of rows actually written."""
    _, payload, df = edgelist_cli_run
    assert payload["output_reads"] == df.height


def test_edgelist_requires_at_least_one_read():
    """Passing neither forward nor reverse read fails with a usage error."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "--cores",
            "2",
            "single-cell-pna",
            "edgelist",
            "--output",
            str(d),
            "--design",
            "proxiome-v1",
            "--panel",
            "proxiome-v1-immuno-155-v1.0",
        ]
        result = runner.invoke(cli.main_cli, args)

        assert result.exit_code != 0


def test_edgelist_fails_when_most_reads_are_discarded(random_fastq, tmp_path):
    """Random sequences do not match the panel, so the command aborts."""
    runner = CliRunner()
    r1, r2 = random_fastq

    args = [
        "--cores",
        "2",
        "single-cell-pna",
        "edgelist",
        "--read-forward",
        str(r1),
        "--read-reverse",
        str(r2),
        "--output",
        str(tmp_path),
        "--design",
        "proxiome-v1",
        "--panel",
        "proxiome-v1-immuno-155-v1.0",
    ]
    result = runner.invoke(cli.main_cli, args)

    assert result.exit_code != 0, result.output
    assert ">50% of reads were discarded" in result.output


def test_edgelist_saves_dropped_reads_even_when_the_run_aborts(random_fastq, tmp_path):
    """`--save-dropped-reads` keeps every discarded read, with its reason.

    Random reads all miss the panel, so the run aborts; the dropped reads are
    still on disk, which is when they are most useful for finding out why.
    """
    runner = CliRunner()
    r1, r2 = random_fastq

    args = [
        "--cores",
        "2",
        "single-cell-pna",
        "edgelist",
        "--read-forward",
        str(r1),
        "--read-reverse",
        str(r2),
        "--output",
        str(tmp_path / "out"),
        "--design",
        "proxiome-v1",
        "--panel",
        "proxiome-v1-immuno-155-v1.0",
        "--save-dropped-reads",
    ]
    result = runner.invoke(cli.main_cli, args)
    assert result.exit_code != 0, result.output

    stage_dir = tmp_path / "out" / "edgelist"
    report = json.loads((stage_dir / "random_S1_001.report.json").read_text())
    dropped = pl.read_parquet(stage_dir / "random_S1_001.dropped_reads.parquet")

    assert dropped.height == report["input_reads"] == 32
    assert set(dropped["read_name"]) == {f"read_{i}" for i in range(32)}
    written = dict(dropped["reason"].cast(pl.String).value_counts().iter_rows())
    assert written == {r: c for r, c in report["dropped_by"].items() if c}
