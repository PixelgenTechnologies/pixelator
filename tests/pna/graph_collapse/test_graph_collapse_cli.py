"""Copyright © 2026 Pixelgen Technologies AB."""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from pixelator import cli


def test_graph_collapse_cli_run(pna_pxl_file):
    pxl_path = pna_pxl_file

    with duckdb.connect(str(pxl_path), read_only=True) as con:
        input_umis = con.execute(
            "SELECT COUNT(DISTINCT umi) FROM ("
            "SELECT umi1 AS umi FROM edgelist UNION ALL SELECT umi2 AS umi FROM edgelist"
            ")"
        ).fetchone()[0]
        input_reads = con.execute("SELECT SUM(read_count) FROM edgelist").fetchone()[0]

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "graph-collapse",
            str(pxl_path),
            "--output",
            d,
            "--design",
            "proxiome-v1",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0, cmd.output

        stage_dir = Path(d) / "graph-collapse"
        pxl_files = list(stage_dir.glob("*.graph-collapse.pxl"))
        report_files = list(stage_dir.glob("*.report.json"))

        assert len(pxl_files) == 1
        assert len(report_files) == 1

        report = json.loads(report_files[0].read_text())
        assert report["component_count"] > 0
        assert (
            report["nodes_considered"]
            >= report["targets_found"]
            >= report["nodes_collapsed"]
            >= 0
        )
        # The distributions cover the same targets, keyed by distance (JSON keys are strings).
        assert (
            sum(report["bfs_distance_distribution"].values()) == report["targets_found"]
        )
        assert (
            sum(report["hamming_distance_distribution"].values())
            == report["targets_found"]
        )
        bfs_distances = [int(d) for d in report["bfs_distance_distribution"]]
        assert bfs_distances == sorted(bfs_distances)
        assert all(d >= 2 and d % 2 == 0 for d in bfs_distances)
        hamming_distances = [int(d) for d in report["hamming_distance_distribution"]]
        assert hamming_distances == sorted(hamming_distances)
        assert all(1 <= d <= 2 for d in hamming_distances)
        assert report["umis_output"] <= report["umis_input"]
        assert report["umis_input"] - report["umis_output"] == report["nodes_collapsed"]
        # The fraction is over all nodes in the input graph, not only the searched ones.
        assert report["nodes_collapsed_fraction"] == pytest.approx(
            report["nodes_collapsed"] / report["umis_input"]
        )
        # Collapsing only re-attributes reads to a different UMI node, never discards them.
        assert report["reads_output"] == report["reads_input"]

        with duckdb.connect(str(pxl_files[0]), read_only=True) as con:
            output_umis = con.execute(
                "SELECT COUNT(DISTINCT umi) FROM ("
                "SELECT umi1 AS umi FROM edgelist UNION ALL SELECT umi2 AS umi FROM edgelist"
                ")"
            ).fetchone()[0]
            output_reads = con.execute(
                "SELECT SUM(read_count) FROM edgelist"
            ).fetchone()[0]
            adata_components = con.execute(
                "SELECT COUNT(*) FROM __adata__obs"
            ).fetchone()[0]

        assert output_umis <= input_umis
        assert output_reads == input_reads
        assert adata_components == report["component_count"]
        assert report["collapsed_sequences_file"] is None


def test_graph_collapse_cli_report_collapsed_sequences(pna_pxl_file):
    pxl_path = pna_pxl_file

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "graph-collapse",
            str(pxl_path),
            "--output",
            d,
            "--design",
            "proxiome-v1",
            "--report-collapsed-sequences",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0, cmd.output

        stage_dir = Path(d) / "graph-collapse"
        report = json.loads((stage_dir / "PNA055_Sample07_S7.report.json").read_text())

        sequences_path = Path(report["collapsed_sequences_file"])
        assert sequences_path.is_file()

        lines = sequences_path.read_text().splitlines()
        assert lines[0] == "component\toriginal_sequence\tcorrected_sequence"
        assert len(lines) - 1 == report["nodes_collapsed"]

        umi_length = len(lines[1].split("\t")[1])
        for line in lines[1:]:
            component, original_sequence, corrected_sequence = line.split("\t")
            assert component
            assert len(original_sequence) == len(corrected_sequence) == umi_length
            assert original_sequence != corrected_sequence
            assert set(original_sequence) <= set("ACGT")
            assert set(corrected_sequence) <= set("ACGT")


def test_graph_collapse_cli_memory_limit_reaches_duckdb(pna_pxl_file, monkeypatch):
    """--memory is a soft DuckDB limit, applied to the read and the writer connection."""
    pxl_path = pna_pxl_file
    seen = []

    import pixelator.pna.graph_collapse.collapse as collapse

    original_execute = collapse.duckdb.DuckDBPyConnection.execute

    def recording_execute(self, query, *args, **kwargs):
        result = original_execute(self, query, *args, **kwargs)
        if "memory_limit" in query:
            seen.append(
                original_execute(
                    self, "SELECT current_setting('memory_limit')"
                ).fetchone()[0]
            )
        return result

    monkeypatch.setattr(
        collapse.duckdb.DuckDBPyConnection, "execute", recording_execute
    )

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        cmd = runner.invoke(
            cli.main_cli,
            [
                "single-cell-pna",
                "graph-collapse",
                str(pxl_path),
                "--output",
                d,
                "--design",
                "proxiome-v1",
                "--memory",
                "2G",
            ],
        )

    assert cmd.exit_code == 0, cmd.output
    # 2G is 2000000000 bytes, which init_duckdb_conn expresses as 2000 MiB.
    assert seen == ["1.9 GiB", "1.9 GiB"]


def test_graph_collapse_cli_rejects_an_unparseable_memory_limit(pna_pxl_file):
    runner = CliRunner()
    cmd = runner.invoke(
        cli.main_cli,
        [
            "single-cell-pna",
            "graph-collapse",
            str(pna_pxl_file),
            "--output",
            "unused",
            "--design",
            "proxiome-v1",
            "--memory",
            "lots",
        ],
    )

    assert cmd.exit_code != 0
    assert "--memory option must be a positive integer" in cmd.output


def test_graph_collapse_cli_hamming_threshold_zero_collapses_nothing(pna_pxl_file):
    """Distinct UMIs differ in at least one symbol, so a threshold of 0 renames none."""
    pxl_path = pna_pxl_file

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "graph-collapse",
            str(pxl_path),
            "--output",
            d,
            "--design",
            "proxiome-v1",
            "--hamming-threshold",
            "0",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0, cmd.output

        report = json.loads(
            (Path(d) / "graph-collapse" / "PNA055_Sample07_S7.report.json").read_text()
        )
        assert report["targets_found"] == 0
        assert report["bfs_distance_distribution"] == {}
        assert report["hamming_distance_distribution"] == {}
        assert report["nodes_collapsed"] == 0
        assert report["umis_output"] == report["umis_input"]
