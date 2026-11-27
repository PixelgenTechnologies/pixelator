"""Tests for the batched graph-collapse driver.

Copyright © 2026 Pixelgen Technologies AB.
"""

import shutil

import duckdb

from pixelator.pna.graph_collapse.collapse import run_graph_collapse
from pixelator.pna.pixeldataset.io import PxlFile


def _collapse(pxl_path, output_path, batch_edges):
    return run_graph_collapse(
        PxlFile(pxl_path),
        output_path,
        n_threads=2,
        bfs_distance=6,
        hamming_threshold=2,
        umi1_length=28,
        umi2_length=28,
        report_collapsed_sequences=True,
        batch_edges=batch_edges,
    )


def _tables(path):
    with duckdb.connect(str(path), read_only=True) as con:
        return {
            # Row order within one component and marker pair is not defined.
            "edgelist": con.execute("SELECT * FROM edgelist ORDER BY ALL").fetchall(),
            "stored_components": [
                row[0]
                for row in con.execute("SELECT component FROM edgelist").fetchall()
            ],
            "obs": con.execute("SELECT * FROM __adata__obs ORDER BY index").fetchall(),
            "metadata": con.execute("SELECT * FROM metadata").fetchall(),
            # The AnnData tables are rebuilt from the collapsed edgelist.
            "tables": sorted(
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM duckdb_tables() "
                    "WHERE NOT starts_with(table_name, '__adata__')"
                ).fetchall()
            ),
        }


def test_result_does_not_depend_on_the_batch_size(graph_pxl_file, tmp_path):
    """One component per batch and every component in one batch give the same file.

    The edgelist is also stored in component order, the input is left untouched,
    and the output has the input's metadata and panel, but not its layouts or
    proximity, which the collapse makes stale.
    """
    input_before = _tables(graph_pxl_file)

    one_per_batch = _collapse(graph_pxl_file, tmp_path / "small.pxl", batch_edges=1)
    one_batch = _collapse(graph_pxl_file, tmp_path / "large.pxl", batch_edges=10**12)

    assert one_per_batch == one_batch
    assert one_batch.nodes_collapsed > 0
    assert len(one_batch.collapsed_sequences) == one_batch.nodes_collapsed

    small, large = _tables(tmp_path / "small.pxl"), _tables(tmp_path / "large.pxl")
    assert small == large
    assert _tables(graph_pxl_file) == input_before

    stored = large["stored_components"]
    assert stored == sorted(stored)
    assert large["metadata"] == input_before["metadata"]
    assert {"layouts", "proximity"} <= set(input_before["tables"])
    assert large["tables"] == ["edgelist", "metadata", "panel_sources", "panels"]
    assert len(large["edgelist"]) < len(input_before["edgelist"])


def test_marker_sequences_are_carried_into_the_output(graph_pxl_file, tmp_path):
    """An edgelist with marker sequences keeps them, one per marker, and collapses as without.

    The fixture predates the marker sequence columns, so they are added here,
    derived from the marker names so that each marker has one sequence.
    """
    with_sequences = tmp_path / "with_sequences.pxl"
    shutil.copy(graph_pxl_file, with_sequences)
    with duckdb.connect(str(with_sequences)) as con:
        con.execute("""
            ALTER TABLE edgelist ADD COLUMN marker_sequence_1 VARCHAR;
            ALTER TABLE edgelist ADD COLUMN marker_sequence_2 VARCHAR;
            UPDATE edgelist SET
                marker_sequence_1 = 'seq-' || marker_1,
                marker_sequence_2 = 'seq-' || marker_2;
        """)

    without = _collapse(graph_pxl_file, tmp_path / "without.pxl", batch_edges=1)
    with_ = _collapse(with_sequences, tmp_path / "with.pxl", batch_edges=1)

    assert with_ == without
    with duckdb.connect(str(tmp_path / "with.pxl"), read_only=True) as con:
        columns = [row[0] for row in con.execute("DESCRIBE edgelist").fetchall()]
        assert {"marker_sequence_1", "marker_sequence_2"} <= set(columns)
        assert (
            con.execute("""
            SELECT COUNT(*) FROM edgelist
            WHERE marker_sequence_1 IS DISTINCT FROM 'seq-' || marker_1
               OR marker_sequence_2 IS DISTINCT FROM 'seq-' || marker_2
        """).fetchone()[0]
            == 0
        )
        n_with = con.execute("SELECT COUNT(*) FROM edgelist").fetchone()[0]
    with duckdb.connect(str(tmp_path / "without.pxl"), read_only=True) as con:
        n_without = con.execute("SELECT COUNT(*) FROM edgelist").fetchone()[0]
    assert n_with == n_without
