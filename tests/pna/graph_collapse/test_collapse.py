"""Tests for the graph-collapse rename rules and the batched driver.

Copyright © 2026 Pixelgen Technologies AB.
"""

import shutil

import duckdb

from pixelator.pna.graph_collapse.collapse import (
    _select_batch,
    drop_renames_with_different_markers,
    find_renames,
    plan_component_batches,
    run_graph_collapse,
    write_collapsed_edges,
)
from pixelator.pna.pixeldataset.io import PxlFile


def test_batches_are_filled_to_the_edge_count_in_order():
    """A batch closes once it reaches the edge count, so c and e stand alone."""
    sizes = [("a", 3), ("b", 4), ("c", 20), ("d", 2), ("e", 9), ("f", 1)]

    assert plan_component_batches(sizes, 7) == [["a", "b"], ["c"], ["d", "e"], ["f"]]
    assert plan_component_batches(sizes, 1000) == [["a", "b", "c", "d", "e", "f"]]
    assert plan_component_batches([], 7) == []


def _connection():
    """Two components shaped like the real graph: trees, with every UMI on one side.

    In component ``c``, the umi1 side holds UMI 1 and 2 (CD3) and UMI 3 (CD19). The
    umi2 side holds UMI 4 and 5 (CD4) and UMI 8 (CD8). The edges are 1-4, 3-4, 3-5,
    3-8 and 2-8.

    The renames in ``c`` are 1→2 and 3→2 on the umi1 side, and 4→5 on the umi2 side,
    so a rename has to update both columns.

    Component ``d`` reuses UMI 1 and 2, but UMI 1 is CD19 there. Its rename 1→2 is
    dropped, which must not affect component ``c``, and the 1→2 rename of ``c``
    must not rename UMI 1 in ``d``.
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE _batch_edges (
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            umi1 UBIGINT,
            umi2 UBIGINT
        )
    """)
    con.execute("""
        INSERT INTO _batch_edges VALUES
            ('c', 'CD3', 'CD4', 1, 4),
            ('c', 'CD19', 'CD4', 3, 4),
            ('c', 'CD19', 'CD4', 3, 5),
            ('c', 'CD19', 'CD8', 3, 8),
            ('c', 'CD3', 'CD8', 2, 8),
            ('d', 'CD19', 'CD4', 1, 4),
            ('d', 'CD3', 'CD4', 2, 4)
    """)
    con.execute("""
        CREATE TEMP TABLE _renames (component VARCHAR, original UBIGINT, corrected UBIGINT)
    """)
    con.execute("""
        INSERT INTO _renames VALUES ('c', 1, 2), ('c', 3, 2), ('c', 4, 5), ('d', 1, 2)
    """)
    con.execute("CREATE TABLE edgelist AS SELECT * FROM _batch_edges LIMIT 0")
    return con


def test_pairs_with_different_markers_are_removed_per_component():
    """1→2 and 4→5 stay in ``c`` because the markers match there.

    3→2 is removed because CD19 is not CD3, and 1→2 in ``d`` is removed because
    UMI 1 is CD19 in ``d``.
    """
    con = _connection()

    drop_renames_with_different_markers(con)

    assert con.execute(
        "SELECT component, original, corrected FROM _renames ORDER BY ALL"
    ).fetchall() == [("c", 1, 2), ("c", 4, 5)]


def test_marker_lookup_records_the_side_of_each_renamed_umi():
    """``_umi_marker`` holds only the UMIs in ``_renames``, with their side."""
    con = _connection()

    drop_renames_with_different_markers(con)

    assert con.execute(
        "SELECT component, umi, marker, side FROM _umi_marker ORDER BY ALL"
    ).fetchall() == [
        ("c", 1, "CD3", "umi1"),
        ("c", 2, "CD3", "umi1"),
        ("c", 3, "CD19", "umi1"),
        ("c", 4, "CD4", "umi2"),
        ("c", 5, "CD4", "umi2"),
        ("d", 1, "CD19", "umi1"),
        ("d", 2, "CD3", "umi1"),
    ]


def test_remaining_original_umis_are_renamed_in_both_columns_within_their_component():
    """After the marker filter, UMI 1 becomes 2 in umi1 and UMI 4 becomes 5 in umi2, in ``c`` only.

    In ``c``, (1, 4) becomes (2, 5), and (3, 4) becomes (3, 5), which merges with
    the existing (3, 5) edge. The 3→2 pair had different markers and was dropped,
    so UMI 3 keeps its place. Component ``d`` is unchanged.
    """
    con = _connection()
    drop_renames_with_different_markers(con)
    write_collapsed_edges(con)

    rows = con.execute("""
        SELECT component, umi1, umi2 FROM edgelist ORDER BY ALL
    """).fetchall()
    assert rows == [
        ("c", 2, 5),
        ("c", 2, 8),
        ("c", 3, 5),
        ("c", 3, 8),
        ("d", 1, 4),
        ("d", 2, 4),
    ]


def test_identical_edges_sum_their_counts():
    """Rows that match on every non-count column become one row, keeping the column types.

    In component ``c``, UMI 1 and 2 are on the umi1 side and UMI 4 and 8 on the
    umi2 side. 1→2 makes (1, 4) the same edge as the existing (2, 4), so their
    counts add. (1, 8) becomes (2, 8), a different UMI pair, so it stays its own
    row. The same UMIs in component ``d`` stay separate.
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE _batch_edges (
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            umi1 UBIGINT,
            umi2 UBIGINT,
            read_count UINTEGER,
            uei_count USMALLINT,
            corrected_read_count UINTEGER
        )
    """)
    con.execute("""
        INSERT INTO _batch_edges VALUES
            ('c', 'CD3', 'CD4', 1, 4, 2, 1, 0),
            ('c', 'CD3', 'CD4', 2, 4, 5, 3, 1),
            ('c', 'CD3', 'CD8', 1, 8, 1, 1, 0),
            ('d', 'CD3', 'CD4', 2, 4, 9, 2, 4)
    """)
    con.execute(
        "CREATE TEMP TABLE _renames (component VARCHAR, original UBIGINT, corrected UBIGINT)"
    )
    con.execute("INSERT INTO _renames VALUES ('c', 1, 2)")
    con.execute("CREATE TABLE edgelist AS SELECT * FROM _batch_edges LIMIT 0")

    write_collapsed_edges(con)

    rows = con.execute("""
        SELECT component, marker_2, umi1, umi2, read_count, uei_count, corrected_read_count
        FROM edgelist
    """).fetchall()
    # Appended in component and marker order.
    assert rows == [
        ("c", "CD4", 2, 4, 7, 4, 1),
        ("c", "CD8", 2, 8, 1, 1, 0),
        ("d", "CD4", 2, 4, 9, 2, 4),
    ]
    types = dict(
        con.execute(
            "SELECT column_name, column_type FROM (DESCRIBE edgelist)"
        ).fetchall()
    )
    assert types["read_count"] == "UINTEGER"
    assert types["uei_count"] == "USMALLINT"


def test_marker_sequences_are_kept_on_merged_edges():
    """``marker_sequence_1`` and ``marker_sequence_2`` follow their markers through a merge.

    1→2 makes (1, 4) the same edge as (2, 4). Both rows carry the same marker
    sequences, so they still merge, and the merged row keeps them.
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE _batch_edges (
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            umi1 UBIGINT,
            umi2 UBIGINT,
            marker_sequence_1 VARCHAR,
            marker_sequence_2 VARCHAR,
            read_count UINTEGER
        )
    """)
    con.execute("""
        INSERT INTO _batch_edges VALUES
            ('c', 'CD3', 'CD4', 1, 4, 'AAAA', 'CCCC', 2),
            ('c', 'CD3', 'CD4', 2, 4, 'AAAA', 'CCCC', 5),
            ('c', 'CD3', 'CD8', 2, 8, 'AAAA', 'GGGG', 1)
    """)
    con.execute(
        "CREATE TEMP TABLE _renames (component VARCHAR, original UBIGINT, corrected UBIGINT)"
    )
    con.execute("INSERT INTO _renames VALUES ('c', 1, 2)")
    con.execute("CREATE TABLE edgelist AS SELECT * FROM _batch_edges LIMIT 0")

    write_collapsed_edges(con)

    assert con.execute("""
        SELECT umi1, umi2, marker_sequence_1, marker_sequence_2, read_count
        FROM edgelist
        ORDER BY ALL
    """).fetchall() == [
        (2, 4, "AAAA", "CCCC", 7),
        (2, 8, "AAAA", "GGGG", 1),
    ]


def test_renames_are_found_per_component_of_the_batch():
    """The binding's component ids map back to the names of the batch.

    The chain is the binding's own fixture, written with every UMI on one side:
    GGGG and TTTT are umi1, and AAAA, CCAA, CAAA and GTGT are umi2. CCAA and CAAA
    collapse onto AAAA in ``cell``, ``other`` does not collapse, and ``outside``
    is not in the batch, so its copy of the chain is not searched.
    """
    import pixelator_core

    umi = pixelator_core.encode_dna_2bit
    chain = [
        (umi("GGGG"), umi("AAAA")),
        (umi("GGGG"), umi("CCAA")),
        (umi("GGGG"), umi("CAAA")),
        (umi("TTTT"), umi("AAAA")),
    ]
    edges = [(u1, u2, "cell") for u1, u2 in chain]
    edges += [(umi("ACAC"), umi("GTGT"), "other")]
    edges += [(u1, u2, "outside") for u1, u2 in chain]
    con = duckdb.connect()
    con.execute("CREATE TABLE edgelist (umi1 UBIGINT, umi2 UBIGINT, component VARCHAR)")
    con.executemany("INSERT INTO edgelist VALUES (?, ?, ?)", edges)

    _select_batch(con, "edgelist", ["cell", "other"])
    stats = find_renames(
        con, n_threads=1, bfs_distance=2, hamming_threshold=1, max_degree=1
    )

    assert stats["total_components"] == 2
    assert con.execute(
        "SELECT component, original, corrected FROM _renames ORDER BY ALL"
    ).fetchall() == sorted(
        [
            ("cell", umi("CCAA"), umi("AAAA")),
            ("cell", umi("CAAA"), umi("AAAA")),
        ]
    )


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
    and the input's tables other than the AnnData ones are carried over.
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
    assert large["tables"] == input_before["tables"]
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
