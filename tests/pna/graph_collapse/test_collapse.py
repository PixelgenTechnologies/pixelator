"""Tests for the graph-collapse rename rules.

Copyright © 2026 Pixelgen Technologies AB.
"""

import duckdb

from pixelator.pna.graph_collapse.collapse import (
    collapse_identical_edges,
    drop_renames_with_different_markers,
    rename_umis,
)


def _connection():
    """One-component edgelist shaped like the real graph: a tree, with every UMI on one side.

    The umi1 side holds UMI 1 and 2 (CD3) and UMI 3 (CD19). The umi2 side holds UMI 4
    and 5 (CD4) and UMI 8 (CD8). The edges are 1-4, 3-4, 3-5, 3-8 and 2-8.

    The renames are 1→2 and 3→2 on the umi1 side, and 4→5 on the umi2 side, so a
    rename has to update both columns.
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE edgelist (
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            umi1 UBIGINT,
            umi2 UBIGINT
        )
    """)
    con.execute("""
        INSERT INTO edgelist VALUES
            ('c', 'CD3', 'CD4', 1, 4),
            ('c', 'CD19', 'CD4', 3, 4),
            ('c', 'CD19', 'CD4', 3, 5),
            ('c', 'CD19', 'CD8', 3, 8),
            ('c', 'CD3', 'CD8', 2, 8)
    """)
    con.execute("""
        CREATE TEMP TABLE _renames (original UBIGINT, corrected UBIGINT)
    """)
    con.execute("""
        INSERT INTO _renames VALUES (1, 2), (3, 2), (4, 5)
    """)
    return con


def test_pairs_with_different_markers_are_removed():
    """1→2 and 4→5 stay because the markers match. 3→2 is removed because CD19 is not CD3."""
    con = _connection()

    drop_renames_with_different_markers(con)

    assert con.execute(
        "SELECT original, corrected FROM _renames ORDER BY original"
    ).fetchall() == [(1, 2), (4, 5)]


def test_remaining_original_umis_are_renamed_in_both_columns():
    """After the marker filter, UMI 1 becomes 2 in umi1 and UMI 4 becomes 5 in umi2.

    (1, 4) becomes (2, 5), and (3, 4) becomes (3, 5), which duplicates the existing
    (3, 5) edge. Merging duplicates is left to ``collapse_identical_edges``. The 3→2
    pair had different markers and was dropped, so UMI 3 keeps its place.
    """
    con = _connection()
    drop_renames_with_different_markers(con)
    rename_umis(con)

    rows = con.execute("""
        SELECT umi1, umi2 FROM edgelist ORDER BY umi1, umi2
    """).fetchall()
    assert rows == [(2, 5), (2, 8), (3, 5), (3, 5), (3, 8)]


def test_identical_edges_sum_their_counts():
    """Rows that match on every non-count column become one row.

    In component ``c``, UMI 1 and 2 are on the umi1 side and UMI 4 and 8 on the
    umi2 side. 1→2 makes (1, 4) the same edge as the existing (2, 4), so their
    counts add. (1, 8) becomes (2, 8), a different UMI pair, so it stays its own
    row. The same UMIs in component ``d`` stay separate.
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE edgelist (
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
        INSERT INTO edgelist VALUES
            ('c', 'CD3', 'CD4', 1, 4, 2, 1, 0),
            ('c', 'CD3', 'CD4', 2, 4, 5, 3, 1),
            ('c', 'CD3', 'CD8', 1, 8, 1, 1, 0),
            ('d', 'CD3', 'CD4', 2, 4, 9, 2, 4)
    """)
    con.execute("CREATE TEMP TABLE _renames (original UBIGINT, corrected UBIGINT)")
    con.execute("INSERT INTO _renames VALUES (1, 2)")

    rename_umis(con)
    collapse_identical_edges(con)

    rows = con.execute("""
        SELECT component, marker_2, umi1, umi2, read_count, uei_count, corrected_read_count
        FROM edgelist
        ORDER BY component, marker_2
    """).fetchall()
    assert rows == [
        ("c", "CD4", 2, 4, 7, 4, 1),
        ("c", "CD8", 2, 8, 1, 1, 0),
        ("d", "CD4", 2, 4, 9, 2, 4),
    ]


def test_record_batches_stream_into_graph_collapse():
    """DuckDB to_arrow_reader feeds the binding the component-sorted edgelist.

    The chain is the binding's own fixture, written with every UMI on one side:
    GGGG and TTTT are umi1, and AAAA, CCAA, CAAA and GTGT are umi2. CCAA and CAAA
    collapse onto AAAA, and a second component does not.
    """
    import pixelator_core

    umi = pixelator_core.encode_dna_2bit
    edges = [
        (umi("GGGG"), umi("AAAA"), "cell"),
        (umi("GGGG"), umi("CCAA"), "cell"),
        (umi("GGGG"), umi("CAAA"), "cell"),
        (umi("TTTT"), umi("AAAA"), "cell"),
        (umi("ACAC"), umi("GTGT"), "other"),
    ]
    con = duckdb.connect()
    con.execute("CREATE TABLE edgelist (umi1 UBIGINT, umi2 UBIGINT, component VARCHAR)")
    con.executemany("INSERT INTO edgelist VALUES (?, ?, ?)", edges)
    con.execute("""
        SELECT umi1, umi2, CAST(component AS VARCHAR) AS component
        FROM edgelist
        ORDER BY component
    """)

    table, _stats = pixelator_core.graph_collapse(
        con.to_arrow_reader(),
        n_threads=1,
        bfs_threshold=2,
        hamming_threshold=1,
    )
    pairs = set(
        zip(
            table.column("original").to_pylist(),
            table.column("corrected").to_pylist(),
            strict=True,
        )
    )

    assert pairs == {(umi("CCAA"), umi("AAAA")), (umi("CAAA"), umi("AAAA"))}
