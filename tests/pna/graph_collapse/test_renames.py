"""Tests for the graph-collapse rename rules, one batch at a time.

Copyright © 2026 Pixelgen Technologies AB.
"""

import duckdb

from pixelator.pna.graph_collapse.batches import select_batch
from pixelator.pna.graph_collapse.renames import (
    CollapseParams,
    drop_renames_with_different_markers,
    fetch_batch_edges,
    search_renames,
    store_renames,
    write_collapsed_edges,
)


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

    select_batch(con, "edgelist", ["cell", "other"])
    params = CollapseParams(
        n_threads=1, bfs_distance=2, hamming_threshold=1, max_degree=1
    )
    renames, stats = search_renames(fetch_batch_edges(con), params)
    store_renames(con, renames)

    assert stats["total_components"] == 2
    assert con.execute(
        "SELECT component, original, corrected FROM _renames ORDER BY ALL"
    ).fetchall() == sorted(
        [
            ("cell", umi("CCAA"), umi("AAAA")),
            ("cell", umi("CAAA"), umi("AAAA")),
        ]
    )
