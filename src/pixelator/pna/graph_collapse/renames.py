"""The steps that find and apply renames for one batch of components.

The binding is called on the UMIs of ``_batch_edges``, and the SQL steps work on the
``_renames`` table it fills. See the ``collapse`` module for how the steps are chained
and which temp objects they pass to each other.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pixelator_core
import pyarrow as pa

from pixelator.pna.utils.two_bit_encoding import unpack_2bits

_COUNT_COLUMNS = ("read_count", "uei_count", "corrected_read_count")
_SORT_COLUMNS = ("component", "marker_1", "marker_2")


@dataclass(frozen=True)
class CollapseParams:
    """The settings of the binding's search.

    Attributes:
        n_threads: Passed to the binding as ``n_threads``.
        bfs_distance: Passed to the binding as ``bfs_threshold``.
        hamming_threshold: Passed to the binding as ``hamming_threshold``.
        max_degree: Passed to the binding as ``max_degree``. Nodes with a higher
            degree are not searched.

    """

    n_threads: int
    bfs_distance: int
    hamming_threshold: int
    max_degree: int


def decode_umi(value: int, length: int) -> bytes:
    """Decode a 2-bit-packed UMI integer into its nucleotide sequence.

    Args:
        value: The packed UMI value, as stored in an edgelist's ``umi1``/``umi2`` column.
        length: The number of nucleotides the UMI was packed from.

    Returns:
        The decoded nucleotide sequence.

    """
    return unpack_2bits(int(value), length)


def fetch_batch_edges(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    """Read the UMIs and component numbers of ``_batch_edges`` as numpy arrays.

    Args:
        con: Connection with an ``_batch_edges`` relation and a
            ``_batch_components`` table that numbers its components.

    Returns:
        The ``umi1``, ``umi2`` and ``component_id`` columns.

    """
    return con.execute("""
        SELECT e.umi1, e.umi2, c.component_id
        FROM _batch_edges e
        JOIN _batch_components c USING (component)
    """).fetchnumpy()


def search_renames(
    edges: dict[str, np.ndarray], params: CollapseParams
) -> tuple[dict, dict]:
    """Run the binding on the edges of a batch.

    Args:
        edges: The output of ``fetch_batch_edges``.
        params: The binding settings.

    Returns:
        The binding's ``(component_id, original, corrected)`` columns and its
        stats for the batch.

    """
    return pixelator_core.graph_collapse(
        edges["umi1"],
        edges["umi2"],
        edges["component_id"],
        params.n_threads,
        bfs_threshold=params.bfs_distance,
        hamming_threshold=params.hamming_threshold,
        max_degree=params.max_degree,
    )


def store_renames(con: duckdb.DuckDBPyConnection, renames: dict) -> None:
    """Store the binding's renames in ``_renames``, with component names.

    Args:
        con: Connection with the ``_batch_components`` table of the batch.
        renames: The renames returned by ``search_renames``.

    """
    con.register("_renames_arrow", pa.table(renames))
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _renames AS
        SELECT c.component, r.original, r.corrected
        FROM _renames_arrow r
        JOIN _batch_components c ON c.component_id = r.component
    """)
    con.unregister("_renames_arrow")


def drop_renames_with_different_markers(con: duckdb.DuckDBPyConnection) -> None:
    """Delete ``_renames`` rows whose original and corrected UMI markers differ.

    ``marker_1`` is the marker of ``umi1`` and ``marker_2`` is the marker of
    ``umi2``. A pair is removed when any marker recorded for the original UMI
    in its component differs from any marker recorded for the corrected UMI in
    that component. Only the UMIs in ``_renames`` are looked up.

    Also leaves ``_umi_marker`` with the component, UMI, marker and side
    (``'umi1'`` or ``'umi2'``) of every UMI in ``_renames``.

    Args:
        con: Connection with an ``_batch_edges`` relation and a ``_renames`` table
            of ``component``, ``original`` and ``corrected``.

    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _umi_marker AS
        WITH renamed AS (
            SELECT component, original AS umi FROM _renames
            UNION
            SELECT component, corrected FROM _renames
        )
        SELECT DISTINCT e.component, e.umi1 AS umi, e.marker_1 AS marker, 'umi1' AS side
        FROM _batch_edges e SEMI JOIN renamed r
            ON r.component = e.component AND r.umi = e.umi1
        UNION
        SELECT DISTINCT e.component, e.umi2, e.marker_2, 'umi2'
        FROM _batch_edges e SEMI JOIN renamed r
            ON r.component = e.component AND r.umi = e.umi2
    """)
    con.execute("""
        DELETE FROM _renames
        USING _umi_marker AS original_marker, _umi_marker AS corrected_marker
        WHERE _renames.component = original_marker.component
          AND _renames.original = original_marker.umi
          AND _renames.component = corrected_marker.component
          AND _renames.corrected = corrected_marker.umi
          AND original_marker.marker IS DISTINCT FROM corrected_marker.marker
    """)


def collapsed_sequences(
    con: duckdb.DuckDBPyConnection,
    umi1_length: int,
    umi2_length: int,
) -> list[dict]:
    """Decode the kept renames of ``_renames`` into nucleotide sequences.

    Args:
        con: Connection with the ``_renames`` and ``_umi_marker`` tables that
            ``drop_renames_with_different_markers`` leaves.
        umi1_length: Nucleotide length used to decode a renamed UMI seen as umi1.
        umi2_length: Nucleotide length used to decode a renamed UMI seen only as umi2.

    Returns:
        One ``component``, ``original_sequence`` and ``corrected_sequence`` record
        per rename.

    """
    found = con.execute("""
        SELECT r.component, r.original, r.corrected, BOOL_OR(m.side = 'umi1') AS seen_as_umi1
        FROM _renames r
        JOIN _umi_marker m ON m.component = r.component AND m.umi = r.original
        GROUP BY r.component, r.original, r.corrected
    """).fetchall()

    rows = []
    for component, original, corrected, seen_as_umi1 in found:
        length = umi1_length if seen_as_umi1 else umi2_length
        rows.append(
            {
                "component": component,
                "original_sequence": decode_umi(original, length).decode(),
                "corrected_sequence": decode_umi(corrected, length).decode(),
            }
        )
    return rows


def write_collapsed_edges(
    con: duckdb.DuckDBPyConnection, target_table: str = "edgelist"
) -> None:
    """Rename ``_batch_edges`` by ``_renames`` and append it to ``target_table``.

    Each original UMI in ``_renames`` is replaced by its corrected UMI, on both
    ends, within its own component. Rows that then share every non-count column
    are merged, summing ``read_count``, ``uei_count``, and
    ``corrected_read_count`` (when present). The same UMIs with a different
    marker stay separate, as do the same UMIs in a different component.

    Rows are appended sorted by component and markers, the order every pxl
    edgelist is stored in. Batches follow component order, so the whole table
    stays sorted.

    Args:
        con: Connection with an ``_batch_edges`` relation, a ``_renames`` table
            and ``target_table``, which has the same columns as ``_batch_edges``.
        target_table: Table the merged rows are appended to.

    """
    columns = [row[0] for row in con.execute("DESCRIBE _batch_edges").fetchall()]
    renamed = {
        "umi1": "COALESCE(r1.corrected, e.umi1)",
        "umi2": "COALESCE(r2.corrected, e.umi2)",
    }
    select_sql = ", ".join(
        f"SUM(e.{column}) AS {column}"
        if column in _COUNT_COLUMNS
        else f"{renamed.get(column, f'e.{column}')} AS {column}"
        for column in columns
    )
    order_by = ", ".join(column for column in _SORT_COLUMNS if column in columns)
    con.execute(f"""
        INSERT INTO {target_table} BY NAME
        SELECT * FROM (
            SELECT {select_sql}
            FROM _batch_edges e
            LEFT JOIN _renames r1 ON r1.component = e.component AND r1.original = e.umi1
            LEFT JOIN _renames r2 ON r2.component = e.component AND r2.original = e.umi2
            GROUP BY ALL
        )
        ORDER BY {order_by}
    """)
