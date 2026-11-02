"""Graph-collapse: rename likely UMI sequencing errors on low-degree nodes, per component.

A UMI sequencing error is unlikely to recur, so an erroneous UMI typically shows up as a
leaf in its component's graph. The Rust binding ``pixelator_core.graph_collapse``
searches every node whose degree is at most ``max_degree`` (leaves by default) and finds,
for each, the nearest UMI on the same side within a Hamming-distance cutoff,
and returns ``(original, corrected)`` UMIs. It does not know markers. Three SQL
statements finish the job: drop pairs whose markers differ, rename the remaining
original UMIs, then merge edges that are identical after that rename.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile
from pixelator.pna.utils import init_duckdb_conn
from pixelator.pna.utils.two_bit_encoding import unpack_2bits

logger = logging.getLogger(__name__)

_COUNT_COLUMNS = ("read_count", "uei_count", "corrected_read_count")


def decode_umi(value: int, length: int) -> bytes:
    """Decode a 2-bit-packed UMI integer into its nucleotide sequence.

    Args:
        value: The packed UMI value, as stored in an edgelist's ``umi1``/``umi2`` column.
        length: The number of nucleotides the UMI was packed from.

    Returns:
        The decoded nucleotide sequence.

    """
    return unpack_2bits(int(value), length)


@dataclass
class GraphCollapseResult:
    """Stats the graph-collapse report reads after the collapse.

    ``component_count``, ``nodes_considered``, ``targets_found`` and the two
    distributions come from the binding. ``targets_found`` and the
    distributions describe the matches the search kept, before pairs with
    different markers are dropped, so ``nodes_collapsed`` can be smaller.
    """

    component_count: int
    nodes_considered: int
    targets_found: int
    bfs_distance_distribution: dict[int, int]
    hamming_distance_distribution: dict[int, int]
    nodes_collapsed: int
    collapsed_sequences: list[dict]


def _store_renames(con: duckdb.DuckDBPyConnection, collapsed) -> None:
    """Copy the binding output into a table that later statements can modify."""
    con.register("_renames_arrow", collapsed)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _renames AS
        SELECT original, corrected FROM _renames_arrow
    """)
    con.unregister("_renames_arrow")


def drop_renames_with_different_markers(con: duckdb.DuckDBPyConnection) -> None:
    """Delete ``_renames`` rows whose original and corrected UMI markers differ.

    ``marker_1`` is the marker of ``umi1`` and ``marker_2`` is the marker of
    ``umi2``. A pair is removed when any marker recorded for the original UMI
    differs from any marker recorded for the corrected UMI.

    Args:
        con: Connection with an ``edgelist`` table and a ``_renames`` table.

    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _umi_marker AS
        SELECT DISTINCT umi1 AS umi, marker_1 AS marker FROM edgelist
        UNION
        SELECT DISTINCT umi2, marker_2 FROM edgelist
    """)
    con.execute("""
        DELETE FROM _renames
        USING _umi_marker AS original_marker, _umi_marker AS corrected_marker
        WHERE _renames.original = original_marker.umi
          AND _renames.corrected = corrected_marker.umi
          AND original_marker.marker IS DISTINCT FROM corrected_marker.marker
    """)


def rename_umis(con: duckdb.DuckDBPyConnection) -> None:
    """Replace each remaining original UMI with its corrected UMI, on both ends.

    Args:
        con: Connection with an ``edgelist`` table and a ``_renames`` table.

    """
    con.execute("""
        UPDATE edgelist
        SET umi1 = _renames.corrected
        FROM _renames
        WHERE edgelist.umi1 = _renames.original
    """)
    con.execute("""
        UPDATE edgelist
        SET umi2 = _renames.corrected
        FROM _renames
        WHERE edgelist.umi2 = _renames.original
    """)


def collapse_identical_edges(con: duckdb.DuckDBPyConnection) -> None:
    """Merge edgelist rows that share every non-count column, summing counts.

    ``read_count``, ``uei_count``, and ``corrected_read_count`` are the count
    columns, when the table has them. Every other column is part of the edge
    identity, so two rows with the same UMIs and a different marker stay
    separate, as do the same UMIs in a different component.

    Args:
        con: Connection with an ``edgelist`` table.

    """
    columns = [row[0] for row in con.execute("DESCRIBE edgelist").fetchall()]
    count_columns = [column for column in _COUNT_COLUMNS if column in columns]
    group_columns = [column for column in columns if column not in count_columns]
    select_sql = ", ".join(
        f"SUM({column}) AS {column}" if column in count_columns else column
        for column in columns
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE edgelist AS
        SELECT {select_sql}
        FROM edgelist
        GROUP BY {", ".join(group_columns)}
        ORDER BY component
    """)


def _collapsed_sequences(
    con: duckdb.DuckDBPyConnection,
    umi1_length: int,
    umi2_length: int,
) -> list[dict]:
    found = con.execute("""
        SELECT
            r.original,
            r.corrected,
            ends.component,
            BOOL_OR(ends.side = 'umi1') AS seen_as_umi1
        FROM _renames r
        JOIN (
            SELECT component, umi1 AS umi, 'umi1' AS side FROM edgelist
            UNION ALL
            SELECT component, umi2, 'umi2' FROM edgelist
        ) ends
            ON ends.umi = r.original
        GROUP BY r.original, r.corrected, ends.component
    """).fetchdf()

    rows = []
    for row in found.itertuples(index=False):
        length = umi1_length if row.seen_as_umi1 else umi2_length
        rows.append(
            {
                "component": row.component,
                "original_sequence": decode_umi(int(row.original), length).decode(),
                "corrected_sequence": decode_umi(int(row.corrected), length).decode(),
            }
        )
    rows.sort(key=lambda record: (record["component"], record["original_sequence"]))
    return rows


def run_graph_collapse(
    pxl_file_target: PxlFile,
    *,
    n_threads: int,
    bfs_distance: int,
    hamming_threshold: int,
    umi1_length: int,
    umi2_length: int,
    max_degree: int = 1,
    report_collapsed_sequences: bool = False,
    memory_limit: int | None = None,
    temp_directory: str | Path | None = None,
) -> GraphCollapseResult:
    """Collapse the pxl edgelist with the Rust binding and rename matching UMIs.

    The edgelist is streamed in component order. ``to_arrow_reader`` is the
    DuckDB method that yields the record-batch reader the binding consumes. The
    binding sorts the rows within each component itself, so the result does not
    depend on the order DuckDB returns them in.

    Args:
        pxl_file_target: Copied pxl file updated in place.
        n_threads: Passed to the binding as ``n_threads``.
        bfs_distance: Passed to the binding as ``bfs_threshold``.
        hamming_threshold: Passed to the binding as ``hamming_threshold``.
        umi1_length: Nucleotide length used to decode a renamed UMI seen as umi1.
        umi2_length: Nucleotide length used to decode a renamed UMI seen only as umi2.
        max_degree: Passed to the binding as ``max_degree``. Nodes with a higher
            degree are not searched. The default, 1, investigates only leaves.
        report_collapsed_sequences: Decode each kept rename into nucleotide
            sequences. Off unless the CLI is writing the collapsed-sequence TSV.
        memory_limit: Soft memory limit in bytes for the DuckDB connections that
            read and rewrite the edgelist. When set, DuckDB spills to disk once
            the limit is reached. The native search is not covered by it.
        temp_directory: DuckDB spill directory. Defaults to
            ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp``.

    Returns:
        Counts and, when requested, the collapsed-sequence rows.

    """
    import pixelator_core

    # The record-batch reader keeps this connection busy, so the edgelist is
    # updated afterwards on the writer's connection.
    with init_duckdb_conn(
        str(pxl_file_target.path),
        read_only=True,
        memory_limit=memory_limit,
        threads=n_threads,
        temp_dir=temp_directory,
    ) as con:
        con.execute("""
            SELECT umi1, umi2, CAST(component AS VARCHAR) AS component
            FROM edgelist
            ORDER BY component
        """)
        reader = con.to_arrow_reader()
        logger.info("Starting graph collapse")
        collapsed, stats = pixelator_core.graph_collapse(
            reader,
            n_threads,
            bfs_threshold=bfs_distance,
            hamming_threshold=hamming_threshold,
            max_degree=max_degree,
        )
        _store_renames(con, collapsed)
        drop_renames_with_different_markers(con)
        sequences = []
        if report_collapsed_sequences:
            logger.info("Starting collapsed-sequence collection")
            sequences = _collapsed_sequences(con, umi1_length, umi2_length)
        con.execute("SELECT original, corrected FROM _renames")
        kept = con.to_arrow_table()

    # The binding maps each original UMI to one corrected UMI, so every kept
    # row is a distinct collapsed node.
    nodes_collapsed = kept.num_rows
    if nodes_collapsed:
        # Rebuild counts from the markers already in var. Using the full panel
        # would put markers removed by an earlier step (e.g. hashing clones
        # removed by sample calling) back. The panel comes from the file only.
        input_dataset = read(pxl_file_target.path)
        panel = PNAAntibodyPanel.from_pxl_dataset(input_dataset)
        markers = list(input_dataset.adata().var_names)
        with PixelFileWriter(pxl_file_target.path, temp_dir=temp_directory) as writer:
            con = writer.get_connection()
            if memory_limit is not None:
                con.execute(f"SET memory_limit = '{memory_limit / 10**6}MiB'")
            _store_renames(con, kept)
            rename_umis(con)
            collapse_identical_edges(con)
            adata = pna_edgelist_to_anndata(con, panel, markers=markers)
            adata = adata[adata.obs_names.sort_values(), :]
            writer.write_adata(adata)

    return GraphCollapseResult(
        component_count=stats["total_components"],
        nodes_considered=stats["nodes_considered"],
        targets_found=stats["targets_found"],
        bfs_distance_distribution=stats["bfs_distance_distribution"],
        hamming_distance_distribution=stats["hamming_distance_distribution"],
        nodes_collapsed=nodes_collapsed,
        collapsed_sequences=sequences,
    )
