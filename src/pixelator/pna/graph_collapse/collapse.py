"""Graph-collapse: rename likely UMI sequencing errors on low-degree nodes, per component.

A UMI sequencing error is unlikely to recur, so an erroneous UMI typically shows up as a
leaf in its component's graph. The Rust binding ``pixelator_core.graph_collapse``
searches every node whose degree is at most ``max_degree`` (leaves by default) and finds,
for each, the nearest UMI on the same side within a Hamming-distance cutoff,
and returns ``(component, original, corrected)`` rows. It does not know markers. Three
SQL steps finish the job: drop pairs whose markers differ, rename the remaining
original UMIs, and merge edges that are identical after that rename.

The input edgelist is processed in batches of whole components and written to a new
pxl file as it goes, so memory follows the batch size rather than the file size. Every
step works within one component: a rename only applies to the component it was found
in, and the marker check only looks at markers recorded in that component.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.graph_collapse.constants import DEFAULT_BATCH_EDGES
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile
from pixelator.pna.utils.two_bit_encoding import unpack_2bits

logger = logging.getLogger(__name__)

_COUNT_COLUMNS = ("read_count", "uei_count", "corrected_read_count")
_SORT_COLUMNS = ("component", "marker_1", "marker_2")
_SOURCE = "collapse_input"
_STAT_KEYS = ("total_components", "nodes_considered", "targets_found")


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


def plan_component_batches(
    component_sizes: list[tuple[str, int]], batch_edges: int
) -> list[list[str]]:
    """Group consecutive components into batches of about ``batch_edges`` edges.

    A batch is closed as soon as it reaches ``batch_edges``, so a component with
    more edges than that ends up in a batch of its own.

    Args:
        component_sizes: ``(component, number of edges)`` pairs, in the order the
            batches should follow.
        batch_edges: Number of edges a batch is filled to.

    Returns:
        The component names of each batch, in order.

    """
    batches: list[list[str]] = []
    batch: list[str] = []
    edges_in_batch = 0
    for component, n_edges in component_sizes:
        batch.append(component)
        edges_in_batch += n_edges
        if edges_in_batch >= batch_edges:
            batches.append(batch)
            batch = []
            edges_in_batch = 0
    if batch:
        batches.append(batch)
    return batches


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fetch_value(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    """Run ``query``, which returns a single row, and return its first value."""
    row = con.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {query}")
    return row[0]


def _select_batch(
    con: duckdb.DuckDBPyConnection, source_table: str, components: list[str]
) -> None:
    """Point ``_batch_edges`` at the rows of ``components`` and number the components.

    ``components`` must be consecutive in component order, so a range filter
    selects exactly their rows. The edgelist is stored sorted by component, so
    DuckDB uses the row-group min/max statistics to read only those rows.
    """
    con.register(
        "_batch_components_arrow",
        pa.table(
            {
                "component": pa.array(components, type=pa.string()),
                "component_id": pa.array(range(len(components)), type=pa.uint32()),
            }
        ),
    )
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _batch_components AS
        SELECT component, component_id FROM _batch_components_arrow
    """)
    con.unregister("_batch_components_arrow")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW _batch_edges AS
        SELECT * FROM {source_table}
        WHERE component BETWEEN {_sql_string(components[0])}
            AND {_sql_string(components[-1])}
    """)


def find_renames(
    con: duckdb.DuckDBPyConnection,
    *,
    n_threads: int,
    bfs_distance: int,
    hamming_threshold: int,
    max_degree: int,
) -> dict:
    """Run the binding on ``_batch_edges`` and store its output in ``_renames``.

    Args:
        con: Connection with an ``_batch_edges`` relation and a
            ``_batch_components`` table that numbers its components.
        n_threads: Passed to the binding as ``n_threads``.
        bfs_distance: Passed to the binding as ``bfs_threshold``.
        hamming_threshold: Passed to the binding as ``hamming_threshold``.
        max_degree: Passed to the binding as ``max_degree``.

    Returns:
        The binding's stats for the batch.

    """
    import pixelator_core

    edges = con.execute("""
        SELECT e.umi1, e.umi2, c.component_id
        FROM _batch_edges e
        JOIN _batch_components c USING (component)
    """).fetchnumpy()
    renames, stats = pixelator_core.graph_collapse(
        edges["umi1"],
        edges["umi2"],
        edges["component_id"],
        n_threads,
        bfs_threshold=bfs_distance,
        hamming_threshold=hamming_threshold,
        max_degree=max_degree,
    )
    con.register("_renames_arrow", pa.table(renames))
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _renames AS
        SELECT c.component, r.original, r.corrected
        FROM _renames_arrow r
        JOIN _batch_components c ON c.component_id = r.component
    """)
    con.unregister("_renames_arrow")
    return stats


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


def _collapsed_sequences(
    con: duckdb.DuckDBPyConnection,
    umi1_length: int,
    umi2_length: int,
) -> list[dict]:
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


def run_graph_collapse(
    input_pxl_file: PxlFile,
    output_path: Path,
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
    batch_edges: int = DEFAULT_BATCH_EDGES,
    markers: list[str] | None = None,
) -> GraphCollapseResult:
    """Collapse the edgelist of ``input_pxl_file`` into a new pxl file at ``output_path``.

    Everything but the edgelist rows is copied from the input first. Its
    components are then processed in batches of about ``batch_edges`` edges, in
    component order: the binding finds the renames, pairs with different
    markers are dropped, and the renamed and merged edges are appended to the
    output edgelist. Finally the output's AnnData is rebuilt from the collapsed
    edgelist.

    Args:
        input_pxl_file: The pxl file to collapse. It is not modified.
        output_path: Where the collapsed pxl file is written. An existing file
            is replaced.
        n_threads: Passed to the binding as ``n_threads``, and DuckDB's thread count.
        bfs_distance: Passed to the binding as ``bfs_threshold``.
        hamming_threshold: Passed to the binding as ``hamming_threshold``.
        umi1_length: Nucleotide length used to decode a renamed UMI seen as umi1.
        umi2_length: Nucleotide length used to decode a renamed UMI seen only as umi2.
        max_degree: Passed to the binding as ``max_degree``. Nodes with a higher
            degree are not searched. The default, 1, investigates only leaves.
        report_collapsed_sequences: Decode each kept rename into nucleotide
            sequences. Off unless the CLI is writing the collapsed-sequence TSV.
        memory_limit: Soft memory limit in bytes for DuckDB. When set, DuckDB
            spills to disk once the limit is reached. The native search is not
            covered by it; its memory follows ``batch_edges``.
        temp_directory: DuckDB spill directory. Defaults to
            ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp``.
        batch_edges: Number of edges a batch of components is filled to.
        markers: The markers to count in the rebuilt AnnData, in ``var`` order.
            Pass the input's ``var_names`` when the caller has already read its
            AnnData; read from the input when ``None``.

    Returns:
        Counts and, when requested, the collapsed-sequence rows.

    """
    # Rebuild counts from the markers already in var. Using the full panel
    # would put markers removed by an earlier step (e.g. hashing clones
    # removed by sample calling) back. The panel comes from the file only.
    input_dataset = read(input_pxl_file.path)
    panel = PNAAntibodyPanel.from_pxl_dataset(input_dataset)
    if markers is None:
        markers = list(input_dataset.adata().var_names)

    stats_total: Counter = Counter()
    bfs_distances: Counter = Counter()
    hamming_distances: Counter = Counter()
    nodes_collapsed = 0
    sequences: list[dict] = []

    with PixelFileWriter(
        output_path,
        exits_ok=True,
        temp_dir=temp_directory,
        memory_limit=memory_limit,
        threads=n_threads,
    ) as writer:
        writer.copy_from(input_pxl_file.path, empty_tables=("edgelist",))
        con = writer.get_connection()
        con.execute(
            f"ATTACH {_sql_string(str(input_pxl_file.path))} AS {_SOURCE} (READ_ONLY)"
        )
        component_sizes = con.execute(f"""
            SELECT component, COUNT(*) FROM {_SOURCE}.edgelist
            GROUP BY component
            ORDER BY component
        """).fetchall()
        batches = plan_component_batches(component_sizes, batch_edges)
        logger.info(
            "Collapsing %d components in %d batches",
            len(component_sizes),
            len(batches),
        )

        for index, components in enumerate(batches, start=1):
            _select_batch(con, f"{_SOURCE}.edgelist", components)
            stats = find_renames(
                con,
                n_threads=n_threads,
                bfs_distance=bfs_distance,
                hamming_threshold=hamming_threshold,
                max_degree=max_degree,
            )
            stats_total.update({key: stats[key] for key in _STAT_KEYS})
            bfs_distances.update(stats["bfs_distance_distribution"])
            hamming_distances.update(stats["hamming_distance_distribution"])

            drop_renames_with_different_markers(con)
            # The binding maps each original UMI to one corrected UMI per
            # component, so every kept row is a distinct collapsed node.
            nodes_collapsed += _fetch_value(con, "SELECT COUNT(*) FROM _renames")
            if report_collapsed_sequences:
                sequences.extend(_collapsed_sequences(con, umi1_length, umi2_length))
            write_collapsed_edges(con)
            logger.info(
                "Batch %d/%d done: %d components processed",
                index,
                len(batches),
                stats_total["total_components"],
            )

        # The view reads from the input, so it goes before the input is detached.
        con.execute("DROP VIEW IF EXISTS _batch_edges")
        con.execute(f"DETACH {_SOURCE}")

        logger.info("Building AnnData from the collapsed edgelist")
        adata = pna_edgelist_to_anndata(con, panel, markers=markers)
        adata = adata[adata.obs_names.sort_values(), :]
        writer.write_adata(adata)

    sequences.sort(
        key=lambda record: (record["component"], record["original_sequence"])
    )
    return GraphCollapseResult(
        component_count=stats_total["total_components"],
        nodes_considered=stats_total["nodes_considered"],
        targets_found=stats_total["targets_found"],
        bfs_distance_distribution=dict(sorted(bfs_distances.items())),
        hamming_distance_distribution=dict(sorted(hamming_distances.items())),
        nodes_collapsed=nodes_collapsed,
        collapsed_sequences=sequences,
    )
