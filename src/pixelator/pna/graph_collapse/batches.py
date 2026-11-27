"""Batching of the input edgelist for graph-collapse, and access to the input file.

The input pxl file is attached read-only to the output's DuckDB connection under
``SOURCE``. Its components are grouped into batches of consecutive components, and
``select_batch`` exposes one batch at a time as the temp view ``_batch_edges``.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa

from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile

SOURCE = "collapse_input"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def attach_input(
    writer: PixelFileWriter, input_pxl_file: PxlFile
) -> duckdb.DuckDBPyConnection:
    """Attach the input and create an empty ``edgelist`` with its columns.

    The empty table keeps the input's column types, which the collapsed rows
    are appended to as they are.

    Args:
        writer: The writer of the output pxl file.
        input_pxl_file: The pxl file to collapse, attached read-only as ``SOURCE``.

    Returns:
        The writer's connection.

    """
    con = writer.get_connection()
    con.execute(
        f"ATTACH {_sql_string(str(input_pxl_file.path))} AS {SOURCE} (READ_ONLY)"
    )
    con.execute(f"CREATE TABLE edgelist AS SELECT * FROM {SOURCE}.edgelist LIMIT 0")
    return con


def component_sizes(con: duckdb.DuckDBPyConnection) -> list[tuple[str, int]]:
    """Return ``(component, number of edges)`` of the attached input, in component order."""
    return con.execute(f"""
        SELECT component, COUNT(*) FROM {SOURCE}.edgelist
        GROUP BY component
        ORDER BY component
    """).fetchall()


def select_batch(
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


def detach_input(con: duckdb.DuckDBPyConnection) -> None:
    """Drop ``_batch_edges`` and detach the input.

    The view reads from the input, so it has to go before the input is detached.
    """
    con.execute("DROP VIEW IF EXISTS _batch_edges")
    con.execute(f"DETACH {SOURCE}")
