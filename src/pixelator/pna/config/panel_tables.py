"""DuckDB storage for panel markers and their sources.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import duckdb

    from pixelator.pna.config.panel import PNAAntibodyPanel


_PANEL_TABLES = ("panels", "panel_sources")
_PANEL_STORAGE_COLUMNS = frozenset({"row_nr", "source_id", "marker_id"})


def panel_tables_present(connection: duckdb.DuckDBPyConnection) -> bool:
    """Return whether both panel tables exist on this connection."""
    names = set(connection.execute("SHOW TABLES").fetchdf()["name"].tolist())
    return set(_PANEL_TABLES).issubset(names)


def stored_panel_marker_columns(connection: duckdb.DuckDBPyConnection) -> set[str]:
    """Return marker columns stored in ``panels``.

    Empty when the panel tables are absent. ``row_nr``, ``source_id``, and
    ``marker_id`` are storage columns, not marker fields.
    """
    if not panel_tables_present(connection):
        return set()
    frame = connection.execute("SELECT * FROM panels LIMIT 0").fetchdf()
    return {str(name) for name in frame.columns if name not in _PANEL_STORAGE_COLUMNS}


def write_panel_tables(
    connection: duckdb.DuckDBPyConnection, panel: PNAAntibodyPanel
) -> None:
    """Replace ``panels`` and ``panel_sources`` with ``panel``."""
    if not panel.sources:
        raise ValueError("Cannot store a panel that has no sources.")

    source_rows = []
    for source_id, source in enumerate(panel.sources):
        metadata = source.metadata
        source_rows.append(
            {
                "source_id": source_id,
                "name": metadata.name,
                "version": metadata.version,
                "product": metadata.product,
                "description": metadata.description,
                "aliases": json.dumps(list(metadata.aliases)),
                "archived": bool(metadata.archived)
                if metadata.archived is not None
                else False,
                "file_name": source.file_name,
                "filepath": source.filepath,
            }
        )
    sources_df = pd.DataFrame(source_rows)

    markers = panel.df.copy()
    markers.insert(0, "marker_id", markers.index.astype(str))
    markers.insert(0, "row_nr", range(len(markers)))
    markers["source_id"] = (
        panel.marker_source_ids.reindex(panel.df.index).astype(int).to_numpy()
    )

    connection.register("_panel_sources_df", sources_df)
    connection.register("_panels_df", markers.reset_index(drop=True))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE panel_sources AS SELECT * FROM _panel_sources_df"
        )
        connection.execute("CREATE OR REPLACE TABLE panels AS SELECT * FROM _panels_df")
    finally:
        connection.unregister("_panel_sources_df")
        connection.unregister("_panels_df")


def read_panel_table_frames(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return ``(markers, sources)`` or None when the tables are absent."""
    if not panel_tables_present(connection):
        return None
    markers = connection.execute("SELECT * FROM panels ORDER BY row_nr").fetchdf()
    sources = connection.execute(
        "SELECT * FROM panel_sources ORDER BY source_id"
    ).fetchdf()
    return markers, sources
