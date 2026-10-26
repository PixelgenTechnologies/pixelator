"""Read the antibody panel of a pxl file that stores it in ``uns`` and ``var``.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json

import duckdb
from anndata import AnnData

from pixelator.pna.config.panel import PNAAntibodyPanel


class LegacyPanelAdapter:
    """Rebuild a panel from ``uns['panel_metadata']`` and the panel columns on ``var``.

    Files from pixelator 0.22 and later keep the panel there, instead of in
    tables of their own.
    """

    def __init__(
        self, connection: duckdb.DuckDBPyConnection, file_name: str | None = None
    ) -> None:
        """Wrap an open connection to a pxl file.

        Args:
            connection: Connection to the pxl file.
            file_name: Basename recorded on the rebuilt panel.
        """
        self._connection = connection
        self._file_name = file_name

    def read(self) -> PNAAntibodyPanel | None:
        """Return the panel stored in ``uns`` and ``var``, or None when there is none."""
        try:
            uns_row = self._connection.execute(
                "SELECT value FROM __adata__uns"
            ).fetchone()
        except duckdb.CatalogException:
            return None
        if uns_row is None:
            return None
        uns = json.loads(uns_row[0]) if isinstance(uns_row[0], str) else uns_row[0]
        if not isinstance(uns, dict) or "panel_metadata" not in uns:
            return None

        var = self._connection.execute("SELECT * FROM __adata__var").fetchdf()
        var = var.set_index("index").rename_axis(index={"index": "marker_id"})
        try:
            return PNAAntibodyPanel.from_adata(
                AnnData(var=var, uns=uns), file_name=self._file_name
            )
        except KeyError:
            # panel_metadata without the panel columns
            return None
