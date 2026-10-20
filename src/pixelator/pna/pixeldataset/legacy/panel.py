"""Rebuild the panel of a pxl file written before the panel tables existed.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import duckdb
from anndata import AnnData

from pixelator.common.utils import logger
from pixelator.pna.config.panel import PNAAntibodyPanel

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig


class LegacyPanelAdapter:
    """Read the panel of a pxl file that has no ``panels`` or ``panel_sources`` table.

    Two older layouts are supported, tried in this order:

    1. Pixelator 0.22 through 0.30 stored ``uns['panel_metadata']`` and the
       panel columns on ``var``. The panel is rebuilt from those.
    2. Earlier files only record ``panel_name`` in the file metadata. The panel
       is looked up by that name in the pixelator config. The recorded
       ``panel_version`` is not trusted, so the name has to resolve to a single
       panel on its own.

    A file with neither yields None.
    """

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        file_name: str | None = None,
        config: PNAConfig | None = None,
    ) -> None:
        """Wrap an open connection to a pxl file.

        Args:
            connection: Connection to the pxl file.
            file_name: Basename recorded on a panel rebuilt from ``uns`` and ``var``.
            config: Config used to look up panels by name. Defaults to the
                installed pixelator config.
        """
        self._connection = connection
        self._file_name = file_name
        self._config = config

    def read(self) -> PNAAntibodyPanel | None:
        """Return the panel this file stores in an older layout, or None."""
        panel = self._from_uns_and_var()
        if panel is None:
            panel = self._from_panel_name()
        return panel

    def _from_uns_and_var(self) -> PNAAntibodyPanel | None:
        uns = self._read_json("SELECT value FROM __adata__uns")
        if not isinstance(uns, dict) or "panel_metadata" not in uns:
            return None
        var = self._connection.execute("SELECT * FROM __adata__var").fetchdf()
        var = var.set_index("index").rename_axis(index={"index": "marker_id"})
        try:
            return PNAAntibodyPanel.from_adata(
                AnnData(var=var, uns=uns), file_name=self._file_name
            )
        except KeyError:
            # panel_metadata without panel_columns: fall back to the panel name
            return None

    def _from_panel_name(self) -> PNAAntibodyPanel | None:
        metadata = self._read_json("SELECT value FROM metadata")
        panel_name = metadata.get("panel_name") if isinstance(metadata, dict) else None
        if not panel_name:
            return None
        if self._config is None:
            from pixelator.pna.config import pna_config

            self._config = pna_config
        panel = self._config.get_panel(panel_name)
        if panel is None:
            logger.warning(
                "The panel %s recorded in the file is not in the pixelator config.",
                panel_name,
            )
            return None
        logger.warning(
            "The file does not store its panel. Using the panel %s %s from the "
            "pixelator config.",
            panel.name,
            panel.version,
        )
        return panel

    def _read_json(self, query: str):
        """Return the JSON value in the first row of ``query``, or None."""
        try:
            row = self._connection.execute(query).fetchone()
        except duckdb.CatalogException:
            return None
        if row is None:
            return None
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]
