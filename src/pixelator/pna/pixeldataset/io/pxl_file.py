"""Pxl file abstraction for pixeldatasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.pna.config.panel import PNAAntibodyPanel, aligned_dataset_panel
from pixelator.pna.config.panel_tables import read_panel_table_frames

if TYPE_CHECKING:
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset

PXL_FILE_MANDATOR_TABLES = [
    "__adata__X",
    "__adata__var",
    "__adata__obs",
    "edgelist",
]

# Should this be a "metadata" be a mandatory table?
PXL_FILE_ADATA_TABLES = ["__adata__X", "__adata__var", "__adata__obs", "__adata__uns"]
PXL_FILE_OTHER_TABLES = ["edgelist", "metadata", "layouts", "proximity"]


class PxlFile:
    """PxlFile represents a a pxl file on disk and provides basic utility methods."""

    def __init__(self, path: Path, sample_name: str | None = None):
        """Initialize the PxlFile."""
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        self.path = path
        self._sample_name = sample_name

    @property
    def sample_name(self) -> str:
        """Return the sample name of the PxlFile."""
        if self._sample_name:
            return self._sample_name
        try:
            return self.metadata()["sample_name"]
        except KeyError:
            raise ValueError(
                f"Could not determine sample name from {self.path} - please provide a sample name."
            )

    def is_pxl_file(self) -> bool:
        """Check if the file is a PXL file."""
        with connect_duckdb(self.path, read_only=True) as con:
            tables = con.sql("SHOW ALL TABLES").to_df()
            return len(
                set(PXL_FILE_MANDATOR_TABLES).intersection(
                    set(tables["name"].to_list())
                )
            ) == len(PXL_FILE_MANDATOR_TABLES)

    def metadata(self) -> dict:
        """Read the metadata from the PXL file."""
        try:
            with connect_duckdb(self.path, read_only=True) as con:
                metadata = con.sql("SELECT * FROM metadata").fetchone()
                return json.loads(metadata[0]) if metadata else {}
        except duckdb.CatalogException:
            return {}

    def read_panel(self) -> PNAAntibodyPanel | None:
        """Load the panel stored in this file.

        New files are read from the ``panels`` and ``panel_sources`` tables.
        Files from pixelator 0.22.0 through 0.30.0 are read from
        ``uns['panel_metadata']`` and the panel columns stored on ``var``.
        Returns None when the file has neither.
        """
        with connect_duckdb(self.path, read_only=True) as connection:
            frames = read_panel_table_frames(connection)
            if frames is not None:
                return PNAAntibodyPanel._from_panel_tables(*frames)
            try:
                uns_row = connection.execute(
                    "SELECT value FROM __adata__uns"
                ).fetchone()
            except duckdb.CatalogException:
                return None
            if uns_row is None:
                return None
            uns = json.loads(uns_row[0]) if isinstance(uns_row[0], str) else uns_row[0]
            if not isinstance(uns, dict) or "panel_metadata" not in uns:
                return None
            var = connection.execute("SELECT * FROM __adata__var").fetchdf()
            var = var.set_index("index").rename_axis(index={"index": "marker_id"})
            return PNAAntibodyPanel.from_legacy_var(
                var, uns["panel_metadata"], file_name=self.path.name
            )

    def __repr__(self) -> str:
        """Return a string representation of the PxlFile."""
        return f"PxlFile({self.path})"

    def __str__(self) -> str:
        """Return a string representation of the PxlFile."""
        return f"{self.path}"

    @staticmethod
    def copy_pxl_file(src: PxlFile, target: Path) -> PxlFile:
        """Copy a PxlFile to a new location.

        Args:
            src: The source PxlFile.
            target: The target path.

        Returns:
            The new PxlFile.
        """
        shutil.copy(src.path, target)
        return PxlFile(target)


def read_dataset_panel(dataset: PNAPixelDataset) -> PNAAntibodyPanel:
    """Load one panel from every file in ``dataset``.

    One file is returned as that panel. Several files are aligned to the
    newest patch of each source they share.

    Raises:
        KeyError: If a file has no panel. Pixelator 0.22.0 through 0.30.0
            stored it in ``uns['panel_metadata']``. Later files store it in
            the ``panels`` and ``panel_sources`` tables. Earlier files have
            neither.
    """
    panels = []
    for path in dataset.view.sample_to_file_mappings.values():
        panel = PxlFile(path).read_panel()
        if panel is None:
            raise KeyError(
                f"{path} has no panel. Pixelator 0.22.0 through 0.30.0 "
                "stored it in uns['panel_metadata']. Later files store it "
                "in the panels and panel_sources tables. Earlier files "
                "have neither."
            )
        panels.append(panel)
    if len(panels) == 1:
        return panels[0]
    return aligned_dataset_panel(panels)
