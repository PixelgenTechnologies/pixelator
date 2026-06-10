"""Pxl file abstraction for pixeldatasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

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
        with duckdb.connect(self.path, read_only=True) as con:
            tables = con.sql("SHOW ALL TABLES").to_df()
            return len(
                set(PXL_FILE_MANDATOR_TABLES).intersection(
                    set(tables["name"].to_list())
                )
            ) == len(PXL_FILE_MANDATOR_TABLES)

    def metadata(self) -> dict:
        """Read the metadata from the PXL file."""
        try:
            with duckdb.connect(self.path, read_only=True) as con:
                metadata = con.sql("SELECT * FROM metadata").fetchone()
                return json.loads(metadata[0]) if metadata else {}
        except duckdb.CatalogException:
            return {}

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
