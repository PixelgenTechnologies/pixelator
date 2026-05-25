"""Writer utilities for writing PXL-backed pixeldatasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import polars as pl
from anndata import AnnData


class PixelFileWriter:
    """Writer class for PXL files.

    This class takes care of writing data to duckdb based pxl files.
    In addition to writing data tables it will should take care of
    creating any indexes that can be used to speed up queries downstream.

    When writing using this it is recommended to use the writer as a
    context manager.

    .. code-block:: python

        from pixelator.pna.pixeldataset.io import PixelFileWriter

        with PixelFileWriter("my_file.pxl") as writer:
            writer.write_edgelist(Path("edgelist.parquet"))
    """

    def __init__(self, path: Path, exits_ok: bool = False):
        """Initialize the PixelFileWriter.

        Args:
        path: The path to the PXL file.
        exists_ok: Whether to remove the file if it exists.
        exits_ok: Exits ok.
        """
        self.path = path
        self.exists_ok = exits_ok
        if self.exists_ok and self.path.exists():
            self.path.unlink()
        self._connection: duckdb.DuckDBPyConnection = None  # type: ignore

    def open(self):
        """Open a connection to the PXL file."""
        self._connection = duckdb.connect(self.path)

    def close(self):
        """Close the connection to the PXL file."""
        self._connection.close()
        self._connection = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get the connection to the PXL file.

        Returns:
        The duckdb connection.

        """
        if self._connection is None:
            raise ValueError("Connection is not open.")
        return self._connection

    def __enter__(self):
        """Open the writer as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the writer context manager.

        Args:
        exc_type: Exc type.
        exc_value: Exc value.
        traceback: Traceback.
        """
        self.close()

    def _write_parquet_file_to_table(
        self, table_name, edgelist_file: Path | list[Path]
    ):
        self._connection.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_parquet($parquet_file);
            """,
            params={
                "parquet_file": [str(f) for f in edgelist_file]
                if isinstance(edgelist_file, list)
                else str(edgelist_file)
            },
        )

    def write_edgelist(self, edgelist: Path | pl.DataFrame) -> None:
        """Write the edgelist to the PXL file.

        Args:
        edgelist: The path to the edgelist parquet file.
        """
        if isinstance(edgelist, Path):
            self._write_parquet_file_to_table("edgelist", edgelist)
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
                file_path = Path(f.name)
                edgelist.write_parquet(file_path)
                self._write_parquet_file_to_table("edgelist", file_path)
        except AttributeError:
            raise ValueError(f"Invalid input type for edgelist: {type(edgelist)}")

    def _clean_existing_adata_tables(self):
        tables = self._connection.sql("SHOW ALL TABLES")
        adata_tables = tables.pl().filter((pl.col("name").str.starts_with("__adata__")))
        for table in adata_tables.iter_rows(named=True):
            self._connection.sql(
                f'DROP TABLE "{table["database"]}"."{table["schema"]}"."{table["name"]}"'
            )

    def write_adata(self, adata: AnnData) -> None:
        """Write the AnnData object to the PXL file.

        Args:
        adata: The AnnData object to write.
        """
        self._clean_existing_adata_tables()

        X = adata.to_df().reset_index(names="index")
        var = adata.var.reset_index(names="index")
        obs = adata.obs.reset_index(names="index")
        uns = adata.uns

        self._connection.sql(
            """
            CREATE TABLE __adata__X AS SELECT * FROM X;
            CREATE TABLE __adata__var AS SELECT * FROM var;
            CREATE TABLE __adata__obs AS SELECT * FROM obs;

            CREATE TABLE __adata__uns (value JSON);
            INSERT INTO __adata__uns VALUES ($uns_data);
            """,
            params={"uns_data": uns},
        )

        for key in adata.obsm:
            obsm_layer = adata.obsm[key].reset_index(names="index")
            self._connection.sql(
                f"""
                CREATE TABLE __adata__obsm_{key} AS SELECT * FROM obsm_layer;
                """,
            )

    def write_metadata(self, metadata: dict) -> None:
        """Write the metadata to the PXL file.

        Args:
        metadata: The metadata dictionary to write.
        """
        self._connection.sql(
            """
            DROP TABLE IF EXISTS metadata;
            CREATE TABLE metadata (value JSON);
            INSERT INTO metadata VALUES ($metadata);
            """,
            params={"metadata": metadata},
        )

    def write_layouts(self, layouts: Path | pl.DataFrame | list[Path]) -> None:
        """Write the layouts to the PXL file.

        Args:
        layouts: The path to the layouts parquet file.
        """
        try:
            if isinstance(layouts, list) or layouts.is_file():  # type: ignore
                self._write_parquet_file_to_table("layouts", layouts)  # type: ignore
                return
        except AttributeError:
            pass

        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
                tmp_path = Path(f)
                layouts.write_parquet(tmp_path)  # type: ignore
                self._write_parquet_file_to_table("layouts", tmp_path)
        except AttributeError:
            raise ValueError(f"Invalid input type for layouts: {type(layouts)}")

    def write_proximity(self, proximity: Path | pl.DataFrame) -> None:
        """Write the proximity data to the PXL file.

        Args:
        proximity: The path to the proximity parquet file.
        """
        try:
            if proximity.is_file():  # type: ignore
                self._write_parquet_file_to_table("proximity", proximity)  # type: ignore
                return
        except AttributeError:
            pass

        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
                tmp_path = Path(f.name)
                proximity.write_parquet(tmp_path)  # type: ignore
                self._write_parquet_file_to_table("proximity", tmp_path)
        except AttributeError:
            raise ValueError(f"Invalid input type for proximity: {type(proximity)}")
