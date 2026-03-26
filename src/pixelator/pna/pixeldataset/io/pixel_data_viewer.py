"""Virtual SQL views over multiple PXL files.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

from .pxl_file import PXL_FILE_MANDATOR_TABLES, PXL_FILE_OTHER_TABLES, PxlFile
from .query_builder import Query


class PixelDataViewer:
    """PixelDataViewer is used to create virtual views over multiple pxl files.

    The `PixelDataViewer` provides a unified interface over multiple pxl files that
    can be used to query the data in the files. It does this by creating SQL views over all
    the tables in memory. Nota bene that this does not actually materialize any data
    in memory, so the operation is very light weight.

    PixelDataViewer is only responsible for providing the SQL view and
    executing SQL queries against it.
    """

    def __init__(
        self,
        sample_name_to_pxl_file_mapping: dict[str, PxlFile],
    ):
        """Initialize the PixelDataViewer.

        :param sample_name_to_pxl_file_mapping: A dictionary mapping sample names to PxlFile objects.
        :raises ValueError: If any of the files are not valid PXL files.
        """
        self._db_to_file_mapping = sample_name_to_pxl_file_mapping
        self._normalized_sample_name_mapping = self._map_sample_names_to_db_names(
            sample_name_to_pxl_file_mapping
        )
        # verify all files are PXL files
        invalid_files = [
            pxl_file
            for pxl_file in self._db_to_file_mapping.values()
            if not pxl_file.is_pxl_file()
        ]
        if invalid_files:
            raise ValueError(f"{invalid_files} are not valid PXL files.")

        self._connection: duckdb.DuckDBPyConnection = None  # type: ignore

    def _map_sample_names_to_db_names(
        self, sample_name_to_pxl_file_mapping: dict[str, PxlFile]
    ) -> dict[str, str]:
        """Create a normalized sample name lookup table for working with the duckdb files."""

        def _normalize_sample_name(name: str) -> str:
            # Remove whitespaces and dashes from the sample name
            # and prefix it with "db_", and prefix all names
            # with "db_" to handle when sample names starts with a number.
            return f"db_{name.replace('-', '_').replace(' ', '_')}"

        normalized_names = {
            sample_name: _normalize_sample_name(sample_name)
            for sample_name, _ in sample_name_to_pxl_file_mapping.items()
        }

        if len(normalized_names) != len(sample_name_to_pxl_file_mapping.keys()):
            raise ValueError(
                "Sample names are not unique after normalizing them - please provide sample names that are unique "
                "even when replacing `-` and whitespaces with `_`."
            )

        return normalized_names

    def _get_normalized_name(self, sample_name: str) -> str:
        """Get the normalized name for a sample name."""
        try:
            return self._normalized_sample_name_mapping[sample_name]
        except KeyError:
            raise KeyError(
                f"Sample name {sample_name} not found in the mapping. Available names are: {set(self._normalized_sample_name_mapping.keys())}"
            )

    @staticmethod
    def from_files(pxl_files: Iterable[PxlFile]) -> "PixelDataViewer":
        """Create a PixelDataViewer from a list of PxlFile objects.

        This will use the sample names from the PxlFile metadata as sample names.
        """
        return PixelDataViewer(
            {pxl_file.sample_name: pxl_file for pxl_file in pxl_files}
        )

    @staticmethod
    def from_sample_to_file_mappings(
        sample_name_to_pxl_file_mapping: dict[str, PxlFile],
    ) -> "PixelDataViewer":
        """Create a PixelDataViewer from a dictionary mapping sample names to PxlFile objects."""
        return PixelDataViewer(sample_name_to_pxl_file_mapping)

    def filter_samples(self, sample_names: set[str]) -> "PixelDataViewer":
        """Create a new PixelDataViewer with only a subset of the samples."""
        filtered_mapping = {
            sample_name: pxl_file
            for sample_name, pxl_file in self._db_to_file_mapping.items()
            if sample_name in sample_names
        }
        return PixelDataViewer.from_sample_to_file_mappings(filtered_mapping)

    @property
    def sample_to_file_mappings(self) -> dict[str, Path]:
        """Return a dictionary mapping sample names to PxlFile objects."""
        return {
            sample_name: file_.path
            for sample_name, file_ in self._db_to_file_mapping.items()
        }

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        """Open the connection to the PXL files.

        Attach all the files to the connection and create views over the tables.
        :return: The connection object.
        """
        self._connection = duckdb.connect(":memory:")
        self._attach_to_files(self._connection)
        for table in PXL_FILE_OTHER_TABLES:
            self._simple_union_table_view(self._connection, table, table)
        return self._connection

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the connection to the PXL files."""
        self._connection.close()

    def execute_lazy(
        self, connection: duckdb.DuckDBPyConnection, query: Query
    ) -> pl.LazyFrame:
        """Execute a query and return a Polars LazyFrame."""
        return connection.execute(query.sql, parameters=query.params).pl(lazy=True)

    def execute_eager(
        self, connection: duckdb.DuckDBPyConnection, query: Query
    ) -> pl.DataFrame:
        """Execute a query and return a Polars DataFrame."""
        return connection.execute(query.sql, parameters=query.params).pl()

    def execute_scalar(
        self, connection: duckdb.DuckDBPyConnection, query: Query
    ) -> int:
        """Execute a scalar query and return first value."""
        return connection.sql(query.sql, params=query.params).execute().fetchone()[0]  # type: ignore

    def execute_arrow_reader(
        self,
        connection: duckdb.DuckDBPyConnection,
        query: Query,
        batch_size: int,
    ):
        """Execute a query and return Arrow reader."""
        result = connection.sql(query.sql, params=query.params)
        return result.fetch_arrow_reader(batch_size=batch_size)

    def sample_names(self) -> list[str]:
        """Return the list of sample names known to the view."""
        return list(self._db_to_file_mapping.keys())

    def normalized_sample_db_name(self, sample_name: str) -> str:
        """Return the attached DuckDB database name for a sample."""
        return self._get_normalized_name(sample_name)

    def _attach_to_files(self, connection: duckdb.DuckDBPyConnection):
        query = ""
        for name, path in self._db_to_file_mapping.items():
            query += f"ATTACH DATABASE '{path}' AS {self._get_normalized_name(name)} (READ_ONLY);\n"
        connection.execute(query)

    def _simple_union_table_view(
        self, connection: duckdb.DuckDBPyConnection, table_name: str, view_name: str
    ):
        """Create a view that unions given table from all the underlying samples."""
        # This will create a view that unions the tables from all the samples,
        # adding a new column that identifies the sample.
        try:
            view_query = [f"""CREATE VIEW {view_name} AS"""]

            table_queries = []
            for sample_name, _ in self._db_to_file_mapping.items():
                table_queries.append(
                    f"SELECT *, '{sample_name}' as 'sample' FROM {self._get_normalized_name(sample_name)}.{table_name}"
                )
            union_query = "\n UNION ALL \n".join(table_queries)

            view_query.append(union_query)
            view_query_str = "\n".join(view_query)

            connection.execute(view_query_str)
        except duckdb.CatalogException:
            if table_name in PXL_FILE_MANDATOR_TABLES:
                raise ValueError(
                    f"Mandatory table {table_name} is missing from {self._db_to_file_mapping[sample_name]}- are you sure this is a pxl file?"
                )
            # note that we are ignoring the exception here, since it is expected
            # that some tables may not be present in all files.
            pass
