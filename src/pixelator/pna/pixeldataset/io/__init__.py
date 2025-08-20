"""PNA Pixel File io.

This module handles writing and reading duckdb based pxl files.

Duckdb is an embedded database that supports storing multiple
tables in one portable file.

Pxl files should contain an edgelist table that contains
the edgelist. Futhermore it should contain a number of tables
that encode the adata object. All these table should be prefixed
with `__adata__`, e.g. the adata.obs table should be named
`__adata__obs`.

In addition to the adata tables, and the edgelist, the database
can contain a metadata table, a layouts table, and a proximity
data table.

A full pxl file has (at least) the following tables:

.. code-block:: none

    ┌─────────────────────┐
    │ name                │
    │ ---                 │
    │ str                 │
    ╞═════════════════════╡
    │ __adata__X          │
    │ __adata__obs        │
    │ __adata__var        │
    │ edgelist            │
    │ layouts             │
    │ metadata            │
    │ proximity           │
    └─────────────────────┘

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from functools import cache
from pathlib import Path
from typing import Iterable, Literal, Sized

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
from anndata import AnnData
from anndata import concat as anndata_concat

from pixelator.mpx.pixeldataset.utils import update_metrics_anndata
from pixelator.pna.utils.utils import normalize_input_to_list

logger = logging.getLogger(__name__)


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

        :param path: The path to the PXL file.
        :param exists_ok: Whether to remove the file if it exists.
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

    def __enter__(self):
        """Open the writer as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the writer context manager."""
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

        :param edgelist: The path to the edgelist parquet file.
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

        :param adata: The AnnData object to write.
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

        for key in adata.obsm_keys():
            obsm_layer = adata.obsm[key].reset_index(names="index")
            self._connection.sql(
                f"""
                CREATE TABLE __adata__obsm_{key} AS SELECT * FROM obsm_layer;
                """,
            )

    def write_metadata(self, metadata: dict) -> None:
        """Write the metadata to the PXL file.

        :param metadata: The metadata dictionary to write.
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

        :param layouts: The path to the layouts parquet file.
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

        :param proximity: The path to the proximity parquet file.
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

        :param src: The source PxlFile.
        :param target: The target path.
        :return: The new PxlFile.
        """
        shutil.copy(src.path, target)
        return PxlFile(target)


class PixelDataViewer:
    """PixelDataViewer is used to create virtual views over multiple pxl files.

    The `PixelDataViewer` provides a unified interface over multiple pxl files that
    can be used to query the data in the files. It does this by creating SQL views over all
    the tables in memory. Nota bene that this does not actually materialize any data
    in memory, so the operation it self is very light weight.

    In addition to providing SQL viewes to run queries over, PixelDataViwer class will
    read and concatenate the adata objects from all the files, and cache the resulting
    AnnData object.
    """

    def __init__(
        self,
        sample_name_to_pxl_file_mapping: dict[str, PxlFile],
        adata_join_strategy: Literal["inner", "outer"] = "inner",
    ):
        """Initialize the PixelDataViewer.

        :param sample_name_to_pxl_file_mapping: A dictionary mapping sample names to PxlFile objects.
        :param adata_join_strategy: The strategy to use when joining the adata objects.
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

        self._adata_join_strategy = adata_join_strategy
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
    def from_files(pxl_files: Iterable[PxlFile]) -> PixelDataViewer:
        """Create a PixelDataViewer from a list of PxlFile objects.

        This will use the sample names from the PxlFile metadata as sample names.
        """
        return PixelDataViewer(
            {pxl_file.sample_name: pxl_file for pxl_file in pxl_files}
        )

    @staticmethod
    def from_sample_to_file_mappings(
        sample_name_to_pxl_file_mapping: dict[str, PxlFile],
    ) -> PixelDataViewer:
        """Create a PixelDataViewer from a dictionary mapping sample names to PxlFile objects."""
        return PixelDataViewer(sample_name_to_pxl_file_mapping)

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

    def sample_names(self) -> list[str]:
        """Return the list of sample names known to the view."""
        return list(self._db_to_file_mapping.keys())

    def read_adata_from_sample(self, sample: str) -> AnnData:
        """Read the AnnData object from the PXL file.

        :return: The AnnData object.
        """
        with self as connection:
            X = connection.sql(
                f"SELECT * FROM {self._get_normalized_name(sample)}.__adata__X"
            ).to_df()
            var = connection.sql(
                f"SELECT * FROM {self._get_normalized_name(sample)}.__adata__var"
            ).to_df()
            obs = connection.sql(
                f"SELECT * FROM {self._get_normalized_name(sample)}.__adata__obs"
            ).to_df()

            maybe_uns = connection.sql(
                f"select * from {self._get_normalized_name(sample)}.__adata__uns"
            ).fetchone()
            uns = json.loads(maybe_uns[0]) if maybe_uns else None

            tables = connection.sql("SHOW ALL TABLES")

            obsm_tables = (
                tables.pl()
                .filter(
                    (pl.col("name").str.starts_with("__adata__obsm"))
                    & (pl.col("database") == self._get_normalized_name(sample))
                )
                .select(
                    pl.concat_str(
                        [pl.col("database"), pl.col("schema"), pl.col("name")],
                        separator=".",
                    ).alias("name")
                )
                .get_column("name")
                .to_list()
            )

            obsm = {
                table.split("__adata__obsm_")[1]: (
                    connection.sql(f"SELECT * FROM {table}")
                    .to_df()
                    .set_index("index")
                    .rename_axis(index={"index": "component"})
                )
                for table in obsm_tables
            }

            adata = AnnData(
                X=X.set_index("index").rename_axis(index={"index": "component"}),
                var=var.set_index("index").rename_axis(index={"index": None}),
                obs=obs.set_index("index").rename_axis(index={"index": "component"}),
                uns=uns,
                obsm=obsm,
            )
            return adata

    @cache
    def read_adata(self) -> AnnData:
        """Read the AnnData object from the PXL file.

        The result will be cached to avoid reading and concatenating
        the anndata multiple times.

        :return: The AnnData object
        """

        def underlying_data():
            for sample_name, _ in self._db_to_file_mapping.items():
                adata = self.read_adata_from_sample(sample_name)
                adata.obs["sample"] = sample_name
                yield adata

        adatas = list(underlying_data())
        concatenated = anndata_concat(adatas, join=self._adata_join_strategy)
        concatenated.var = adatas[0].var
        update_metrics_anndata(concatenated, inplace=True)
        return concatenated

    def _attach_to_files(self, connection: duckdb.DuckDBPyConnection):
        for name, path in self._db_to_file_mapping.items():
            query = f"ATTACH DATABASE '{path}' AS {self._get_normalized_name(name)} (READ_ONLY);"
            connection.execute(query)

    def _simple_union_table_view(
        self, connection: duckdb.DuckDBPyConnection, table_name: str, view_name: str
    ):
        """Create a view that unions given table from all the underlying samples."""
        # This will create a view tha unions the tables from all the samples,
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
        except duckdb.CatalogException as e:
            if table_name in PXL_FILE_MANDATOR_TABLES:
                raise ValueError(
                    f"Mandatory table {table_name} is missing from {self._db_to_file_mapping[sample_name]}- are you sure this is a pxl file?"
                )
            # note that we are ignoring the exception here, since it is expected
            # that some tables may not be present in all files.
            pass


class PixelDataQuerier:
    """Class to read data from a PXL view - this view an represent one or more underlying files on disk."""

    def __init__(self, view: PixelDataViewer):
        """Initialize the PixelDataQuerier."""
        self.view = view

    def read_adata(self) -> AnnData:
        """Read the AnnData object from the PXL file.

        :return: The AnnData object.
        """
        return self.view.read_adata()

    def read_all_component_names(self) -> set[str]:
        """Read all component names from the PXL file.

        :return: A set of component names.
        """
        return set(self.view.read_adata().obs.index.to_list())

    def read_all_marker_names(self) -> set[str]:
        """Read all marker names from the PXL file.

        :return: A set of component names.
        """
        return set(self.view.read_adata().var.index.to_list())

    def _optimized_component_where_condition(self, components: Sized | None) -> str:
        """Create an optimized where clause depending on the number of components.

        Since duckdb does not support predicate pushdown for IN clauses, we need to
        optimize the query by using an equal statement to speed things up when
        selecting a single component.

        This may change in later version of duckdb, see conversation here:
        https://discord.com/channels/909674491309850675/1032659480539824208/1336979122512986144
        """
        if not components:
            return "TRUE"
        if len(components) == 1:
            return f"component = $components"
        return f"component IN $components"

    def read_edgelist(
        self,
        components: Iterable[str] | str | None = None,
        as_pandas=False,
    ) -> pl.DataFrame | pd.DataFrame:
        """Read the edgelist from the PXL file.

        :param components: The components to filter by.
        :return: A DataFrame containing the edgelist.
        """
        with self.view as connection:
            components = normalize_input_to_list(components)
            query = f"""SELECT * FROM edgelist
                        WHERE {self._optimized_component_where_condition(components)}
                    """
            params = {}
            if components is not None:
                params["components"] = (
                    components if len(components) > 1 else components[0]
                )
            if as_pandas:
                return connection.sql(query, params=params).df()
            return connection.sql(query, params=params).pl()

    def read_edgelist_len(
        self,
        components: Iterable[str] | str | None = None,
    ) -> int:
        """Get the length of the edgelist.

        :param components: The components to filter by.
        :return: The length of the edgelist.
        """
        components = normalize_input_to_list(components)

        query = f"""SELECT COUNT(*) FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
        """
        params = {}
        if components is not None:
            params["components"] = components if len(components) > 1 else components[0]
        with self.view as connection:
            return connection.sql(query, params=params).execute().fetchone()[0]  # type: ignore

    def read_edgelist_stream(
        self,
        components: Iterable[str] | str | None = None,
        batch_size: int = 1_000_000,
    ) -> Iterable[pa.RecordBatch]:
        """Stream the edgelist from the PXL file.

        :param components: The components to filter by.
        :param batch_size: The batch size for streaming.
        :return: An iterable of RecordBatches.
        """
        components = normalize_input_to_list(components)

        query = f"""SELECT * FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
                """
        params = {}
        if components is not None:
            params["components"] = components if len(components) > 1 else components[0]

        with self.view as connection:
            result = connection.sql(query, params=params)
            reader = result.fetch_arrow_reader(batch_size=batch_size)
            for batch in reader:
                yield batch

    def read_metadata(self) -> dict:
        """Read the metadata from the PXL file.

        :return: The metadata dictionary.
        """
        with self.view as connection:
            maybe_metadata = list(
                map(
                    lambda x: json.loads(x[0]),
                    connection.sql("SELECT * FROM metadata").fetchall(),
                )
            )
            if not maybe_metadata:
                return {}

            metadata = {}
            for metadata_dict in maybe_metadata:
                metadata[metadata_dict["sample_name"]] = metadata_dict
        return metadata

    def read_layouts(
        self,
        components: str | Iterable[str] | None = None,
        add_marker_counts: bool = False,
    ) -> pl.DataFrame:
        """Read the layouts from the PXL file.

        :param components: The components to filter by.
        :param add_marker_counts: Whether to add marker counts.
        :return: A DataFrame containing the layouts.
        """

        def _pivot_marker_table(df):
            return (
                df.select(pl.col("*"), val=pl.lit(1))
                .pivot(
                    on="marker",
                    index=None,
                    values="val",
                    aggregate_function=pl.len().cast(pl.UInt8),
                )
                .fill_null(0)
            )

        components = normalize_input_to_list(components)

        if add_marker_counts:
            query = f"""
                WITH filtered_edgelist AS (
                    SELECT umi1 as umi, marker_1 as marker
                    FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}

                    UNION

                    SELECT umi2 as umi, marker_2 as marker
                    FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
                )
                SELECT *
                FROM layouts
                LEFT JOIN filtered_edgelist as umi_and_markers
                ON layouts.index = umi_and_markers.umi
                WHERE {self._optimized_component_where_condition(components)}
            """
        else:
            query = f"""SELECT * FROM layouts
                        WHERE {self._optimized_component_where_condition(components)}
                    """

        try:
            params = {}
            if components is not None:
                params["components"] = (
                    components if len(components) > 1 else components[0]
                )
            with self.view as connection:
                result = connection.sql(query, params=params).pl()
                result = _pivot_marker_table(result) if add_marker_counts else result
                return result.drop(["umi", "marker"], strict=False)
        except duckdb.CatalogException:
            return pl.DataFrame()

    def read_layouts_len(
        self,
        components: str | Iterable[str] | None = None,
    ) -> int:
        """Get the length of the layouts.

        :param components: The components to filter by.
        :return: The length of the layouts.
        """
        components = normalize_input_to_list(components)  # type: ignore
        query = f"""SELECT COUNT(*)
                    FROM layouts
                    WHERE {self._optimized_component_where_condition(components)}
                """
        try:
            params = {}
            if components is not None:
                params["components"] = (
                    components if len(components) > 1 else components[0]  # type: ignore
                )
            with self.view as connection:
                return connection.sql(query, params=params).execute().fetchone()[0]  # type: ignore
        except duckdb.CatalogException:
            return 0

    def read_proximity(
        self,
        components: str | Iterable[str] | None = None,
        markers: str | Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Read the proximity data from the PXL file.

        :param components: The components to filter by.
        :param markers: The markers to filter by.
        :return: A DataFrame containing the proximity data.
        """
        components = normalize_input_to_list(components)  # type: ignore
        markers = normalize_input_to_list(markers)  # type: ignore
        try:
            query = f"""SELECT * FROM proximity
                         WHERE  {self._optimized_component_where_condition(components)} AND
                                {"(marker_1 IN $markers AND marker_2 IN $markers)" if markers else "TRUE"};
                    """
            params = {}
            if components is not None:
                params["components"] = (
                    components if len(components) > 1 else components[0]  # type: ignore
                )
            if markers is not None:
                params["markers"] = markers
            with self.view as connection:
                return connection.sql(query, params=params).pl()
        except duckdb.CatalogException:
            return pl.DataFrame()

    def read_proximity_len(
        self,
        components: str | Iterable[str] | None = None,
        markers: str | Iterable[str] | None = None,
    ) -> int:
        """Get the length of the proximity data.

        :param components: The components to filter by.
        :param markers: The markers to filter by.
        :return: The length of the proximity data.
        """
        components = normalize_input_to_list(components)  # type: ignore
        markers = normalize_input_to_list(markers)  # type: ignore
        try:
            query = f"""SELECT COUNT(*) FROM proximity
                        WHERE {self._optimized_component_where_condition(components)} AND
                              {"(marker_1 IN $markers AND marker_2 IN $markers)" if markers else "TRUE"};
                    """
            params = {}
            if components is not None:
                params["components"] = components
            if markers is not None:
                params["markers"] = markers
            with self.view as connection:
                return connection.sql(query, params=params).execute().fetchone()[0]  # type: ignore
        except duckdb.CatalogException:
            return 0


class InplacePixelDataFilterer:
    """Class to filter a PXL file in place.

    This is mostly useful for testing purposes, when one wants to strip
    componentens from a PXL file to make it smaller and faster to work with.
    """

    def __init__(self, pxl_file: PxlFile):
        """Initialize the InplacePixelDataFilterer."""
        self.pxl_file = pxl_file

    def _update_metadata(
        self, connection: duckdb.DuckDBPyConnection, metadata: dict
    ) -> None:
        connection.sql(
            """
            DROP TABLE IF EXISTS metadata;
            CREATE TABLE metadata (value JSON);
            INSERT INTO metadata VALUES ($metadata);
            """,
            params={"metadata": metadata},
        )

    def _filter_edgelist(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        query = f"""
            DELETE FROM edgelist
            WHERE component NOT IN $components
        """
        connection.sql(query, params={"components": components})

    def _filter_proximity(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        try:
            query = f"""
                DELETE FROM proximity
                WHERE component NOT IN $components
            """
            connection.sql(query, params={"components": components})
        except duckdb.CatalogException:
            pass

    def _filter_layouts(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        try:
            query = f"""
                DELETE FROM layouts
                WHERE component NOT IN $components
            """
            connection.sql(query, params={"components": components})
        except duckdb.CatalogException:
            pass

    def _filter_adata(self, pxl_file: PxlFile, components: list[str]) -> None:
        adata = PixelDataQuerier(PixelDataViewer.from_files([pxl_file])).read_adata()
        adata = adata[adata.obs.index.isin(components)]
        with PixelFileWriter(pxl_file.path) as writer:
            writer.write_adata(adata)

    def filter_components(
        self, components: set[str] | list[str], metadata: dict | None = None
    ) -> None:
        """Filter the PXL file by components, only keeping the provided components.

        Note that if you provide metadata it will overwrite the existing metadata.
        If your do not provide metadata, the existing metadata will be kept.

        :param components: The components to keep.
        :param metadata: The metadata to write to the PXL
        """
        # I there are not components provided, do nothing.
        if not components:
            raise ValueError("You must provided at least one component to filter.")
        components_as_list: list[str] = normalize_input_to_list(components)  # type: ignore

        with duckdb.connect(self.pxl_file.path) as connection:
            self._filter_edgelist(connection, components_as_list)
            self._filter_proximity(connection, components_as_list)
            self._filter_layouts(connection, components_as_list)
            if metadata:
                self._update_metadata(connection, metadata)

        self._filter_adata(self.pxl_file, components_as_list)


def copy_databases(src_db: Path, target_db: Path) -> None:
    """Copy the contents of one PXL file to another.

    This is a trick that can be used to reclaim disk-space. See:
    https://duckdb.org/docs/stable/operations_manual/footprint_of_duckdb/reclaiming_space.html

    :param src_db: The source PXL file.
    :param target_db: The target PXL file.
    """
    query = f"""
    ATTACH '{str(src_db)}' AS src (READ_ONLY);
    ATTACH '{str(target_db)}' AS target;
    COPY FROM DATABASE src TO target;
    """

    with duckdb.connect() as connection:
        connection.execute(query)
