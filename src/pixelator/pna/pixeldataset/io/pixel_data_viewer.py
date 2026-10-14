"""Virtual SQL views over multiple PXL files.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl
from anndata import AnnData

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.pna.config.panel import PNAAntibodyPanel, align_panel_patches
from pixelator.pna.config.panel_align import (
    apply_hash_count_renames,
    apply_marker_renames_to_adata,
)

from .pxl_file import PXL_FILE_MANDATOR_TABLES, PXL_FILE_OTHER_TABLES, PxlFile
from .query_builder import Query

# Unquoted DuckDB identifiers used in generated ATTACH / CREATE VIEW SQL.
_SESSION_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")

# Characters / substrings that must not appear in sample labels embedded as '...' literals.
_SAMPLE_LABEL_UNSAFE_RE = re.compile(r"['\";\x00-\x1f\\]|--|/\*|\*/")


def _validate_session_sql_identifier(value: str, *, what: str) -> None:
    if not _SESSION_SQL_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {what} {value!r}: use only ASCII letters, digits, and underscores, "
            "and start with a letter or underscore."
        )


def _validate_session_sample_label(value: str, *, what: str = "sample name") -> None:
    if _SAMPLE_LABEL_UNSAFE_RE.search(value):
        raise ValueError(
            f"Invalid {what} {value!r}: must not contain quotes, semicolons, "
            "ASCII control characters, backslashes, or SQL comment markers (--, /*, */)."
        )


def _validate_session_file_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PXL path {path!r} does not exist.")
    s = str(path)
    if "\x00" in s or "'" in s or any(ord(c) < 32 for c in s):
        raise ValueError(
            f"Invalid PXL path {path!r}: path must not contain apostrophe (U+0027), "
            "NUL, or other ASCII control characters."
        )


class PixelDataViewer:
    """Maps sample names to PXL files and can open a :class:`~pixelator.pna.pixeldataset.io.pixel_data_viewer.PixelDataViewerSession`.

    Query execution uses a :class:`~pixelator.pna.pixeldataset.io.pixel_data_viewer.PixelDataViewerSession` from ``viewer.open()``

    .. code-block:: python

        with viewer.open() as session:
            df = session.execute_eager(Query("SELECT * FROM edgelist", {}))

        # or: session = viewer.open(); ...; session.close()
    """

    def __init__(
        self,
        sample_name_to_pxl_file_mapping: dict[str, PxlFile],
    ):
        """Initialize the PixelDataViewer.

        Args:
            sample_name_to_pxl_file_mapping: A dictionary mapping sample names to PxlFile objects.

        Raises:
            ValueError: If any of the files are not valid PXL files.
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

        self._marker_renames: dict[str, dict[str, str]] | None = None
        self._hash_count_renames: dict[str, dict[str, str]] | None = None
        self._upgraded_panels: dict[str, PNAAntibodyPanel | None] | None = None

    def _map_sample_names_to_db_names(
        self, sample_name_to_pxl_file_mapping: dict[str, PxlFile]
    ) -> dict[str, str]:
        """Create a normalized sample name lookup table for working with the duckdb files.

        Args:
            sample_name_to_pxl_file_mapping: A dictionary mapping sample names to PxlFile objects.
        """

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
        """Create a PixelDataViewer from a dictionary mapping sample names to PxlFile objects.

        Args:
            sample_name_to_pxl_file_mapping: A dictionary mapping sample names to PxlFile objects.
        """
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

    def open(self) -> PixelDataViewerSession:
        """Return a new session with an open DuckDB connection (context manager or ``close()``)."""
        sources: list[tuple[str, Path, str]] = [
            (sample_name, pxl_file.path, self._get_normalized_name(sample_name))
            for sample_name, pxl_file in self._db_to_file_mapping.items()
        ]
        return PixelDataViewerSession(sources)  # type: ignore[arg-type]

    def sample_names(self) -> list[str]:
        """Return the list of sample names known to the view."""
        return list(self._db_to_file_mapping.keys())

    def normalized_sample_db_name(self, sample_name: str) -> str:
        """Return the attached DuckDB database name for a sample."""
        return self._get_normalized_name(sample_name)

    def apply_panel_patch_to_adatas(
        self,
        samples: list[str] | list[AnnData],
        adatas: list[AnnData] | None = None,
    ) -> list[AnnData]:
        """Apply the view's panel patch bump and attach panel columns.

        Marker ids are renamed first. Panel columns already on ``var`` are
        then replaced by the panel this view settled on, so a legacy file and
        a new file end up with the same in-memory columns. ``uns`` panel
        metadata is removed. ``samples`` may be omitted by passing only the
        AnnData list. Sample names are then taken from the view, in view order.
        """
        if adatas is None:
            adatas = list(samples)  # type: ignore[arg-type]
            samples = list(self.sample_names())
        sample_names = [str(sample) for sample in samples]
        if self._marker_renames is None:
            self._compute_panel_patch(sample_names, adatas)
        else:
            self._apply_cached_panel_patch(sample_names, adatas)
        return adatas

    def _compute_panel_patch(self, samples: list[str], adatas: list[AnnData]) -> None:
        panels: list[PNAAntibodyPanel] = []
        indexed_adatas: list[AnnData] = []
        indexed_samples: list[str] = []
        indexed_metadata: list[dict] = []
        for sample, adata in zip(samples, adatas):
            pxl_file = self._db_to_file_mapping[sample]
            panel = pxl_file.read_panel()
            if panel is None:
                continue
            panels.append(panel)
            indexed_adatas.append(adata)
            indexed_samples.append(sample)
            indexed_metadata.append(pxl_file.metadata())

        marker_renames: dict[str, dict[str, str]] = {sample: {} for sample in samples}
        hash_count_renames: dict[str, dict[str, str]] = {
            sample: {} for sample in samples
        }
        upgraded_panels: dict[str, PNAAntibodyPanel | None] = {
            sample: None for sample in samples
        }
        for sample, panel in zip(indexed_samples, panels):
            upgraded_panels[sample] = panel
        if len(panels) >= 2:
            upgraded, data_renames, hash_renames = align_panel_patches(
                panels, indexed_adatas, pxl_file_metadata=indexed_metadata
            )
            for sample, panel, data_map, hash_map in zip(
                indexed_samples, upgraded, data_renames, hash_renames
            ):
                marker_renames[sample] = data_map
                hash_count_renames[sample] = hash_map
                upgraded_panels[sample] = panel
        self._marker_renames = marker_renames
        self._hash_count_renames = hash_count_renames
        self._upgraded_panels = upgraded_panels
        self._apply_cached_panel_patch(samples, adatas)

    def _apply_cached_panel_patch(
        self, samples: list[str], adatas: list[AnnData]
    ) -> None:
        marker_renames = self._marker_renames or {}
        hash_renames = self._hash_count_renames or {}
        upgraded = self._upgraded_panels or {}
        for sample, adata in zip(samples, adatas):
            apply_marker_renames_to_adata(adata, marker_renames.get(sample, {}))
            apply_hash_count_renames(adata, hash_renames.get(sample, {}))
            panel = upgraded.get(sample)
            if panel is not None:
                _attach_panel_columns(adata, panel)


def _attach_panel_columns(adata: AnnData, panel: PNAAntibodyPanel) -> None:
    """Join ``panel`` onto ``var`` and drop legacy ``uns`` panel metadata.

    Columns already on ``var`` that also belong to the panel are removed
    first, so a legacy file does not keep the pre-bump values. A column the
    stored panel listed and this panel no longer has is removed as well.
    Markers that are not already in ``var`` are not added.
    """
    metadata = adata.uns.get("panel_metadata") if adata.uns is not None else None
    legacy_columns = set()
    if isinstance(metadata, dict):
        legacy_columns = {str(column) for column in metadata.get("panel_columns", [])}
    panel_columns = set(map(str, panel.df.columns))
    to_drop = [
        column
        for column in adata.var.columns
        if str(column) in panel_columns or str(column) in legacy_columns
    ]
    if to_drop:
        adata.var = adata.var.drop(columns=to_drop)
    adata.var = adata.var.join(panel.df.reindex(adata.var_names), how="left")
    if adata.uns is not None and "panel_metadata" in adata.uns:
        del adata.uns["panel_metadata"]


class PixelDataViewerSession:
    """DuckDB session over one or more attached PXL files.

    Pass a list of ``(sample_name, pxl_path, db_name)`` tuples, where
    ``db_name`` is the DuckDB attach alias (see :meth:`~pixelator.pna.pixeldataset.io.pixel_data_viewer.PixelDataViewer.normalized_sample_db_name`).

    At construction time, ``sample_name``, ``db_name``, and the path string are
    validated to only use valid DuckDB identifiers.

    The connection is opened when the session is constructed. Close it with
    ``session.close()`` or by using ``with viewer.open() as session:`` (exit
    calls ``close()``). Each session uses its own connection.
    """

    def __init__(self, sources: list[tuple[str, Path | str, str]]) -> None:
        """Open a DuckDB connection and attach each PXL file in ``sources``."""
        normalized: list[tuple[str, Path, str]] = [
            (sample_name, Path(path), db_name) for sample_name, path, db_name in sources
        ]
        for sample_name, path, db_name in normalized:
            _validate_session_sample_label(sample_name)
            _validate_session_sql_identifier(db_name, what="database alias (db_name)")
            _validate_session_file_path(path)
        self._sources = normalized
        self._connection: duckdb.DuckDBPyConnection | None = (
            self._create_open_connection()
        )

    def _create_open_connection(self) -> duckdb.DuckDBPyConnection:
        """Create an in-memory DuckDB connection with PXL files attached."""
        connection = connect_duckdb(":memory:")
        self._attach_to_files(connection)
        for table in PXL_FILE_OTHER_TABLES:
            self._simple_union_table_view(connection, table, table)
        return connection

    def _attach_to_files(self, connection: duckdb.DuckDBPyConnection) -> None:
        query = ""
        for _sample_name, path, db_name in self._sources:
            query += f"ATTACH DATABASE '{path}' AS {db_name} (READ_ONLY);\n"
        connection.execute(query)

    def _simple_union_table_view(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_name: str,
        view_name: str,
    ) -> None:
        """Create a view that unions given table from all the underlying samples."""
        _validate_session_sql_identifier(table_name, what="table name")
        _validate_session_sql_identifier(view_name, what="view name")
        try:
            # We are creating the view manually with a string here
            # since duckdb does not support creating views with parameters,
            # see: https://github.com/duckdb/duckdb/issues/13069
            table_queries: list[str] = []
            for sample_name, _path, db_name in self._sources:
                table_queries.append(
                    f"SELECT *, '{sample_name}' AS sample FROM {db_name}.{table_name}"
                )
            if not table_queries:
                return

            union_query = "\n UNION ALL \n".join(table_queries)
            view_query_str = f"CREATE VIEW {view_name} AS\n{union_query}"

            connection.execute(view_query_str)
        except duckdb.CatalogException:
            if table_name in PXL_FILE_MANDATOR_TABLES:
                ref_path = self._sources[0][1] if self._sources else Path()
                raise ValueError(
                    f"Mandatory table {table_name} is missing from {ref_path}- are you sure this is a pxl file?"
                ) from None
            # note that we are ignoring the exception here, since it is expected
            # that some tables may not be present in all files.
            pass

    def close(self) -> None:
        """Close the DuckDB connection. Safe to call more than once."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> PixelDataViewerSession:
        """Return this session for use in the ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the DuckDB connection."""
        self.close()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the DuckDB connection for this session."""
        if self._connection is None:
            raise RuntimeError(
                "The viewer session is closed; open a new session with viewer.open()."
            )
        return self._connection

    def execute_lazy(self, query: Query) -> pl.LazyFrame:
        """Execute a query and return a Polars LazyFrame."""
        return (
            self.get_connection()
            .execute(query.sql, parameters=query.params)
            .pl(lazy=True)
        )

    def execute_eager(self, query: Query) -> pl.DataFrame:
        """Execute a query and return a Polars DataFrame."""
        return self.get_connection().execute(query.sql, parameters=query.params).pl()

    def execute_scalar(self, query: Query) -> int:
        """Execute a scalar query and return first value."""
        self.get_connection().execute(query.sql, parameters=query.params)
        row = self.get_connection().fetchone()
        if row is None:
            raise RuntimeError("Scalar query returned no rows")
        return row[0]  # type: ignore[return-value]

    def execute_arrow_reader(self, query: Query, batch_size: int):
        """Execute a query and return Arrow reader."""
        result = self.get_connection().sql(query.sql, params=query.params)
        return result.fetch_arrow_reader(batch_size=batch_size)

    def load_stochastic_extension(self) -> None:
        """Load the DuckDB stochastic extension, installing it only if needed."""
        if self._connection is None:
            return
        try:
            self._connection.execute("LOAD stochastic;")
        except duckdb.Error:
            self._connection.execute("INSTALL stochastic FROM community;")
            self._connection.execute("LOAD stochastic;")
