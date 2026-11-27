"""Obsm table names read from a PXL file must not run as SQL.

A ``.pxl`` file is a DuckDB database. ``read(...).adata()`` lists every table
whose name starts with ``__adata__obsm`` and pastes that name into later
statements. These tests build such a file and load it through that reader.

They fail while the embedded statement runs: it writes a file, or DuckDB
raises while executing it. They pass once each name is quoted before
execution and an ordinary obsm table still loads. Turning off DuckDB file
access is not enough, because ``COPY`` then raises and the name still ran.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import anndata
import duckdb
import numpy as np
import pandas as pd
import polars as pl

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelDataViewer, PixelFileWriter, PxlFile
from pixelator.pna.pixeldataset.io.query_builder import QueryBuilder

_COMPONENT = "c1"
_MARKER = "m1"
_INJECTED = "obsm-table-name-was-executed"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_path(path: Path) -> str:
    text = path.as_posix()
    if "'" in text or "\x00" in text:
        raise RuntimeError(f"test path cannot be embedded in SQL: {text}")
    return text


def _copy_statement(target: Path) -> str:
    return (
        f"COPY (SELECT '{_INJECTED}') TO '{_sql_path(target)}' (HEADER FALSE, QUOTE '')"
    )


def _statement_table_name(target: Path) -> str:
    """Name that becomes extra statements after a real ``__adata__obsm_x`` token."""
    return (
        "__adata__obsm_x WHERE FALSE; "
        f"{_copy_statement(target)}; "
        f"SELECT '{_COMPONENT}' AS index FROM range(1)"
    )


def _describe_table_name(target: Path) -> str:
    """Name that closes ``DESCRIBE SELECT * FROM (...)`` and then copies a file.

    Marker renames run ``table_columns_query`` on the same string before
    ``adata_obsm_query``. The copy has to sit outside that ``DESCRIBE``.
    """
    return (
        f"__adata__obsm_x); {_copy_statement(target)}; SELECT 'index' AS column_name --"
    )


def _create_table(
    connection: duckdb.DuckDBPyConnection, name: str, columns_sql: str
) -> None:
    connection.execute(f"CREATE TABLE {_quote(name)} ({columns_sql})")


def _write_single_sample(path: Path, obsm_name: str) -> None:
    """Write the tables ``read().adata()`` needs, plus one obsm table."""
    connection = duckdb.connect(str(path))
    try:
        connection.execute('CREATE TABLE "__adata__X" ("index" VARCHAR, "m1" INTEGER)')
        connection.execute('INSERT INTO "__adata__X" VALUES (?, ?)', [_COMPONENT, 1])
        connection.execute('CREATE TABLE "__adata__var" ("index" VARCHAR)')
        connection.execute('INSERT INTO "__adata__var" VALUES (?)', [_MARKER])
        connection.execute('CREATE TABLE "__adata__obs" ("index" VARCHAR)')
        connection.execute('INSERT INTO "__adata__obs" VALUES (?)', [_COMPONENT])
        connection.execute("CREATE TABLE edgelist (component VARCHAR)")
        connection.execute('CREATE TABLE "__adata__uns" (value VARCHAR)')
        connection.execute('INSERT INTO "__adata__uns" VALUES (?)', [json.dumps({})])
        connection.execute("CREATE TABLE metadata (value VARCHAR)")
        connection.execute(
            "INSERT INTO metadata VALUES (?)",
            [json.dumps({"sample_name": "s1"})],
        )
        # The injected text's first statement names this table. It is also a
        # legitimate obsm key the reader should keep once the name is quoted.
        _create_table(connection, "__adata__obsm_x", '"index" VARCHAR, v INTEGER')
        connection.execute(
            'INSERT INTO "__adata__obsm_x" VALUES (?, ?)', [_COMPONENT, 1]
        )
        _create_table(connection, obsm_name, '"index" VARCHAR, v INTEGER')
        connection.execute(
            f"INSERT INTO {_quote(obsm_name)} VALUES (?, ?)", [_COMPONENT, 1]
        )
    finally:
        connection.close()


def _assert_embedded_sql_not_run(load: Callable[[], object], target: Path) -> object:
    """Return ``load()`` only when ``target`` was not written by embedded SQL.

    A DuckDB error counts as execution too. Disabling file access makes
    ``COPY`` raise and still leaves the table name running as SQL.
    """
    error: Exception | None = None
    loaded: object | None = None
    try:
        loaded = load()
    except Exception as exc:
        error = exc
    if target.exists():
        raise AssertionError(
            "an obsm table name was executed as SQL and wrote "
            f"{target} ({target.read_bytes()!r})"
        )
    if isinstance(error, duckdb.Error):
        raise AssertionError(
            f"an obsm table name was executed as SQL: {error}"
        ) from error
    if error is not None:
        raise error
    return loaded


def test_adata_runs_sql_embedded_in_an_obsm_table_name(tmp_path: Path) -> None:
    """``read().adata()`` must not run statements stored in an obsm table name."""
    target = tmp_path / "obsm-copy.txt"
    pxl = tmp_path / "sample.pxl"
    _write_single_sample(pxl, _statement_table_name(target))

    loaded = _assert_embedded_sql_not_run(lambda: read(pxl).adata(), target)

    assert "x" in loaded.obsm


def test_marker_renames_run_sql_embedded_in_an_obsm_table_name(tmp_path: Path) -> None:
    """Column lookup for a renamed marker must not run the obsm table name."""
    target = tmp_path / "columns-copy.txt"
    old = _write_panel_sample(tmp_path / "old.pxl", "old", "CD3", "1.0.0")
    new = _write_panel_sample(tmp_path / "new.pxl", "new", "CD3E", "1.0.1")
    name = _describe_table_name(target)
    connection = duckdb.connect(str(old.path))
    try:
        _create_table(connection, name, '"index" VARCHAR, v INTEGER')
        connection.execute(f"INSERT INTO {_quote(name)} VALUES (?, ?)", ["old", 1])
    finally:
        connection.close()

    loaded = _assert_embedded_sql_not_run(
        lambda: read([old.path, new.path]).adata(), target
    )

    assert "x" in loaded.obsm


def test_obsm_query_runs_sql_embedded_in_an_undotted_table_name(tmp_path: Path) -> None:
    """``adata_obsm_query`` must quote a table name that has no dot in it.

    ``AnnDataHelper`` always passes a dotted name. This is the other branch of
    the same builder, executed on a real session.
    """
    # The name must not contain ".", or QueryBuilder takes the dotted branch.
    target = tmp_path / "undottedcopy"
    pxl = tmp_path / "sample.pxl"
    bare_name = _statement_table_name(target)
    assert "." not in bare_name
    _write_single_sample(pxl, bare_name)
    viewer = PixelDataViewer.from_files([PxlFile(pxl)])
    db_name = viewer.normalized_sample_db_name("s1")

    def load() -> pl.DataFrame:
        with viewer.open() as session:
            return session.execute_eager(
                QueryBuilder().adata_obsm_query(db_name, bare_name)
            )

    frame = _assert_embedded_sql_not_run(load, target)
    assert "v" in frame.columns


def test_ordinary_obsm_table_is_loaded(tmp_path: Path) -> None:
    """A normal ``__adata__obsm_<key>`` table is still part of ``adata().obsm``."""
    pxl = tmp_path / "sample.pxl"
    _write_single_sample(pxl, "__adata__obsm_pca")
    connection = duckdb.connect(str(pxl))
    try:
        connection.execute('DELETE FROM "__adata__obsm_pca"')
        connection.execute(
            'INSERT INTO "__adata__obsm_pca" VALUES (?, ?)', [_COMPONENT, 4]
        )
    finally:
        connection.close()

    loaded = read(pxl).adata()

    assert list(loaded.obsm["pca"]["v"]) == [4]
    assert list(loaded.obsm["x"]["v"]) == [1]


def _panel(version: str, marker: str) -> PNAAntibodyPanel:
    rows = [
        {
            "marker_id": marker,
            "control": False,
            "sequence_1": "AAAA",
            "sequence_2": "AAAA",
        }
    ]
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(name="base", version=version, product="kit"),
    )


def _write_panel_sample(path: Path, sample: str, marker: str, version: str) -> PxlFile:
    edgelist = pl.DataFrame(
        {
            "umi1": [1],
            "umi2": [2],
            "read_count": [1],
            "marker_1": [marker],
            "marker_2": [marker],
            "component": [sample],
        }
    )
    with PixelFileWriter(path) as writer:
        writer.write_edgelist(edgelist)
        writer.write_panel(_panel(version, marker))
        adata = anndata.AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame(index=[sample]),
            var=pd.DataFrame(index=pd.Index([marker], name="marker_id")),
            obsm={"x": pd.DataFrame({"v": [1]}, index=[sample])},
        )
        writer.write_adata(adata)
        writer.write_metadata({"sample_name": sample})
    return PxlFile(path)
