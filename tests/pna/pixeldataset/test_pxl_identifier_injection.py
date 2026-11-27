"""Opening a .pxl file must not run SQL stored in its catalog identifiers.

A .pxl file is a DuckDB database. These tests build one whose table or schema
names contain a second statement, then open it through the reader the product
uses. They also pass an obsm key and a catalog name through ``PixelFileWriter``.

Each malicious statement would overwrite a sentinel file. Those tests fail
while that statement runs, including when the product raises afterwards. They
pass when the name is rejected or quoted and the sentinel is left untouched.
The two tests of a plain ``spatial`` name already pass and must keep passing.

``pixelator`` CLI commands that load AnnData (denoise, analysis, graph collapse,
sample calling) call the same ``adata()`` path and are not spawned here.
``describe``, ``__str__``, and ``filter`` call ``sample_names``, ``components``,
or ``markers``, which are tested.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper

_SENTINEL_TEXT = "SAFE\n"
_INJECTION_MARKER = "pixelator-obsm-injection"
_COMPONENT = "comp1"
_MARKER = "MarkerA"


def _quoted_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _copy_statement(sentinel: Path) -> str:
    """Statement that overwrites ``sentinel`` when DuckDB runs it."""
    return (
        f"COPY (SELECT {_sql_literal(_INJECTION_MARKER)}) TO "
        f"{_sql_literal(str(sentinel))} (FORMAT CSV, HEADER false, QUOTE '')"
    )


def _expect_sentinel_unchanged(sentinel: Path, action) -> None:
    """Run ``action`` and fail if it creates, deletes, or rewrites ``sentinel``."""
    sentinel.write_text(_SENTINEL_TEXT)
    try:
        action()
    except Exception:
        pass
    assert sentinel.is_file()
    assert sentinel.read_text() == _SENTINEL_TEXT


def _obsm_select_name(sentinel: Path) -> str:
    """Table name pasted into ``SELECT * FROM <name>`` by the obsm reader.

    The name starts with a real table, ``__adata__obsm_a``, so the first
    statement is valid. The copy is a following statement. The trailing select
    matches the component id so the read can finish after the copy has run.
    """
    return (
        "__adata__obsm_a; "
        + _copy_statement(sentinel)
        + f'; SELECT {_sql_literal(_COMPONENT)} AS "index" FROM range(1)'
    )


def _obsm_describe_name(sentinel: Path) -> str:
    """Table name pasted into ``DESCRIBE SELECT * FROM <name>``.

    ``table_columns_query`` wraps the name in parentheses. The name closes
    that expression before the copy, which is the statement the column lookup
    runs before the obsm select.
    """
    return (
        "__adata__obsm_a); "
        + _copy_statement(sentinel)
        + "; SELECT column_name FROM (SELECT 'index' AS column_name"
    )


def _create_base_pxl(path: Path, *, sample: str, marker: str) -> None:
    """Write the tables a reader requires before it lists obsm tables."""
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            f"""
            CREATE TABLE edgelist (
                umi1 UBIGINT,
                umi2 UBIGINT,
                read_count UINTEGER,
                marker_1 VARCHAR,
                marker_2 VARCHAR,
                component VARCHAR
            );
            CREATE TABLE "__adata__X" ("index" VARCHAR, {_quoted_identifier(marker)} DOUBLE);
            INSERT INTO "__adata__X" VALUES ({_sql_literal(_COMPONENT)}, 1.0);
            CREATE TABLE "__adata__var" ("index" VARCHAR);
            INSERT INTO "__adata__var" VALUES ({_sql_literal(marker)});
            CREATE TABLE "__adata__obs" ("index" VARCHAR);
            INSERT INTO "__adata__obs" VALUES ({_sql_literal(_COMPONENT)});
            CREATE TABLE "__adata__uns" (value JSON);
            INSERT INTO "__adata__uns" VALUES ('{{}}');
            CREATE TABLE metadata (value JSON);
            INSERT INTO metadata VALUES ({_sql_literal('{"sample_name": "' + sample + '"}')});
            CREATE TABLE "__adata__obsm_a" ("index" VARCHAR, {_quoted_identifier(marker)} DOUBLE);
            INSERT INTO "__adata__obsm_a" VALUES ({_sql_literal(_COMPONENT)}, 1.0);
            """
        )
    finally:
        connection.close()


def _add_table(path: Path, name: str) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            f"CREATE TABLE {_quoted_identifier(name)} (\"index\" VARCHAR)"
        )
    finally:
        connection.close()


def _reader_pxl(path: Path, extra_table: str | None) -> Path:
    _create_base_pxl(path, sample="sample", marker=_MARKER)
    if extra_table is not None:
        _add_table(path, extra_table)
    return path


def _minimal_adata() -> AnnData:
    return AnnData(
        X=np.zeros((1, 1)),
        obs=pd.DataFrame(index=pd.Index([_COMPONENT], name="component")),
        var=pd.DataFrame(index=pd.Index([_MARKER], name="marker_id")),
    )


def _panel(version: str, markers: dict[str, str]) -> PNAAntibodyPanel:
    rows = [
        {
            "marker_id": marker_id,
            "control": False,
            "sequence_1": sequence,
            "sequence_2": sequence,
        }
        for marker_id, sequence in markers.items()
    ]
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(name="base", version=version, product="kit"),
    )


def _write_aligned_pxl(path: Path, sample: str, panel: PNAAntibodyPanel, marker: str) -> None:
    edgelist = pl.DataFrame(
        {
            "umi1": [1],
            "umi2": [2],
            "read_count": [1],
            "marker_1": [marker],
            "marker_2": [marker],
            "component": [sample],
        },
        schema={
            "umi1": pl.UInt64,
            "umi2": pl.UInt64,
            "read_count": pl.UInt32,
            "marker_1": pl.Utf8,
            "marker_2": pl.Utf8,
            "component": pl.Utf8,
        },
    )
    adata = AnnData(
        X=np.array([[1.0]]),
        obs=pd.DataFrame(index=pd.Index([sample], name="component")),
        var=pd.DataFrame(index=pd.Index([marker], name="marker_id")),
    )
    with PixelFileWriter(path) as writer:
        writer.write_edgelist(edgelist)
        writer.write_panel(panel)
        writer.write_adata(adata)
        writer.write_metadata({"sample_name": sample})


@pytest.mark.parametrize(
    "entry",
    ["adata", "sample_names", "components", "markers"],
)
def test_pxl_read_does_not_execute_obsm_table_names(tmp_path: Path, entry: str) -> None:
    """Readers that materialize AnnData must not run SQL from obsm table names."""
    sentinel = tmp_path / "sentinel.txt"
    pxl = _reader_pxl(tmp_path / "sample.pxl", _obsm_select_name(sentinel))
    dataset = read(pxl)

    def open_dataset() -> None:
        method = getattr(dataset, entry)
        if entry == "adata":
            method(add_log1p_transform=False, add_clr_transform=False)
        else:
            method()

    _expect_sentinel_unchanged(sentinel, open_dataset)


def test_component_filtered_anndata_read_does_not_execute_obsm_table_names(
    tmp_path: Path,
) -> None:
    """A component filter still executes the obsm query, with its parameters bound."""
    sentinel = tmp_path / "sentinel.txt"
    pxl = _reader_pxl(tmp_path / "sample.pxl", _obsm_select_name(sentinel))
    dataset = read(pxl)
    helper = AnnDataHelper(dataset.view, components={_COMPONENT})

    _expect_sentinel_unchanged(
        sentinel,
        lambda: helper.read_adata(add_log1p_transform=False, add_clr_transform=False),
    )


def test_aligned_read_does_not_execute_obsm_names_in_column_lookup(
    tmp_path: Path,
) -> None:
    """Marker alignment looks up obsm columns before reading them.

    That lookup passes the qualified table name into ``table_columns_query``.
    The name below is shaped for that statement. The older panel is renamed,
    so the lookup runs.
    """
    sentinel = tmp_path / "sentinel.txt"
    old = tmp_path / "old.pxl"
    new = tmp_path / "new.pxl"
    _write_aligned_pxl(old, "old", _panel("1.0.0", {"CD3": "AAAA"}), "CD3")
    _write_aligned_pxl(new, "new", _panel("1.0.1", {"CD3E": "AAAA"}), "CD3E")
    _add_table(old, "__adata__obsm_a")
    _add_table(old, _obsm_describe_name(sentinel))

    dataset = read([old, new])
    _expect_sentinel_unchanged(
        sentinel,
        lambda: dataset.adata(add_log1p_transform=False, add_clr_transform=False),
    )


def test_write_adata_does_not_execute_obsm_keys(tmp_path: Path) -> None:
    """``write_adata`` must not run SQL from an obsm key."""
    sentinel = tmp_path / "sentinel.txt"
    key = (
        'x AS SELECT \'comp1\' AS "index"; '
        + _copy_statement(sentinel)
        + "; CREATE TABLE __adata__obsm_kept AS SELECT * FROM obsm_layer; --"
    )
    adata = _minimal_adata()
    adata.obsm[key] = pd.DataFrame({"v": [1.0]}, index=adata.obs.index)

    def write() -> None:
        with PixelFileWriter(tmp_path / "out.pxl") as writer:
            writer.write_adata(adata)

    _expect_sentinel_unchanged(sentinel, write)


def test_write_adata_does_not_execute_existing_catalog_names(tmp_path: Path) -> None:
    """Rewriting AnnData drops existing tables by name, schema, and database.

    Those identifiers are quoted, but a quote inside the schema name is not
    escaped. The schema below closes that quote. ``foo`` is a real table, so
    the leading ``DROP`` succeeds and the following statement runs.
    """
    sentinel = tmp_path / "sentinel.txt"
    pxl = tmp_path / "existing.pxl"
    schema = 'foo"; ' + _copy_statement(sentinel) + "; --"
    connection = duckdb.connect(str(pxl))
    try:
        connection.execute('CREATE TABLE foo ("index" VARCHAR)')
        connection.execute('CREATE TABLE "__adata__X" ("index" VARCHAR)')
        connection.execute(f"CREATE SCHEMA {_quoted_identifier(schema)}")
        connection.execute(
            f"CREATE TABLE {_quoted_identifier(schema)}.__adata__evil (\"index\" VARCHAR)"
        )
    finally:
        connection.close()

    def rewrite() -> None:
        with PixelFileWriter(pxl) as writer:
            writer.write_adata(_minimal_adata())

    _expect_sentinel_unchanged(sentinel, rewrite)


def test_plain_obsm_table_is_loaded(tmp_path: Path) -> None:
    """A normal obsm table name is still returned on ``adata().obsm``."""
    path = tmp_path / "sample.pxl"
    _create_base_pxl(path, sample="sample", marker=_MARKER)
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            'CREATE TABLE "__adata__obsm_spatial" ("index" VARCHAR, "x" DOUBLE)'
        )
        connection.execute(
            f"INSERT INTO \"__adata__obsm_spatial\" VALUES ({_sql_literal(_COMPONENT)}, 1.0)"
        )
    finally:
        connection.close()

    adata = read(path).adata(add_log1p_transform=False, add_clr_transform=False)

    assert set(adata.obsm) == {"a", "spatial"}
    assert list(adata.obsm["spatial"].index) == [_COMPONENT]


def test_plain_obsm_key_is_written(tmp_path: Path) -> None:
    """A normal obsm key is still stored under its ``__adata__obsm_`` table."""
    adata = _minimal_adata()
    adata.obsm["spatial"] = pd.DataFrame({"x": [1.0]}, index=adata.obs.index)
    path = tmp_path / "out.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_adata(adata)

    connection = duckdb.connect(str(path), read_only=True)
    try:
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    finally:
        connection.close()
    assert "__adata__obsm_spatial" in names
