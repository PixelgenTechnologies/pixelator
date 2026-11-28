"""Edgelist column names must be quoted identifiers, not SQL text.

``collapse_identical_edges`` reads column names from the caller-supplied ``.pxl``
and pastes each one into the ``SELECT`` and ``GROUP BY`` of ``CREATE OR REPLACE
TABLE edgelist``. These tests fail while that text is executed. They pass once
every name, including the count columns inside ``SUM``, is quoted as a DuckDB
identifier and embedded double quotes are doubled.

Copyright © 2026 Pixelgen Technologies AB.
"""

import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pixelator_core
from anndata import AnnData
from click.testing import CliRunner

from pixelator import __version__, cli
from pixelator.common.config.panel import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.graph_collapse.collapse import collapse_identical_edges
from pixelator.pna.pixeldataset.io import PixelFileWriter
from pixelator.pna.utils import init_duckdb_conn

# Placeholder file body. This is not a credential.
SENTINEL = "PIXELATOR_SENTINEL_FILE_BODY"


def _file_read_expression(path: Path) -> str:
    """Scalar subquery that reads ``path`` when DuckDB evaluates it as SQL."""
    return (
        "(SELECT content FROM read_csv("
        f"'{path.as_posix()}', header=false, "
        "columns={'content': 'VARCHAR'}))"
    )


def _write_sentinel(path: Path) -> None:
    path.write_text(SENTINEL + "\n")


def _create_duplicate_edges(con: duckdb.DuckDBPyConnection, column_name: str) -> None:
    """Two identical edges plus one extra column whose name is attacker-chosen."""
    escaped = column_name.replace('"', '""')
    con.execute(
        f"""
        CREATE TABLE edgelist (
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            umi1 UBIGINT,
            umi2 UBIGINT,
            read_count UINTEGER,
            uei_count USMALLINT,
            corrected_read_count UINTEGER,
            "{escaped}" VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO edgelist VALUES
            ('c', 'CD3', 'CD4', 1, 2, 2, 1, 0, 'plain'),
            ('c', 'CD3', 'CD4', 1, 2, 5, 3, 1, 'plain')
        """
    )


def _assert_edges_collapsed_without_reading(con: duckdb.DuckDBPyConnection) -> None:
    """The two edges are one row, counts are summed, and the file body is absent."""
    rows = con.execute("SELECT * FROM edgelist").fetchall()
    assert len(rows) == 1
    values = rows[0]
    rendered = [str(value) for value in values]
    assert SENTINEL not in rendered
    assert "plain" in rendered
    # read_count 2+5, uei_count 1+3, corrected_read_count 0+1
    assert 7 in values
    assert 4 in values
    assert 1 in values


def test_collapse_identical_edges_does_not_evaluate_column_names(tmp_path: Path):
    """A column name that is a file-reading subquery must stay data.

    The stored cell is ``plain``. Evaluating the name replaces that cell with
    the file body and still sums the count columns.
    """
    sentinel = tmp_path / "sentinel.csv"
    _write_sentinel(sentinel)
    con = init_duckdb_conn()
    try:
        _create_duplicate_edges(con, _file_read_expression(sentinel))
        collapse_identical_edges(con)
        _assert_edges_collapsed_without_reading(con)
    finally:
        con.close()


def test_collapse_identical_edges_escapes_quotes_inside_column_names(tmp_path: Path):
    """Wrapping a name in double quotes is not enough when the name contains one.

    The embedded quote closes a naively quoted identifier. The following
    subquery is then SQL, and ``--`` comments out the wrapper's closing quote.
    Doubling the embedded quote keeps the whole name one identifier.
    """
    sentinel = tmp_path / "sentinel.csv"
    _write_sentinel(sentinel)
    column_name = f'x", {_file_read_expression(sentinel)} --'
    con = init_duckdb_conn()
    try:
        _create_duplicate_edges(con, column_name)
        collapse_identical_edges(con)
        _assert_edges_collapsed_without_reading(con)
    finally:
        con.close()


def _panel() -> PNAAntibodyPanel:
    markers = pd.DataFrame(
        [
            {
                "marker_id": "CD3",
                "control": False,
                "sequence_1": "AAAA",
                "sequence_2": "AAAA",
            },
            {
                "marker_id": "CD4",
                "control": False,
                "sequence_1": "CCCC",
                "sequence_2": "CCCC",
            },
        ]
    ).set_index("marker_id")
    return PNAAntibodyPanel.from_metadata(
        markers,
        AntibodyPanelMetadata(name="base", version="1.0.0", product="proxiome"),
        file_name="base.csv",
    )


def _write_collapsible_pxl(path: Path, column_name: str) -> None:
    """A pxl whose edgelist has one leaf UMI within the default Hamming cutoff.

    The shape matches the binding fixture in ``test_record_batches_stream_into_graph_collapse``:
    ``CCAA`` and ``CAAA`` are degree-1 umi2 nodes that collapse onto ``AAAA``.
    Both ends of each collapsed pair share a marker, so the rename is kept and
    ``collapse_identical_edges`` runs.
    """
    if path.exists():
        path.unlink()
    umi = pixelator_core.encode_dna_2bit
    edges = [
        (umi("GGGG"), umi("AAAA"), "cell", "CD3", "CD4", 2, "plain"),
        (umi("GGGG"), umi("CCAA"), "cell", "CD3", "CD4", 3, "plain"),
        (umi("GGGG"), umi("CAAA"), "cell", "CD3", "CD4", 4, "plain"),
        (umi("TTTT"), umi("AAAA"), "cell", "CD3", "CD4", 5, "plain"),
    ]
    escaped = column_name.replace('"', '""')
    con = duckdb.connect(str(path))
    con.execute(
        f"""
        CREATE TABLE edgelist (
            umi1 UBIGINT,
            umi2 UBIGINT,
            component VARCHAR,
            marker_1 VARCHAR,
            marker_2 VARCHAR,
            read_count UINTEGER,
            "{escaped}" VARCHAR
        )
        """
    )
    con.executemany("INSERT INTO edgelist VALUES (?, ?, ?, ?, ?, ?, ?)", edges)
    con.close()

    obs = pd.DataFrame(
        {"reads_in_component": [14], "n_umi": [5]},
        index=pd.Index(["cell"], name="component"),
    )
    var = pd.DataFrame(index=pd.Index(["CD3", "CD4"], name="marker_id"))
    adata = AnnData(X=np.zeros((1, 2), dtype=np.uint32), obs=obs, var=var)
    with PixelFileWriter(path) as writer:
        writer.write_metadata(
            {
                "sample_name": "victim",
                "version": __version__,
                "technology": "single-cell-pna",
            }
        )
        writer.write_panel(_panel())
        writer.write_adata(adata)


def _cell_texts(path: Path) -> list[str]:
    """Every cell of every table, so a value written before a later error is visible."""
    texts: list[str] = []
    with duckdb.connect(str(path), read_only=True) as con:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            for row in con.execute(f"SELECT * FROM {quoted}").fetchall():
                texts.extend(str(value) for value in row)
    return texts


def test_graph_collapse_command_does_not_read_a_file_named_by_a_column(tmp_path: Path):
    """``pixelator single-cell-pna graph-collapse`` must not evaluate column names.

    The input is a ``.pxl`` the command accepts: it has the mandatory tables, a
    stored panel, and a collapsible UMI. The extra edgelist column is named with
    a subquery that reads a local file. After a kept rename, that name is copied
    into the collapse statement.
    """
    sentinel = tmp_path / "sentinel.csv"
    _write_sentinel(sentinel)
    source = tmp_path / "victim.pxl"
    _write_collapsible_pxl(source, _file_read_expression(sentinel))
    output = tmp_path / "out"
    if output.exists():
        shutil.rmtree(output)

    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph-collapse",
            str(source),
            "--output",
            str(output),
            "--design",
            "proxiome-v1",
        ],
    )
    assert result.exit_code == 0, result.output

    report_files = list(output.rglob("*.report.json"))
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_text())
    assert report["nodes_collapsed"] > 0

    written = list(output.rglob("*.pxl"))
    assert len(written) == 1
    texts = _cell_texts(written[0])
    assert SENTINEL not in texts
    assert "plain" in texts
    assert sentinel.read_text() == SENTINEL + "\n"
