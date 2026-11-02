"""Graph-collapse rebuilds counts from the markers already in the pxl file.

Copyright © 2026 Pixelgen Technologies AB.
"""

import shutil

import duckdb
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna import read
from pixelator.pna.pixeldataset.io import PixelFileWriter


def test_graph_collapse_does_not_restore_markers_removed_upstream(
    graph_pxl_file, tmp_path
):
    """A marker absent from var (e.g. a hashing clone) stays absent after the collapse."""
    input_path = tmp_path / "sampleA.pxl"
    shutil.copy(graph_pxl_file, input_path)

    adata = read(input_path).adata()
    removed = str(adata.var_names[0])
    with duckdb.connect(str(input_path)) as con:
        con.execute(
            "DELETE FROM edgelist WHERE marker_1 = ? OR marker_2 = ?",
            [removed, removed],
        )
    with PixelFileWriter(input_path) as writer:
        writer.write_adata(adata[:, adata.var_names != removed].copy())
    input_markers = list(read(input_path).adata().var_names)
    assert removed not in input_markers

    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph-collapse",
            str(input_path),
            "--output",
            str(output),
            "--design",
            "proxiome-v1",
        ],
    )
    assert result.exit_code == 0, result.output

    output_adata = read(
        output / "graph-collapse" / "sampleA.graph-collapse.pxl"
    ).adata()
    assert list(output_adata.var_names) == input_markers


def test_graph_collapse_rejects_files_without_a_stored_panel(pna_pxl_file, tmp_path):
    """A file that only records a panel name is not resolved by name."""
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph-collapse",
            str(pna_pxl_file),
            "--output",
            str(tmp_path),
            "--design",
            "proxiome-v1",
        ],
    )
    assert isinstance(result.exception, KeyError)
    assert "regenerate your data" in str(result.exception)
