"""Tests for reading a panel stored in the ``uns`` and ``var`` tables of a pxl file.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import anndata
import duckdb
import numpy as np
import pandas as pd

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.anndata import add_panel_information
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset.io import PixelFileWriter
from pixelator.pna.pixeldataset.legacy import LegacyPanelAdapter


def _panel() -> PNAAntibodyPanel:
    rows = pd.DataFrame(
        {
            "marker_id": ["CD3", "CD4"],
            "control": [False, False],
            "sequence_1": ["AAAA", "CCCC"],
            "sequence_2": ["AAAA", "CCCC"],
        }
    ).set_index("marker_id")
    return PNAAntibodyPanel.from_metadata(
        rows, AntibodyPanelMetadata(name="panel", version="1.0.0", product="kit")
    )


def _adata() -> anndata.AnnData:
    return anndata.AnnData(
        X=np.zeros((1, 2)),
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["CD3", "CD4"], name="marker_id")),
    )


def _read(path: Path) -> PNAAntibodyPanel | None:
    with duckdb.connect(str(path), read_only=True) as connection:
        return LegacyPanelAdapter(connection, file_name=path.name).read()


def _write(path: Path, adata: anndata.AnnData) -> Path:
    with PixelFileWriter(path) as writer:
        writer.write_adata(adata)
    return path


def test_reads_the_panel_stored_in_uns_and_var(tmp_path):
    panel = _panel()
    path = _write(tmp_path / "a.pxl", add_panel_information(_adata(), panel))

    assert _read(path) == panel


def test_records_the_file_name_on_the_panel(tmp_path):
    path = _write(tmp_path / "a.pxl", add_panel_information(_adata(), _panel()))

    assert _read(path).filename == "a.pxl"


def test_returns_none_without_panel_metadata(tmp_path):
    path = _write(tmp_path / "a.pxl", _adata())

    assert _read(path) is None


def test_returns_none_when_panel_metadata_has_no_panel_columns(tmp_path):
    adata = _adata()
    adata.uns["panel_metadata"] = {"name": "panel", "version": "1.0.0"}
    path = _write(tmp_path / "a.pxl", adata)

    assert _read(path) is None


def test_returns_none_without_an_uns_table(tmp_path):
    path = tmp_path / "empty.pxl"
    duckdb.connect(str(path)).close()

    assert _read(path) is None
