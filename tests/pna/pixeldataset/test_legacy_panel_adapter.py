"""Tests for reading the panel of pxl files that predate the panel tables.

Copyright © 2026 Pixelgen Technologies AB.
"""

import json
from pathlib import Path

import anndata
import duckdb
import numpy as np
import pandas as pd

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile
from pixelator.pna.pixeldataset.legacy import LegacyPanelAdapter
from tests.common.data_generator.pixelfile import with_legacy_panel_columns

PANEL_NAME = "proxiome-v1-immuno-155-v1.0"


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
    path = _write(tmp_path / "a.pxl", with_legacy_panel_columns(_adata(), panel))

    assert _read(path) == panel


def test_records_the_file_name_on_the_panel(tmp_path):
    path = _write(tmp_path / "a.pxl", with_legacy_panel_columns(_adata(), _panel()))

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


def _write_pxl_without_panel(path: Path, metadata: dict, uns: dict | None = None):
    """Write a minimal pxl file with neither panel tables nor panel columns."""
    with PixelFileWriter(path) as writer:
        writer.write_metadata(metadata)
        if uns is not None:
            connection = writer.get_connection()
            connection.execute("CREATE TABLE __adata__uns (value JSON)")
            connection.execute("INSERT INTO __adata__uns VALUES (?)", [json.dumps(uns)])
            connection.execute("CREATE TABLE __adata__var (index VARCHAR)")


def test_panel_is_resolved_from_the_panel_name_of_pre_0_22_files(pna_pxl_file):
    # this file has neither panel tables nor uns['panel_metadata']
    panel = PxlFile(pna_pxl_file).read_panel()

    assert panel == pna_config.get_panel(PANEL_NAME)


def test_the_recorded_panel_version_is_ignored(tmp_path):
    path = tmp_path / "old.pxl"
    _write_pxl_without_panel(path, {"panel_name": PANEL_NAME, "panel_version": "0.1.0"})

    panel = PxlFile(path).read_panel()

    assert panel.version != "0.1.0"


def test_from_pxl_dataset_uses_the_adapter_for_older_files(pna_pxl_file):
    panel = PNAAntibodyPanel.from_pxl_dataset(read(pna_pxl_file))

    assert panel.name == PANEL_NAME


def test_unknown_panel_name_yields_no_panel(tmp_path):
    path = tmp_path / "old.pxl"
    _write_pxl_without_panel(path, {"panel_name": "not-a-panel"})

    assert PxlFile(path).read_panel() is None


def test_file_without_panel_name_yields_no_panel(tmp_path):
    path = tmp_path / "old.pxl"
    _write_pxl_without_panel(path, {"sample_name": "x"})

    assert PxlFile(path).read_panel() is None


def test_panel_metadata_without_panel_columns_falls_back_to_the_panel_name(tmp_path):
    path = tmp_path / "old.pxl"
    _write_pxl_without_panel(
        path,
        {"panel_name": PANEL_NAME},
        uns={"panel_metadata": {"name": "x", "version": "1.0.0"}},
    )

    panel = PxlFile(path).read_panel()

    assert panel is not None
    assert panel.name == PANEL_NAME


def test_adapter_looks_panels_up_in_the_given_config(pna_pxl_file):
    class EmptyConfig:
        def get_panel(self, name):
            return None

    with duckdb.connect(str(pna_pxl_file), read_only=True) as connection:
        assert LegacyPanelAdapter(connection, config=EmptyConfig()).read() is None
