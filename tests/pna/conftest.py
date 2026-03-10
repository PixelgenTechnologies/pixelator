"""
Configuration and shared files/objects for the testing framework

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
import os
import tempfile
from io import StringIO

# pylint: disable=redefined-outer-name
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import PNAPixelDataset, read
from pixelator.pna.pixeldataset.io import PixelFileWriter
from tests.pna.data.pxl_data import (
    EDGELIST_DATA,
    LAYOUT_DATA,
    PROXIMITY_DATA,
    TEST_PANEL,
)

PNA_DATA_ROOT = Path(__file__).parent / "data"

# We need to add basic logging config here to make sure
# integration tests output logs to stdout
logging.basicConfig(level=logging.INFO)
logging.getLogger("pixelator").setLevel(logging.DEBUG)


@pytest.fixture(name="pna_data_root", scope="module")
def pna_data_root_fixture():
    return PNA_DATA_ROOT


@pytest.fixture()
def run_in_tmpdir():
    """Run a function in a temporary directory."""
    old_cwd = Path.cwd()

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        yield d
        os.chdir(old_cwd)


@pytest.fixture(name="pna_pxl_file", scope="module")
def pna_pxl_file_fixture(pna_data_root):
    """Load an example pna pixel from disk."""
    return pna_data_root / "PNA055_Sample07_S7.layout.pxl"


@pytest.fixture(name="pna_pxl_dataset", scope="module")
def pna_pxl_dataset_fixture(pna_pxl_file):
    """Load an example pna pixel from disk."""
    pixel = read(pna_pxl_file)
    return pixel


@pytest.fixture(name="pna_pxl_panel_dataset", scope="module")
def pna_pxl_panel_dataset_fixture(pna_data_root):
    return read(pna_data_root / "pxl_file_with_panel.pxl")


@pytest.fixture(scope="session")
def full_run_dir() -> Path:
    return Path(PNA_DATA_ROOT) / "full_run"


@pytest.fixture(scope="module")
def testdata_300k(pna_data_root) -> tuple[Path, Path]:
    return (
        Path(pna_data_root / "PNA055_Sample07_300k_S7_R1_001.fastq.gz"),
        Path(pna_data_root / "PNA055_Sample07_300k_S7_R2_001.fastq.gz"),
    )


@pytest.fixture(scope="module")
def testdata_unbalanced_r12(pna_data_root) -> tuple[Path, Path]:
    return (
        Path(pna_data_root / "unbalanced_R1.fastq.gz"),
        Path(pna_data_root / "unbalanced_R2.fastq.gz"),
    )


@pytest.fixture(scope="module")
def testdata_amplicon_fastq(full_run_dir) -> Path:
    p = full_run_dir / "amplicon" / "PNA055_Sample07_filtered_S7.amplicon.fq.zst"
    return p


@pytest.fixture(scope="module")
def testdata_paired_small_demux(pna_data_root) -> Path:
    p = Path(
        pna_data_root
        / "paired-demux-results"
        / "PNA055_Sample07_filtered_S7.demux.part_000.parquet"
    )
    return p


@pytest.fixture(scope="module")
def testdata_demux_passed_reads(pna_data_root) -> Path:
    p = pna_data_root / "PNA055_Sample07_filtered_S7.100_reads.demux.passed.fq"
    return p


@pytest.fixture(name="layout_dataframe", scope="module")
def layout_dataframe_fixture():
    layout = pl.read_csv(
        StringIO(LAYOUT_DATA),
        schema={
            "component": pl.Utf8,
            "layout": pl.Utf8,
            "projection": pl.Utf8,
            "index": pl.UInt64,
            "pixel_type": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "x_norm": pl.Float64,
            "y_norm": pl.Float64,
            "z_norm": pl.Float64,
        },
    )
    return layout


@pytest.fixture(name="layout_parquet_path", scope="module")
def layout_parquet_path_fixture(tmp_path_factory, layout_dataframe):
    path = tmp_path_factory.mktemp("data") / "layout.parquet"
    layout_dataframe.write_parquet(path)
    return path


@pytest.fixture(name="edgelist_dataframe", scope="module")
def edgelist_dataframe_fixture(edgelist_data):
    edgelist = pl.read_csv(
        StringIO(edgelist_data),
        schema={
            "umi1": pl.UInt64,
            "umi2": pl.UInt64,
            "read_count": pl.UInt64,
            "uei_count": pl.UInt64,
            "marker_1": pl.Utf8,
            "marker_2": pl.Utf8,
            "component": pl.Utf8,
        },
    )
    return edgelist


@pytest.fixture(name="edgelist_parquet_path", scope="module")
def edgelist_parquet_path_fixture(tmp_path_factory, edgelist_dataframe):
    path = tmp_path_factory.mktemp("data") / "edgelist.parquet"
    edgelist_dataframe.write_parquet(path)
    return path


@pytest.fixture(name="proximity_dataframe", scope="module")
def proximity_dataframe_fixture(proximity_data):
    proximity = pl.read_csv(
        StringIO(proximity_data),
    )
    return proximity


@pytest.fixture(name="proximity_parquet_path", scope="module")
def proximity_parquet_path_fixture(tmp_path_factory, proximity_dataframe):
    path = tmp_path_factory.mktemp("data") / "proximity.parquet"
    proximity_dataframe.write_parquet(path)
    return path


@pytest.fixture(name="panel", scope="module")
def panel_fixture():
    panel_df = pd.read_csv(StringIO(TEST_PANEL)).set_index("marker_id")

    panel_df = panel_df
    panel_df["uniprot_id"] = panel_df["uniprot_id"].fillna("")
    panel_df["control"] = (
        panel_df["control"].astype(str).map(lambda s: s.lower() == "yes")
    )

    return PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="test-pna-panel",
            version="0.1.0",
            aliases=["test-pna"],
            description="Test R&D panel for RNA",
        ),
    )


def create_pxl_file(
    target,
    sample_name,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
    panel,
):
    with PixelFileWriter(target) as writer:
        writer.write_edgelist(edgelist_parquet_path)
        con = writer.get_connection()
        adata = pna_edgelist_to_anndata(con, panel=panel)
        writer.write_adata(adata)
        writer.write_metadata(
            {
                "sample_name": sample_name,
                "version": "0.1.0",
                "panel_name": "custom_panel",
            }
        )
        if proximity_parquet_path:
            writer.write_proximity(proximity_parquet_path)
        if layout_parquet_path:
            writer.write_layouts(layout_parquet_path)

    return target


@pytest.fixture(
    name="pxl_file",
    scope="module",
)
def pixel_file_fixture(
    tmp_path_factory,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
    panel,
):
    target = tmp_path_factory.mktemp("data") / "file.pxl"
    target = create_pxl_file(
        target=target,
        sample_name="test_sample",
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
        panel=panel,
    )
    return target


@pytest.fixture(name="pxl_dataset", scope="function")
def pixel_dataset_fixture(pxl_file):
    return PNAPixelDataset.from_pxl_files(pxl_file)


@pytest.fixture(name="edgelist_data", scope="module")
def edgelist_data_fixture():
    return EDGELIST_DATA


@pytest.fixture(name="proximity_data", scope="module")
def proximity_data_fixture():
    return PROXIMITY_DATA


@pytest.fixture(name="adata_data", scope="function")
def adata_data_fixture(edgelist_parquet_path, panel):
    with duckdb.connect() as con:
        con.execute(f"""
                    CREATE TABLE edgelist AS SELECT *
                    FROM read_parquet('{edgelist_parquet_path}')
                    """)
        adata = pna_edgelist_to_anndata(con, panel=panel)

    return adata
