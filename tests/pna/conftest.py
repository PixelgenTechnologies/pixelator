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
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

from pixelator.pna.pixeldataset import read
from pixelator.mpx.pixeldataset.utils import update_metrics_anndata
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter

from tests.pna.data.pxl_data import EDGELIST_DATA, \
        ADATA_X, ADATA_OBS, ADATA_VAR, PROXIMITY_DATA, UNS_DATA, \
        LAYOUT_DATA, LAYOUT_DATA_WITH_MARKER_COUNTS

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
            "read_count": pl.UInt32,
            "uei_count": pl.UInt32,
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


def create_pxl_file(
    target,
    sample_name,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
):
    with PixelFileWriter(target) as writer:
        writer.write_edgelist(edgelist_parquet_path)
        writer.write_adata(adata_data_func())
        writer.write_metadata({"sample_name": sample_name, "version": "0.1.0"})
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
):
    target = tmp_path_factory.mktemp("data") / "file.pxl"
    target = create_pxl_file(
        target=target,
        sample_name="test_sample",
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
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


@pytest.fixture(name="uns_data", scope="module")
def uns_data_fixture():
    return UNS_DATA


def adata_data_func():
    X = pd.read_csv(StringIO(ADATA_X), index_col="component")

    adata = AnnData(
        X=X,
        obs=pd.read_csv(StringIO(ADATA_OBS), index_col="component"),
        var=pd.read_csv(StringIO(ADATA_VAR), index_col=0),
        uns=UNS_DATA,
    )
    adata = update_metrics_anndata(adata, inplace=False)
    return adata


@pytest.fixture(name="adata_data", scope="function")
def adata_data_fixture():
    return adata_data_func()
