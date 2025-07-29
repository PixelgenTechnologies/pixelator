"""
Configuration and shared files/objects for the testing framework

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
import os
import tempfile

# pylint: disable=redefined-outer-name
from pathlib import Path

import pytest

from pixelator.pna.pixeldataset import read

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
