"""Copyright © 2025 Pixelgen Technologies AB."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def demux_m1_tmp_parquet_data(pna_data_root):
    """Demux m1 tmp parquet data.

    Args:
    pna_data_root: pna data root.

    """
    return (
        pna_data_root
        / "intermediate-demux-results/tmp"
        / "PNA055_Sample07_filtered_S7.demux.m1.part_000.parquet"
    )


@pytest.fixture()
def demux_m2_tmp_parquet_data(pna_data_root):
    """Demux m2 tmp parquet data.

    Args:
    pna_data_root: pna data root.

    """
    return (
        pna_data_root
        / "intermediate-demux-results/tmp"
        / "PNA055_Sample07_filtered_S7.demux.m2.part_000.parquet"
    )


@pytest.fixture()
def demux_intermediary_dir(
    tmpdir, demux_m1_tmp_parquet_data, demux_m2_tmp_parquet_data
) -> Path:
    """Demux intermediary dir.

    Args:
    tmpdir: tmpdir.
    demux_m1_tmp_parquet_data: demux m1 tmp parquet data.
    demux_m2_tmp_parquet_data: demux m2 tmp parquet data.

    """
    shutil.copy(demux_m1_tmp_parquet_data, tmpdir)
    shutil.copy(demux_m2_tmp_parquet_data, tmpdir)
    return Path(tmpdir)
