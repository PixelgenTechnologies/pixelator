"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="module")
def m1_demuxed_data_part0(pna_data_root) -> Path:
    return (
        pna_data_root
        / "intermediate-demux-results"
        / "PNA055_Sample07_filtered_S7.demux.m1.part_000.parquet"
    )


@pytest.fixture(scope="module")
def m2_demuxed_data_part0(pna_data_root) -> Path:
    return (
        pna_data_root
        / "intermediate-demux-results"
        / "PNA055_Sample07_filtered_S7.demux.m2.part_000.parquet"
    )


@pytest.fixture(scope="module")
def m1_demuxed_data(pna_data_root):
    return list(
        (pna_data_root / "intermediate-demux-results").glob(
            "PNA055_Sample07_filtered_S7.demux.m1.part_*.parquet"
        )
    )


@pytest.fixture(scope="module")
def m2_demuxed_data(pna_data_root):
    return list(
        (pna_data_root / "intermediate-demux-results").glob(
            "PNA055_Sample07_filtered_S7.demux.m2.part_*.parquet"
        )
    )


@pytest.fixture(scope="module")
def m1_collapsed_data(full_run_dir):
    return list(
        (full_run_dir / "collapse").glob(
            "PNA055_Sample07_filtered_S7.collapse.m1.part_*.parquet"
        )
    )


@pytest.fixture(scope="module")
def m2_collapsed_data(full_run_dir):
    return list(
        (full_run_dir / "collapse").glob(
            "PNA055_Sample07_filtered_S7.collapse.m2.part_*.parquet"
        )
    )


@pytest.fixture(scope="module")
def m1_collapsed_report(full_run_dir):
    return (
        full_run_dir
        / "collapse"
        / "PNA055_Sample07_filtered_S7.collapse.m1.part_000.report.json"
    )


@pytest.fixture(scope="module")
def m2_collapsed_report(full_run_dir):
    return (
        full_run_dir
        / "collapse"
        / "PNA055_Sample07_filtered_S7.collapse.m2.part_000.report.json"
    )


@pytest.fixture(scope="module")
def umi1_partition(m1_demuxed_data_part0):
    df = pl.read_parquet(m1_demuxed_data_part0)
    # Partition the data by the marker1 and marker2 columns and store each partition as a separate DataFrame
    partitions = df.partition_by("marker_1", as_dict=True, include_key=True)
    data = next(iter(partitions.values()))
    return data
