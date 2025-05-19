"""Copyright © 2025 Pixelgen Technologies AB."""

import itertools

import pandas as pd
import polars as pl
import pytest

from pixelator.common.utils import create_output_stage_dir
from pixelator.pna.config import pna_config
from pixelator.pna.demux import (
    correct_marker_barcodes,
    demux_barcode_groups,
    finalize_batched_groups,
)


@pytest.mark.slow
def test_demux_writing_strategy_paired(tmp_path, testdata_amplicon_fastq):
    input_file = testdata_amplicon_fastq
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    strategy = "paired"

    input_files = [input_file]
    demux_output = create_output_stage_dir(tmp_path, "demux")
    threads = 2

    stats, corrected, failed = correct_marker_barcodes(
        input=input_files[0],
        assay=assay,
        panel=panel,
        output=demux_output,
        save_failed=True,
        threads=threads,
    )

    demux_barcode_groups(
        corrected_reads=corrected,
        assay=assay,
        panel=panel,
        stats=stats,
        output_dir=demux_output,
        threads=threads,
        reads_per_chunk=10_000,
        stategy=strategy,
    )

    finalize_batched_groups(
        input_dir=demux_output,
        output_dir=demux_output,
        strategy=strategy,
        remove_intermediates=False,
    )

    output_reads = stats.as_json()["output_reads"]

    files = demux_output.glob("*.parquet")
    data = [pd.read_parquet(f) for f in files]
    tbl = pd.concat(data)

    assert output_reads == tbl["read_count"].sum()


def verify_demuxed_groups(files, marker_col):
    """Helper function to verify that there are no common markers between partitions"""
    marker_sets = {}
    for f in files:
        df = pd.read_parquet(f)
        marker_sets[f.name] = set(df[marker_col].unique())

    for subset in itertools.combinations(marker_sets.values(), 2):
        res = subset[0].intersection(subset[1])
        assert len(res) == 0


@pytest.mark.slow
def test_demux_writing_independent(tmp_path, testdata_amplicon_fastq):
    input_file = testdata_amplicon_fastq
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    strategy = "independent"

    input_files = [input_file]
    demux_output = create_output_stage_dir(tmp_path, "demux")
    threads = 2

    stats, corrected, failed = correct_marker_barcodes(
        input=input_files[0],
        assay=assay,
        panel=panel,
        output=demux_output,
        save_failed=True,
        threads=threads,
    )

    demux_barcode_groups(
        corrected_reads=corrected,
        assay=assay,
        panel=panel,
        stats=stats,
        output_dir=demux_output / "tmp",
        threads=threads,
        reads_per_chunk=50_000,
        max_chunks=8,
        stategy=strategy,
    )

    finalize_batched_groups(
        input_dir=demux_output / "tmp",
        output_dir=demux_output,
        strategy=strategy,
        remove_intermediates=False,
        memory=1 * 2**30,
    )

    output_reads = stats.as_json()["output_reads"]

    files = list(demux_output.glob("*.parquet"))

    # Verify that there are no common marker per partition file
    m1_files = sorted([f for f in files if "m1" in f.name])
    m2_files = sorted([f for f in files if "m2" in f.name])

    verify_demuxed_groups(m1_files, "marker_1")
    verify_demuxed_groups(m2_files, "marker_2")

    # Independent demuxing will duplicate each molecule once in the partitions for marker1
    # and once in the partitions for marker2. So when summing all read counts together
    # it will be twice the actual output reads.
    m1_df = pl.scan_parquet(m1_files)
    m2_df = pl.scan_parquet(m2_files)

    m1_reads = m1_df.select(pl.col("read_count")).sum().collect().item()
    m2_reads = m2_df.select(pl.col("read_count")).sum().collect().item()

    assert m1_reads == m2_reads
    assert m1_reads == output_reads
    assert m2_reads == output_reads
