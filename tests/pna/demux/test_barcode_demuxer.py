"""Copyright © 2025 Pixelgen Technologies AB."""

import dnaio
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pixelator.common.utils import create_output_stage_dir
from pixelator.pna.config import pna_config
from pixelator.pna.demux import correct_marker_barcodes, finalize_batched_groups
from pixelator.pna.demux.barcode_demuxer import (
    DemuxRecordBatch,
    IndependentBarcodeDemuxer,
    create_barcode_group_to_batch_mapping,
    independent_marker_groups_mapping,
)
from pixelator.pna.demux.barcode_identifier import BarcodeIdentifierStatistics


@pytest.fixture
def barcode_marker_group_sizes():
    return {
        ("A", "B"): 100,
        ("A", "C"): 200,
        ("B", "C"): 300,
        ("B", "D"): 400,
        ("B", "A"): 20,
        ("C", "D"): 500,
        ("C", "A"): 100,
    }


def test_create_barcode_group_to_batch_mapping(barcode_marker_group_sizes):
    res = create_barcode_group_to_batch_mapping(
        barcode_marker_group_sizes, reads_per_chunk=500
    )

    assert res == {
        ("A", "B"): 0,
        ("A", "C"): 1,
        ("B", "C"): 2,
        ("B", "D"): 3,
        ("B", "A"): 3,
        ("C", "D"): 2,
        ("C", "A"): 3,
    }


def test_create_markers_to_batch_mapping(barcode_marker_group_sizes):
    m1, m2 = independent_marker_groups_mapping(barcode_marker_group_sizes, 1000)

    assert m1 == {"A": 0, "B": 1, "C": 1}
    assert m2 == {"A": 3, "B": 2, "C": 3, "D": 3}


def test_demux_record_batch():
    batch = DemuxRecordBatch(capacity=10)
    batch.add_record(1, 0, "abc".encode("ascii"))
    batch.add_record(0, 1, "abc".encode("ascii"))

    assert len(batch) == 2
    assert batch.capacity() == 10

    arrow_batch = batch.to_arrow()
    assert arrow_batch.shape == (2, 3)

    batch.clear()
    assert len(batch) == 0

    arrow_batch = batch.to_arrow()
    assert arrow_batch.shape == (0, 3)


def test_independent_demuxing(testdata_demux_passed_reads):
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")

    marker_counts = {
        ("CD18", "CD45"): 1,
        ("CD31", "HLA-ABC"): 1,
        ("CD45RA", "CD16"): 2,
        ("CD357", "CD18"): 1,
        ("CD41", "CD36"): 1,
        ("CD18", "CD18"): 1,
        ("mIgG1", "CD19"): 1,
        ("CD33", "CD7"): 4,
        ("CD169", "CD159c"): 1,
        ("CD43", "CD59"): 1,
        ("CD55", "CD41"): 1,
        ("CD36", "CD9"): 2,
        ("CD226", "CD141"): 1,
        ("CD328", "CD44"): 2,
        ("CD13", "CX3CR1"): 1,
        ("CD41", "CD159c"): 1,
        ("CD89", "CD66b"): 1,
        ("CD269", "CD10"): 1,
        ("CD37", "CD85j"): 1,
        ("CD352", "CD45RA"): 1,
        ("CD45", "CD16"): 3,
        ("TCRVB5", "CD39"): 1,
        ("CD45RB", "CD80"): 1,
        ("CD82", "CD35"): 1,
        ("mIgG2a", "CD5"): 2,
        ("CD26", "CD5"): 2,
        ("CD66b", "B2M"): 1,
        ("HLA-ABC", "CD102"): 1,
        ("Siglec-9", "CD82"): 1,
        ("CD41", "CD102"): 2,
        ("CD191", "CD138"): 1,
        ("HLA-DR-DP-DQ", "CD72"): 2,
        ("CD16", "CD16"): 1,
        ("CD209", "CD41"): 1,
        ("CD95", "CD18"): 1,
        ("CD199", "CD319"): 2,
        ("CD35", "CD18"): 1,
        ("CD94", "CD199"): 1,
        ("CD21", "CD20"): 3,
        ("CD162", "CD366"): 1,
        ("CD102", "HLA-ABC"): 3,
        ("B2M", "CD29"): 1,
        ("CD169", "CD89"): 4,
        ("CD36", "HLA-DR"): 1,
        ("TCRab", "CD45RA"): 1,
        ("CD18", "CD11c"): 1,
        ("CD305", "mIgG2b"): 2,
        ("HLA-ABC", "CD59"): 2,
        ("CD13", "CD89"): 2,
        ("CD89", "CD16"): 1,
        ("HLA-ABC", "CD89"): 1,
        ("HLA-DR", "CD319"): 1,
        ("CD58", "CD29"): 4,
        ("HLA-ABC", "CD36"): 1,
        ("CD16", "CD47"): 1,
        ("HLA-ABC", "CD9"): 1,
        ("CD59", "CD24"): 2,
        ("B2M", "CD59"): 3,
        ("CD102", "CD9"): 3,
        ("CD24", "HLA-ABC"): 2,
        ("B2M", "CD64"): 1,
        ("CD41", "CD90"): 1,
        ("CD117", "CD58"): 1,
        ("IgE", "CD19"): 1,
        ("CD314", "CD302"): 1,
        ("CD62P", "CD41"): 1,
        ("CD43", "CD45"): 1,
        ("CD29", "HLA-DR-DP-DQ"): 1,
    }

    m1_group_map, m2_group_map = independent_marker_groups_mapping(marker_counts, 10)
    barcode_demuxer = IndependentBarcodeDemuxer(
        assay=assay,
        panel=panel,
        marker1_groups=m1_group_map,
        marker2_groups=m2_group_map,
    )

    chunks = []
    # only check the first 100 reads
    for read in dnaio.open(testdata_demux_passed_reads):
        r = barcode_demuxer(read)
        if r:
            chunks.extend(r)

    # Default
    expected_sizes = {
        0: 13,
        8: 10,
        1: 7,
        9: 19,
        2: 10,
        10: 12,
        3: 13,
        11: 13,
        4: 17,
        12: 10,
        5: 9,
        13: 11,
        6: 13,
        14: 16,
        7: 18,
        15: 9,
    }

    for k, v in barcode_demuxer._output_groups_buffer.items():
        assert len(v) == expected_sizes[k]


def is_sorted(x: npt.NDArray[np.integer]) -> bool:
    return bool(np.all(x[:-1] <= x[1:]))  # type: ignore


def test_finalize_batched_groups_independent(demux_intermediary_dir):
    (demux_intermediary_dir / "dedup").mkdir()
    res = finalize_batched_groups(
        demux_intermediary_dir, demux_intermediary_dir / "dedup", strategy="independent"
    )

    assert len(res) == 2

    for f in res:
        assert f.exists()

    res.sort()

    # Check if the parquet files are properly sorted

    m1_file = res[0]
    m2_file = res[1]

    df1 = pd.read_parquet(m1_file)
    assert is_sorted(df1["marker_1"].to_numpy())

    df2 = pd.read_parquet(m2_file)
    assert is_sorted(df2["marker_2"].to_numpy())


def test_barcode_identifier_statistics_accumulation():
    acc = BarcodeIdentifierStatistics()
    acc.corrected = 3094
    acc.exact = 128662
    acc.missing_pid1 = 897
    acc.missing_pid2 = 1794
    acc.missing_pid1_pid2 = 8

    assert acc.input == 134455

    ch = BarcodeIdentifierStatistics()
    ch.corrected = 1396
    ch.exact = 68605
    ch.missing_pid1 = 425
    ch.missing_pid2 = 684
    ch.missing_pid1_pid2 = 4
    ch.n_in_umi1 = 0
    ch.n_in_umi2 = 0

    assert ch.input == 71114
    assert ch.failed == 1113

    acc += ch

    assert acc.input == 205569
    assert acc.failed == 2699 + 1113


@pytest.mark.slow
def test_marker_correction_pipeline(tmp_path, testdata_amplicon_fastq):
    input_file = testdata_amplicon_fastq
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")

    input_files = [input_file]
    demux_output = create_output_stage_dir(tmp_path, "demux")
    threads = 6

    stats, corrected, failed = correct_marker_barcodes(
        input=input_files[0],
        assay=assay,
        panel=panel,
        output=demux_output,
        save_failed=True,
        threads=threads,
    )

    assert (demux_output / "PNA055_Sample07_filtered_S7.failed.fq.zst").exists()
