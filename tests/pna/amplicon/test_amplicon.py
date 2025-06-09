"""Copyright © 2025 Pixelgen Technologies AB."""

import pytest

from pixelator.pna.amplicon.process import amplicon_fastq
from pixelator.pna.config import pna_config
import zstandard as zstd
import numpy as np


@pytest.mark.slow
def test_amplicon_300k(tmp_path, testdata_300k):
    input_files = testdata_300k
    pna_assay = pna_config.get_assay("pna-2")

    amplicon_fastq(
        inputs=input_files,
        assay=pna_assay,
        output=tmp_path / "testdata_300k.fq.zst",
    )

    assert (tmp_path / "testdata_300k.fq.zst").exists()


@pytest.mark.slow
def test_amplicon_unbalanced_single_end(tmp_path, testdata_unbalanced_r12):
    input_files = testdata_unbalanced_r12
    pna_assay = pna_config.get_assay("pna-2")

    out_paired = tmp_path / "testdata_unbalanced.fq.zst"
    out_r1 = tmp_path / "testdata_unbalanced_r1.fq.zst"
    out_r2 = tmp_path / "testdata_unbalanced_r2.fq.zst"

    amplicon_fastq(
        inputs=input_files,
        assay=pna_assay,
        output=out_paired,
    )

    amplicon_fastq(
        inputs=input_files[0:1],
        assay=pna_assay,
        output=out_r1,
    )

    amplicon_fastq(
        inputs=input_files[1:],
        assay=pna_assay,
        output=out_r2,
    )

    assert out_paired.exists()
    assert out_r1.exists()
    assert out_r2.exists()

    def read_zst_lines(path):
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                return reader.read().splitlines()

    paired_lines = read_zst_lines(out_paired)
    r1_lines = read_zst_lines(out_r1)
    r2_lines = read_zst_lines(out_r2)

    min_diffs_r1 = []
    min_diffs_r2 = []
    for p in paired_lines[1::4]:
        min_diff_r1 = 142
        min_diff_r2 = 142
        for p1 in r1_lines[1::4]:
            diff = sum([b != a for a, b in zip(p, p1)])
            if diff < min_diff_r1:
                min_diff_r1 = diff
            if min_diff_r1 == 0:
                break
        for p2 in r2_lines[1::4]:
            diff = sum([b != a for a, b in zip(p, p2)])
            if diff < min_diff_r2:
                min_diff_r2 = diff
            if min_diff_r2 == 0:
                break
        min_diffs_r1.append(min_diff_r1)
        min_diffs_r2.append(min_diff_r2)

    assert np.mean([m > 6 for m in min_diffs_r1]) < 0.1
    assert np.mean([m > 6 for m in min_diffs_r2]) < 0.1
