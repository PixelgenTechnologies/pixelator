"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
import pytest
import xopen

from pixelator.pna.amplicon.process import amplicon_fastq
from pixelator.pna.config import pna_config


def _match_amplicon_reads(r1, r2):
    r1_names = [r[:-2] for r in r1[0::4]]
    r2_names = [r[:-2] for r in r2[0::4]]
    common_reads = set(r1_names).intersection(set(r2_names))
    distances = {}
    for r in common_reads:
        index_1 = next(i for i, s in enumerate(r1_names) if r in s)
        index_2 = next(i for i, s in enumerate(r2_names) if r in s)
        r1_seq = r1[index_1 * 4 + 1]
        r2_seq = r2[index_2 * 4 + 1]
        distances[r] = sum([b != a for a, b in zip(r1_seq, r2_seq)])
    return distances


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
    # Testing paired-end vs single-end amplicon where one read (R2) has higher read quality.
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

    def read_lines(path):
        with xopen.xopen(path, "rt") as f:
            return f.read().splitlines()

    paired_lines = read_lines(out_paired)
    r1_lines = read_lines(out_r1)
    r2_lines = read_lines(out_r2)

    min_diffs_r1 = _match_amplicon_reads(r1_lines, paired_lines)
    min_diffs_r2 = _match_amplicon_reads(r2_lines, paired_lines)

    assert np.mean([m > 6 for m in min_diffs_r1.values()]) < 0.1
    assert np.mean([m > 6 for m in min_diffs_r2.values()]) < 0.1

    # Verify that R2 matches the paired read more closely than R1
    assert np.mean(list(min_diffs_r2.values())) < np.mean(list(min_diffs_r1.values()))
