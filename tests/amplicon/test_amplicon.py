"""Tests for amplicon module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
# pylint: disable=redefined-outer-name
# noqa: D103,D415,D400

import itertools

import numpy as np
import pyfastx
import pytest

from pixelator.amplicon.process import generate_amplicon
from pixelator.config import config, get_position_in_parent


@pytest.fixture()
def uropod_reads(data_root):
    """Paired end reads from Uropod control sample."""
    r1 = data_root / "uropod_control_300k_S1_R1_001.fastq.gz"
    r2 = data_root / "uropod_control_300k_S1_R2_001.fastq.gz"
    return r1, r2


@pytest.fixture()
def d21_150_150_reads(data_root):
    """Paired end reads from D21 amplicon.

    Each read is longer then the amplicon length.
    """
    r1 = data_root / "amplicon/D21_150_150_R1.fq.gz"
    r2 = data_root / "amplicon/D21_150_150_R2.fq.gz"
    return r1, r2


@pytest.fixture(
    params=[
        (
            "uropod_control_300k_S1_R1_001.fastq.gz",
            "uropod_control_300k_S1_R2_001.fastq.gz",
        ),
        ("amplicon/D21_150_150_R1.fq.gz", "amplicon/D21_150_150_R2.fq.gz"),
    ],
    ids=["uropod", "d21_150_150"],
)
def reads(data_root, request):
    """Parameterized fixture with paired end reads."""
    r1 = data_root / request.param[0]
    r2 = data_root / request.param[1]
    return r1, r2


def test_generate_amplicon(reads):
    """Test generating amplicons from paired end reads."""
    output_reads = []

    records_iterator = zip(
        pyfastx.Fastq(str(reads[0]), build_index=False),
        pyfastx.Fastq(str(reads[1]), build_index=False),
    )
    amplicon = config.get_assay("D21").get_region_by_id("amplicon")
    amplicon_len = amplicon.get_len()[0]

    for record1, record2 in itertools.islice(records_iterator, 5):
        r = generate_amplicon(record1, record2, amplicon)
        output_reads.append(r)

    assert len(output_reads) == 5
    for n1, s1, q1 in output_reads:
        assert len(s1) == len(q1) == amplicon_len

    pbs1_pos = get_position_in_parent(amplicon, "pbs-1")
    pbs2_pos = get_position_in_parent(amplicon, "pbs-2")

    pbs1_reg = amplicon.get_region_by_id("pbs-1")
    pbs2_reg = amplicon.get_region_by_id("pbs-2")

    for n, r, q in output_reads:
        pbs1_read_seq = r[pbs1_pos[0] : pbs1_pos[1]]
        pbs2_read_seq = r[pbs2_pos[0] : pbs2_pos[1]]

        # Allow maximum of two mismatches
        assert np.count_nonzero(pbs1_read_seq != pbs1_reg.get_sequence()) < 2
        assert np.count_nonzero(pbs2_read_seq != pbs2_reg.get_sequence()) < 2
