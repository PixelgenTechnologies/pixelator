"""Copyright © 2025 Pixelgen Technologies AB."""

import pytest

from pixelator.pna.amplicon.process import amplicon_fastq
from pixelator.pna.config import pna_config


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
def test_amplicon_300k_single_end(tmp_path, testdata_300k):
    input_files = testdata_300k
    pna_assay = pna_config.get_assay("pna-2")

    amplicon_fastq(
        inputs=input_files[0:1],
        assay=pna_assay,
        output=tmp_path / "testdata_300k.fq.zst",
    )

    assert (tmp_path / "testdata_300k.fq.zst").exists()
