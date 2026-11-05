"""Tests for the pipeline test data generator.

Copyright © 2026 Pixelgen Technologies AB.
"""

import gzip
import hashlib
import json

from pixelator.testing.test_data import main


def _content_md5(path):
    return hashlib.md5(gzip.open(path).read()).hexdigest()


def test_sample1_matches_published_data(tmp_path):
    """The paired sample1 reads match the published test data.

    The hashes are of the decompressed published ``sample1_R1/R2.fastq.gz``
    files; the gzip headers differ only by their mtime.
    """
    main(
        [
            "--output-dir",
            str(tmp_path),
            "--libraries",
            "sample1",
            "--sequencing",
            "paired",
            "single-end",
        ]
    )
    assert _content_md5(tmp_path / "sample1_R1.fastq.gz") == (
        "d99f71c92d59341ce4e94cccfd2c6d5b"
    )
    assert _content_md5(tmp_path / "sample1_R2.fastq.gz") == (
        "b455a1e3386c86eca860545b79c5e2b0"
    )

    metadata = json.loads((tmp_path / "sample1_single_end.metadata.json").read_text())
    assert metadata["sample_name"] == "sample1_single_end"
    assert metadata["edgelist"] == "sample1"
    assert metadata["edgelist_seed"] == 3
    assert metadata["n_reads"] == 21800
    assert metadata["sequencing"] == "single_end"
    with gzip.open(tmp_path / "sample1_single_end.fastq.gz", "rt") as fh:
        assert sum(1 for _ in fh) == 4 * 21800
