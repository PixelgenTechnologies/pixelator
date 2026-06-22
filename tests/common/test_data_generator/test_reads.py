"""Tests for fastq read generation.

Copyright © 2026 Pixelgen Technologies AB.
"""

import gzip

import numpy as np
import pytest

from pixelator.common.utils.test_data_generator.reads import (
    _add_substitutions,
    _assemble_amplicons,
    _decode_2bit_dna,
    _random_qualities,
    _reverse_complement,
    _write_fastq,
    _write_paired_reads,
    to_fastq,
)

# Amplicon length for proxiome-v2 with the length-10 marker_panel sequences:
# umi1(28) + pid1(10) + lbs1(28) + uei(15) + lbs2(14) + pid2(10) + umi2(28).
_AMPLICON_LENGTH = 133


def _read_fastq(path):
    """Parse a gzipped fastq file into a list of (header, sequence, quality)."""
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    records = []
    for i in range(0, len(lines), 4):
        assert lines[i].startswith("@")
        assert lines[i + 2] == "+"
        records.append((lines[i][1:], lines[i + 1], lines[i + 3]))
    return records


def test_decode_2bit_dna_values():
    """Decoding is little-endian: the lowest 2 bits encode the first base."""
    assert _decode_2bit_dna(np.array([0b00000000]), 4) == ["AAAA"]
    assert _decode_2bit_dna(np.array([0b11100100]), 4) == ["ACGT"]


def test_decode_2bit_dna_truncation():
    """Bits beyond the requested length are ignored."""
    assert _decode_2bit_dna(np.array([0b11111111]), 2) == ["TT"]


def test_decode_2bit_dna_vectorized():
    """Decoding maps over many values and respects the requested length."""
    out = _decode_2bit_dna(np.array([0, 1, 2, 3]), 3)
    assert out == ["AAA", "CAA", "GAA", "TAA"]
    assert all(len(s) == 3 for s in out)
    assert set("".join(out)) <= set("ACGT")


def test_reverse_complement_known():
    """Reverse complement of known sequences."""
    assert _reverse_complement("AAAC") == "GTTT"
    assert _reverse_complement("ATCG") == "CGAT"


def test_reverse_complement_involution():
    """Applying reverse complement twice returns the original sequence."""
    sequence = "ACGTTGCAA"
    assert _reverse_complement(_reverse_complement(sequence)) == sequence
    assert len(_reverse_complement(sequence)) == len(sequence)


def test_add_substitutions_rate_zero():
    """A zero substitution rate leaves the sequences unchanged."""
    sequences = ["ACGTACGT", "TTTTAAAA"]
    assert _add_substitutions(sequences, 0.0, np.random.default_rng(0)) == sequences


def test_add_substitutions_rate_one():
    """A substitution rate of one changes every base to a different base."""
    sequences = ["ACGTACGT", "TTTTAAAA"]
    out = _add_substitutions(sequences, 1.0, np.random.default_rng(0))
    assert len(out) == len(sequences)
    for original, mutated in zip(sequences, out):
        assert len(mutated) == len(original)
        assert all(o != m for o, m in zip(original, mutated))
        assert set(mutated) <= set("ACGT")


def test_add_substitutions_reproducible():
    """The same seed yields the same substitutions."""
    sequences = ["ACGTACGTACGT"] * 5
    first = _add_substitutions(sequences, 0.3, np.random.default_rng(0))
    assert first == _add_substitutions(sequences, 0.3, np.random.default_rng(0))


def test_random_qualities_shape_and_range():
    """Quality strings have the requested shape and stay within the clip range."""
    out = _random_qualities(5, 12, np.random.default_rng(0))
    assert len(out) == 5
    assert all(len(q) == 12 for q in out)
    codes = [ord(c) for q in out for c in q]
    # clip to [2, 40] then +33 (Phred+33) -> ASCII in [35, 73]
    assert all(35 <= code <= 73 for code in codes)


def test_random_qualities_mean():
    """Quality scores are centred around the requested mean (default Q30)."""
    out = _random_qualities(200, 100, np.random.default_rng(0))
    scores = [ord(c) - 33 for q in out for c in q]
    assert abs(np.mean(scores) - 30) < 0.5


def test_random_qualities_reproducible():
    """The same seed yields the same quality strings."""
    first = _random_qualities(10, 20, np.random.default_rng(0))
    assert first == _random_qualities(10, 20, np.random.default_rng(0))


def test_write_fastq_roundtrip(tmp_path):
    """Records are written in the four-line fastq format and read back intact."""
    path = tmp_path / "out.fastq.gz"
    headers = ["s:0", "s:1"]
    sequences = ["ACGT", "TTTT"]
    qualities = ["IIII", "####"]
    _write_fastq(path, headers, sequences, qualities)
    assert _read_fastq(path) == [
        ("s:0", "ACGT", "IIII"),
        ("s:1", "TTTT", "####"),
    ]


def test_assemble_amplicons_count_and_length(populated_edgelist, marker_panel, assay):
    """The assembler returns n_reads amplicons of the expected length and alphabet."""
    n_reads = 10
    amplicons = _assemble_amplicons(
        populated_edgelist, n_reads, marker_panel, assay, np.random.default_rng(0)
    )
    assert len(amplicons) == n_reads
    assert all(len(a) == _AMPLICON_LENGTH for a in amplicons)
    assert set("".join(amplicons)) <= set("ACGT")


def test_assemble_amplicons_segments(populated_edgelist, marker_panel, assay):
    """Each amplicon segment matches its umi/pid/lbs/uei source."""
    edge = populated_edgelist.head(1)  # single row -> deterministic sampling
    amplicons = _assemble_amplicons(
        edge, 1, marker_panel, assay, np.random.default_rng(0)
    )
    assert len(amplicons) == 1
    amplicon = amplicons[0]

    umi1_len = assay.get_region_by_id("umi-1").max_len
    umi2_len = assay.get_region_by_id("umi-2").max_len
    uei_len = assay.get_region_by_id("uei").max_len
    lbs1 = assay.get_region_by_id("lbs-1").get_sequence()
    lbs2 = assay.get_region_by_id("lbs-2").get_sequence()

    panel_df = marker_panel.to_polars()
    seq1 = dict(zip(panel_df["marker_id"], panel_df["sequence_1"]))
    seq2 = dict(zip(panel_df["marker_id"], panel_df["sequence_2"]))
    pid1 = seq1[edge["marker1"][0]]
    pid2 = seq2[edge["marker2"][0]]

    expected_len = (
        umi1_len + len(pid1) + len(lbs1) + uei_len + len(lbs2) + len(pid2) + umi2_len
    )
    assert len(amplicon) == expected_len == _AMPLICON_LENGTH

    end1 = umi1_len
    assert amplicon[:end1] == _decode_2bit_dna(np.array([edge["umi1"][0]]), umi1_len)[0]
    end2 = end1 + len(pid1)
    assert amplicon[end1:end2] == pid1
    end3 = end2 + len(lbs1)
    assert amplicon[end2:end3] == lbs1
    end4 = end3 + uei_len
    uei_seq = amplicon[end3:end4]
    assert len(uei_seq) == uei_len
    assert set(uei_seq) <= set("ACGT")
    end5 = end4 + len(lbs2)
    assert amplicon[end4:end5] == lbs2
    end6 = end5 + len(pid2)
    assert amplicon[end5:end6] == pid2
    end7 = end6 + umi2_len
    assert amplicon[end6:end7] == _decode_2bit_dna(np.array([edge["umi2"][0]]), umi2_len)[0]
    assert end7 == len(amplicon)


def test_assemble_amplicons_reproducible(populated_edgelist, marker_panel, assay):
    """The same seed yields the same amplicons; a different seed differs."""
    first = _assemble_amplicons(
        populated_edgelist, 20, marker_panel, assay, np.random.default_rng(0)
    )
    same = _assemble_amplicons(
        populated_edgelist, 20, marker_panel, assay, np.random.default_rng(0)
    )
    other = _assemble_amplicons(
        populated_edgelist, 20, marker_panel, assay, np.random.default_rng(1)
    )
    assert first == same
    assert first != other


def test_write_paired_reads(tmp_path):
    """R1 comes from the amplicon start, R2 is the reverse complement of the end."""
    amplicons = ["A" * 100, "C" * 100]
    r1_path, r2_path = _write_paired_reads(
        amplicons, "sample", 70, 90, tmp_path, np.random.default_rng(0)
    )
    assert r1_path == tmp_path / "sample_R1.fastq.gz"
    assert r2_path == tmp_path / "sample_R2.fastq.gz"

    r1 = _read_fastq(r1_path)
    r2 = _read_fastq(r2_path)
    assert len(r1) == len(r2) == 2

    # both mates of a pair share the same read name
    assert [rec[0] for rec in r1] == ["sample:0", "sample:1"]
    assert [rec[0] for rec in r2] == ["sample:0", "sample:1"]

    assert r1[0][1] == "A" * 70
    assert r2[0][1] == _reverse_complement("A" * 90)
    assert r1[1][1] == "C" * 70
    assert r2[1][1] == _reverse_complement("C" * 90)

    assert all(len(rec[2]) == 70 for rec in r1)
    assert all(len(rec[2]) == 90 for rec in r2)


def test_to_fastq_end_to_end(tmp_path, populated_edgelist, marker_panel, assay):
    """to_fastq writes valid paired fastq files with the requested read count."""
    n_reads = 50
    r1_path, r2_path = to_fastq(
        "sample",
        n_reads,
        populated_edgelist,
        marker_panel,
        assay,
        output_dir=tmp_path,
        rng=0,
    )
    assert r1_path == tmp_path / "sample_R1.fastq.gz"
    assert r2_path == tmp_path / "sample_R2.fastq.gz"

    r1 = _read_fastq(r1_path)
    r2 = _read_fastq(r2_path)
    assert len(r1) == n_reads
    assert len(r2) == n_reads
    assert all(len(rec[1]) == 70 for rec in r1)
    assert all(len(rec[1]) == 90 for rec in r2)
    # sequence and quality lengths agree
    assert all(len(rec[1]) == len(rec[2]) for rec in r1)
    assert all(len(rec[1]) == len(rec[2]) for rec in r2)
    assert set("".join(rec[1] for rec in r1)) <= set("ACGT")
    assert set("".join(rec[1] for rec in r2)) <= set("ACGT")


def test_to_fastq_reproducible(tmp_path, populated_edgelist, marker_panel, assay):
    """The same seed yields identical reads; a different seed differs."""
    dirs = [tmp_path / name for name in ("a", "b", "c")]
    for directory in dirs:
        directory.mkdir()

    paths_a = to_fastq(
        "s", 30, populated_edgelist, marker_panel, assay, output_dir=dirs[0], rng=0
    )
    paths_b = to_fastq(
        "s", 30, populated_edgelist, marker_panel, assay, output_dir=dirs[1], rng=0
    )
    paths_c = to_fastq(
        "s", 30, populated_edgelist, marker_panel, assay, output_dir=dirs[2], rng=1
    )

    assert _read_fastq(paths_a[0]) == _read_fastq(paths_b[0])
    assert _read_fastq(paths_a[1]) == _read_fastq(paths_b[1])
    assert _read_fastq(paths_a[0]) != _read_fastq(paths_c[0])
