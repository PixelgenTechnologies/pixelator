"""Fastq read generation from populated cell edge lists.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from pixelator.pna.config.assay import PNAAssay
    from pixelator.pna.config.panel import PNAAntibodyPanel


def to_fastq(
    sample_name: str,
    n_reads: int,
    edgelist: pl.DataFrame,
    panel: PNAAntibodyPanel,
    assay: PNAAssay,
    substitution_error_rate: float = 1e-5,
    read1_length: int = 70,
    read2_length: int = 90,
    output_dir: str | Path = ".",
    rng=None,
) -> tuple[Path, Path]:
    """Convert a populated edge list into reads and write paired-end fastq files.

    Args:
        sample_name: base name for the output ``<sample_name>_R1/2.fastq.gz`` files.
        n_reads: number of reads to sample.
        edgelist: populated edge list with umi/marker columns for both endpoints.
        panel: antibody panel providing the marker sequences.
        assay: assay describing the read structure (region lengths and sequences).
        substitution_error_rate: per-base probability of substituting a base for a
            different (random) one.
        read1_length: length of the forward (R1) read, taken from the amplicon start.
        read2_length: length of the reverse (R2) read, reverse-complemented from the
            amplicon end.
        output_dir: directory to write the fastq files into.
        rng: a seed or numpy Generator for the random number generator.

    Returns:
        The paths of the written ``R1`` and ``R2`` fastq files.
    """
    rng = np.random.default_rng(rng)
    amplicons = _assemble_amplicons(edgelist, n_reads, panel, assay, rng)
    amplicons = _add_substitutions(amplicons, substitution_error_rate, rng)
    return _write_paired_reads(
        amplicons, sample_name, read1_length, read2_length, output_dir, rng
    )


def _decode_2bit_dna(values: np.ndarray, length: int) -> list[str]:
    """Decode 2-bit-encoded integers into dna strings (A=0, C=1, G=2, T=3)."""
    bases = np.array(list("ACGT"))
    digits = (values[:, None] >> (2 * np.arange(length))) & 3
    return ["".join(row) for row in bases[digits]]


def _add_substitutions(
    sequences: list[str], rate: float, rng: np.random.Generator
) -> list[str]:
    """Substitute each base for a different random base with probability ``rate``."""
    bases = np.array(list("ACGT"))  # sorted, so searchsorted gives A=0,C=1,G=2,T=3
    idx = np.searchsorted(bases, np.array([list(s) for s in sequences]))
    mask = rng.random(idx.shape) < rate
    # offset of 1-3 (mod 4) guarantees a different base
    idx[mask] = (idx[mask] + rng.integers(1, 4, size=mask.sum())) % 4
    return ["".join(row) for row in bases[idx]]


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a dna sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


def _random_qualities(
    n: int, length: int, rng: np.random.Generator, mean_q: int = 30, std_q: float = 2.0
) -> list[str]:
    """Phred+33 quality strings with scores drawn around ``mean_q`` (default Q30)."""
    q = rng.normal(mean_q, std_q, size=(n, length)).round()
    q = np.clip(q, 2, 40).astype(np.uint8) + 33  # Phred+33 ASCII offset
    return [row.tobytes().decode("ascii") for row in q]


def _write_fastq(
    path: Path, headers: list[str], sequences: list[str], qualities: list[str]
) -> None:
    """Write reads to a gzip-compressed fastq file."""
    with gzip.open(path, "wt") as fh:
        for header, sequence, quality in zip(headers, sequences, qualities):
            fh.write(f"@{header}\n{sequence}\n+\n{quality}\n")


def _assemble_amplicons(
    edgelist: pl.DataFrame,
    n_reads: int,
    panel: PNAAntibodyPanel,
    assay: PNAAssay,
    rng: np.random.Generator,
) -> list[str]:
    """Sample n_reads edges and build {umi1}{pid1}{lbs1}{uei}{lbs2}{pid2}{umi2}.

    umi1/umi2/uei are 2-bit-encoded integers decoded back to dna; pid1/pid2 are
    the marker pid sequences from the panel (sequence_1 for marker1, sequence_2
    for marker2). uei is a random number with twice as many bits as the uei
    region is long (each base encodes 2 bits).
    """
    uei_len = assay.get_region_by_id("uei").max_len
    umi1_len = assay.get_region_by_id("umi-1").max_len
    umi2_len = assay.get_region_by_id("umi-2").max_len
    lbs1 = assay.get_region_by_id("lbs-1").get_sequence()
    lbs2 = assay.get_region_by_id("lbs-2").get_sequence()

    edgelist = edgelist.with_columns(
        uei=rng.integers(0, 1 << (2 * uei_len), size=edgelist.height, dtype=np.int64)
    )
    reads = edgelist[rng.integers(0, edgelist.height, size=n_reads)]
    reads = reads.with_columns(
        umi1_seq=pl.Series(_decode_2bit_dna(reads["umi1"].to_numpy(), umi1_len)),
        umi2_seq=pl.Series(_decode_2bit_dna(reads["umi2"].to_numpy(), umi2_len)),
        uei_seq=pl.Series(_decode_2bit_dna(reads["uei"].to_numpy(), uei_len)),
    )

    panel_df = panel.to_polars()
    return (
        reads.join(
            panel_df.select(marker1="marker_id", pid1="sequence_1"), on="marker1"
        )
        .join(panel_df.select(marker2="marker_id", pid2="sequence_2"), on="marker2")
        .select(
            pl.concat_str(
                "umi1_seq",
                "pid1",
                pl.lit(lbs1),
                "uei_seq",
                pl.lit(lbs2),
                "pid2",
                "umi2_seq",
            )
        )
        .to_series()
        .to_list()
    )


def _write_paired_reads(
    amplicons: list[str],
    sample_name: str,
    read1_length: int,
    read2_length: int,
    output_dir: str | Path,
    rng: np.random.Generator,
) -> tuple[Path, Path]:
    """Split amplicons into R1/R2 mates, add qualities, and write fastq.gz files.

    R1 is taken from the amplicon start, R2 is reverse-complemented from the
    amplicon end, and both mates of a pair share the same read name. Quality
    scores are distributed around Q30.
    """
    n = len(amplicons)
    headers = [f"{sample_name}:{i}" for i in range(n)]
    r1_seqs = [seq[:read1_length] for seq in amplicons]
    r2_seqs = [_reverse_complement(seq[-read2_length:]) for seq in amplicons]

    output_dir = Path(output_dir)
    r1_path = output_dir / f"{sample_name}_R1.fastq.gz"
    r2_path = output_dir / f"{sample_name}_R2.fastq.gz"
    _write_fastq(r1_path, headers, r1_seqs, _random_qualities(n, read1_length, rng))
    _write_fastq(r2_path, headers, r2_seqs, _random_qualities(n, read2_length, rng))
    return r1_path, r2_path
