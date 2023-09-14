"""Collection of functions for the concatenation of raw fastq reads.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import json
import logging
from pathlib import Path

# List is used as type hint in comment
from typing import BinaryIO, Sequence, Tuple  # noqa: F401

import pyfastx
from xopen import xopen

from pixelator.amplicon.statistics import SequenceQualityStatsCollector
from pixelator.config import Region, config
from pixelator.types import PathType
from pixelator.utils import reverse_complement

logger = logging.getLogger(__name__)

ILLMN_QUAL_0 = "!"

FastxReadTuple = Tuple[str, str, str]


def trim_amplicon(record1: FastxReadTuple, amplicon: Region) -> FastxReadTuple:
    """Trim a single-end fastq record to match the amplicon length.

    :param record1: the forward fastq record
    :param amplicon: the amplicon region from the assay specification
    :returns: a tuple with the sequence id, sequence and quality
    :rtype: FastxReadTuple
    """
    n1, s1, q1 = record1
    s1_len = len(s1)

    # Full amplicon with Ns placeholders for random sequences
    min_amplicon_len, max_amplicon_len = amplicon.get_len()

    if s1_len <= min_amplicon_len:
        return record1

    amplicon_seq = s1[:min_amplicon_len]
    amplicon_qual = q1[:min_amplicon_len]

    return n1, amplicon_seq, amplicon_qual


def generate_amplicon(
    record1: FastxReadTuple, record2: FastxReadTuple, amplicon: Region
) -> FastxReadTuple:
    """Generate a read from paired end fastq records.

    :param record1: a tuple with name, sequence and quality of the forward reads
    :param record2: a tuple with name, sequence and quality of the reverse read
    :param amplicon: the amplicon region from the assay specification
    :return: a tuple with the name, sequence and quality
    :rtype: FastxReadTuple
    :raises ValueError: if the headers of the two records are different
    """
    n1, s1, q1 = record1
    n2, s2, q2 = record2

    if n1 != n2:
        msg = "Found different headers when parsing fastq files"
        raise ValueError(msg)

    s1_len = len(s1)
    s2_len = len(s2)

    # Full amplicon with N placeholders for random sequences
    ref_amplicon_seq = amplicon.get_sequence()
    min_amplicon_len, max_amplicon_len = amplicon.get_len()
    ref_amplicon_qual = "!" * len(ref_amplicon_seq)

    # R1 already longer than amplicon -> just trim similar to single-end
    if s1_len >= min_amplicon_len:
        amplicon_seq = s1[:min_amplicon_len]
        amplicon_qual = q1[:min_amplicon_len]

        return n1, amplicon_seq, amplicon_qual

    # R2 already longer than amplicon -> just trim similar to single-end
    if s2_len >= min_amplicon_len:
        amplicon_seq = reverse_complement(s2)[:min_amplicon_len]
        amplicon_qual = q1[::-1][:min_amplicon_len]

        return n1, amplicon_seq, amplicon_qual

    # The amplicon sequence and quality strings
    # are overwritten with R1 from the start and R2 from the end.
    amplicon_seq = s1 + ref_amplicon_seq[s1_len:]
    amplicon_qual = q1 + ref_amplicon_qual[s1_len:]

    amplicon_seq = amplicon_seq[:-s2_len] + reverse_complement(s2)
    amplicon_qual = amplicon_qual[:-s2_len] + q2[::-1]

    return n1, amplicon_seq, amplicon_qual


def write_record(f: BinaryIO, header: str, sequence: str, quality: str) -> None:
    """Write a fastq record to a file.

    :param f: the file to write to
    :param header: the header of the record
    :param sequence: the sequence of the record
    :param quality: the quality of the record
    :rtype: None
    """
    # Do not generate intermediate strings here to avoid unneeded copies
    f.write(b"@")
    f.write(header.encode("utf-8"))
    f.write(b"\n")
    f.write(sequence.encode("utf-8"))  # type: ignore
    f.write(b"\n+\n")
    f.write(quality.encode("utf-8"))  # type: ignore
    f.write(b"\n")


def amplicon_fastq(
    inputs: Sequence[PathType],
    design: str,
    metrics: PathType,
    output: PathType,
) -> None:
    """Build MPX amplicons and save them to fastq files.

    The amplicon building process works without trying to overlap PE reads at the
    moment, even if PE reads would overlap. This function will iterate through the
    contents of the MPX fastq files and generate a new one based on the designed
    amplicon.

    :param inputs: a list of path to the fastq reads
    :param design: the design used in the config file
    :param metrics: the path to the json metrics file
    :param output: the path to the output file (processed)
    :returns: None
    :rtype: None
    :raises RuntimeError: raises an exception
    """

    def _mode_read_len(input: PathType, n: int = 10000) -> int:
        """Find the most common mode of read lengths from n number of reads."""
        from collections import Counter

        lengths = []
        for record in pyfastx.Fastq(str(input), build_index=False):
            r_name, r_seq, r_qual = record
            lengths.append(len(r_seq))
            if len(lengths) == n:
                break
        data = Counter(lengths)

        return data.most_common(1)[0][0]

    logger.debug("Using design %s", design)
    assay = config.get_assay(design)

    if assay is None:
        raise RuntimeError(f"Unknown design {design}")

    if not 0 < len(inputs) <= 2:
        raise RuntimeError("Invalid number of input files, expected 1 or 2 files")

    mode = "single-end" if len(inputs) == 1 else "paired-end"
    stats = SequenceQualityStatsCollector(design)

    start1_log_msg = "Starting the concatenation of %s to %s"
    start2_log_msg = "Starting the concatenation of %s and %s to %s"
    end1_log_msg = "Finished the concatenation of %s to %s"
    end2_log_msg = "Finished the concatenation of %s and %s to %s"

    amplicon = assay.get_region_by_id("amplicon")
    if amplicon is None:
        raise RuntimeError("Design does not have a region with id: amplicon")

    # Single end mode
    if mode == "single-end":
        input = inputs[0] if Path(inputs[0]).is_file() else inputs[1]
        logger.debug(start1_log_msg, input, output)

        mode = _mode_read_len(input)
        amplicon_len = amplicon.get_len()[0]
        if mode < amplicon_len:
            raise ValueError(
                (
                    f"Read input length in {input} ({mode}) is less "
                    f"than amplicon length ({amplicon_len})"
                )
            )

        with xopen(output, "wb") as f:
            for record in pyfastx.Fastq(str(inputs[0]), build_index=False):
                name, new_seq, new_qual = trim_amplicon(record, amplicon)
                write_record(f, name, new_seq, new_qual)
                stats.update(new_qual)

    if mode == "paired-end":
        logger.debug(start2_log_msg, inputs[0], inputs[1], output)

        mode1 = _mode_read_len(inputs[0])
        mode2 = _mode_read_len(inputs[1])
        total_amplicon_space = mode1 + mode2
        curr_amplicon_space = total_amplicon_space
        region_ids = amplicon.get_subregion_ids()
        regions_covered = []
        while total_amplicon_space and len(region_ids):
            first_el = region_ids.pop(0)
            regions_covered.append(first_el)
            curr_amplicon_space -= amplicon.get_region_by_id(first_el).max_len
            if curr_amplicon_space and len(region_ids):
                last_el = region_ids.pop(-1)
                regions_covered.append(last_el)
                curr_amplicon_space -= amplicon.get_region_by_id(last_el).max_len

        logger.debug(amplicon.get_subregion_ids())
        logger.debug(assay.region_ids)
        for r in assay.region_ids:
            logger.debug(amplicon.get_region_by_id(r).required)
            logger.debug(amplicon.get_region_by_id(r).is_required())
        logger.debug(assay)
        for r in assay.assay_spec:
            logger.debug(r)
        regions_required = assay.required_regions
        logger.debug(regions_required)
        if not (regions_required.issubset(set(regions_covered))):
            raise ValueError(
                (
                    f"Read input length in {inputs[0]} and {inputs[1]} "
                    f"({total_amplicon_space}) is less than recommendation"
                )
            )
        logger.fatal("Here we go!")

        with xopen(output, "wb") as f:
            for record1, record2 in zip(
                pyfastx.Fastq(str(inputs[0]), build_index=False),
                pyfastx.Fastq(str(inputs[1]), build_index=False),
            ):
                name, new_seq, new_qual = generate_amplicon(record1, record2, amplicon)
                write_record(f, name, new_seq, new_qual)
                stats.update(new_qual)

    # add metrics to JSON file
    avg_stats = stats.stats

    data = {"phred_result": avg_stats.asdict()}
    with open(str(metrics), "w") as json_file:
        json.dump(data, json_file, sort_keys=True, indent=4)

    if mode == "single-end":
        logger.debug(end1_log_msg, inputs[0], output)
    else:
        logger.debug(end2_log_msg, inputs[0], inputs[1], output)
