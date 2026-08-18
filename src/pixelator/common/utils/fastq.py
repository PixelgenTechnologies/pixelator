"""Utility functions for working with FASTQ read files and read pairs.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import itertools
import re
import typing
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import click

from pixelator.common.types import PathType
from pixelator.common.utils.paths import get_extension, get_sample_name

R1_REGEX = R"(.[Rr]1$)|(_[Rr]?1$)|(_[Rr]?1)(?P<suffix>_[0-9]{3})$"
R2_REGEX = R"(.[Rr]2$)|(_[Rr]?2$)|(_[Rr]?2)(?P<suffix>_[0-9]{3})$"


def get_read_sample_name(read: str) -> str:
    """Extract the sample name from a read file.

    Strip fq.gz or fastq.gz extension and remove R1/R2 suffixes.
    Supported R1 R2 identifiers are:
    _R1,_R2 | _r1, _r2 | _1, _2 | .R1, .R2 | .r1, .r2

    Args:
        read: filename of a fastq read file

    Returns:
        sample name (str)

    Raises:
        ValueError: if the read file does not have a valid extension
    """
    # group input file by sample id and order reads by R1 and R2
    if not (read.endswith("fq.gz") or read.endswith("fastq.gz")):
        raise ValueError("Invalid file extension: expected .fq.gz or .fastq.gz")

    read_stem = Path(read).name
    read_stem = read_stem.removesuffix(get_extension(read_stem, 2)).rstrip(".")
    r1_match = re.search(R1_REGEX, read_stem)
    r2_match = re.search(R2_REGEX, read_stem)

    # Check if the r1 and r2 suffixes are "exclusive or"
    if r1_match and r2_match or (not r1_match and not r2_match):
        raise ValueError("Invalid R1/R2 suffix.")

    # We need to cast away the optional here r1 or r2 will always
    # return a match object since we checked for both being None above
    match = typing.cast(re.Match[str], r1_match or r2_match)

    # Remove the R1 or R2 suffix by using the indices returned by the match
    s, e = match.span()
    sample_name = read_stem[0:s] + read_stem[e:-1]

    if match.groupdict().get("suffix"):
        sample_name += match.group("suffix")

    return sample_name


def is_read_file(read: Path | str, read_type: Literal["r1"] | Literal["r2"]) -> bool:
    """Check if a read filename matches the specified read_type.

    Detects the presence of a common read 1 or read 2 suffix in the filename.

    Args:
        read: filename of a fastq read file
        read_type: the read type to check for (r1 or r2)

    Returns:
        True if the read file is a read 1 or 2 file (bool)

    Raises:
        ValueError: if the read file does not have a valid extension
        AssertionError: if the read_type is not 'r1' or 'r2'
    """
    read = Path(read).name

    if read_type not in ("r1", "r2"):
        raise AssertionError("Invalid read type: expected 'r1' or 'r2'")

    if not (read.endswith("fq.gz") or read.endswith("fastq.gz")):
        raise ValueError("Invalid file extension: expected .fq.gz or .fastq.gz")

    match: re.Match[str] | None = None
    read_stem = Path(read.removesuffix(get_extension(read, 2)).rstrip(".")).name
    if read_type == "r1":
        match = re.search(R1_REGEX, read_stem)
    elif read_type == "r2":
        match = re.search(R2_REGEX, read_stem)
    else:
        raise AssertionError(
            "Invalid read type: could not find a read suffix in filename."
        )

    if not match:
        return False

    return True


def group_input_reads(
    inputs: Sequence[PathType], input1_pattern: str, input2_pattern: str
) -> Dict[str, List[Path]]:
    """Group input files by read pairs and sample id.

    Args:
        inputs: list of input files
        input1_pattern: pattern to match read1 files
        input2_pattern: pattern to match read2 files

    Returns:
        a dictionary with the grouped reads (Dict[str, List[Path]])

    Raises:
        ValueError: if the number of reads for a sample is more than 2
    """

    def group_fn(s):
        """Return the normalized sample name used for grouping inputs."""
        sn = get_sample_name(s)
        return sn.replace(input1_pattern, "").replace(input2_pattern, "")

    inputs = sorted(inputs, key=group_fn)
    # group reads by sample id
    grouped_inputs = {
        key: list(val_iter) for key, val_iter in itertools.groupby(inputs, group_fn)
    }

    # If the input contains 2 files, match them to the read1 and read2 patterns
    # otherwise, assume that the input is a single file and ignore the read patterns
    sorted_grouped_reads = {}
    for key, values in grouped_inputs.items():
        if len(values) == 2:
            input1 = sorted([Path(x) for x in values if input1_pattern in str(x)])
            input2 = sorted([Path(x) for x in values if input2_pattern in str(x)])

            if len(input1) != 1:
                raise click.ClickException(
                    f"Expected an input files identified with {input1_pattern}"
                )

            if len(input2) != 1:
                raise click.ClickException(
                    f"Expected an input files identified with {input2_pattern}"
                )

            sorted_grouped_reads[key] = [input1[0], input2[0]]
        elif len(values) == 1:
            sorted_grouped_reads[key] = [Path(values[0])]
        else:
            raise ValueError(f"Unexpected number of inputs for sample {key}")

    return sorted_grouped_reads
