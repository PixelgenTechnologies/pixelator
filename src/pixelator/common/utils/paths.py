"""Utility functions for working with file paths and validating input files.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import collections.abc
import gzip
import logging
import re
import typing
from pathlib import Path, PurePath
from typing import Optional, Sequence, Union

from pixelator.common.types import PathType

logger = logging.getLogger(__name__)


def create_output_stage_dir(root: PathType, name: str) -> Path:
    """Create a new subfolder with `name` under the given `root` directory.

    Args:
        root: the parent directory
        name: the name of the directory to create

    Returns:
        the created folder (Path) (Path)
    """
    output = Path(root) / name
    if not output.is_dir():
        output.mkdir(parents=True)
    return output


def get_extension(filename: PathType, len_ext: int = 2) -> str:
    """Extract file extensions from a filename.

    Args:
        filename: the file name
        len_ext: the number of expected extensions parts e.g.: fq.gz gives len_ext=2

    Returns:
        the file extension (str) (str)
    """
    return "".join(PurePath(filename).suffixes[-len_ext:]).lstrip(".")


# Compression extensions that may be appended to any pixelator output file.
_KNOWN_COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".zst", ".xz"})

# Extensions of the sequence file formats read and written by pixelator.
_KNOWN_SEQUENCE_SUFFIXES = frozenset({".fastq", ".fq", ".fasta", ".fa"})

# Extensions of the non-sequence file formats written by pixelator.
_KNOWN_DATA_SUFFIXES = frozenset({".parquet", ".pxl", ".json", ".csv"})

# Suffixes that pixelator stages append to a sample name to describe the
# content of a file, e.g. `<sample>.demux.m1.part_000.parquet`.
_KNOWN_STAGE_SUFFIXES = frozenset(
    {
        ".amplicon",
        ".analysis",
        ".collapse",
        ".collapsed",
        ".dehashed",
        ".demux",
        ".denoised_graph",
        ".failed",
        ".graph",
        ".layout",
        ".m1",
        ".m2",
        ".meta",
        ".passed",
        ".post_failed",
        ".report",
        ".sample_calling",
    }
)

_PART_SUFFIX_PATTERN = re.compile(r"\.part_(\d+)\Z")


def _strip_known_suffixes(
    name: str, is_known_suffix: typing.Callable[[str], bool]
) -> str:
    """Repeatedly remove trailing dot-separated suffixes accepted by a predicate.

    Args:
        name: the file name (without directory components) to strip
        is_known_suffix: predicate deciding whether a suffix (including the
            leading dot) should be removed

    Returns:
        the name without its trailing known suffixes (str)
    """
    while True:
        stem, dot, suffix = name.rpartition(".")
        # An empty stem means a leading dot, which is part of the name itself.
        if not dot or not stem or not is_known_suffix(f".{suffix}"):
            return name
        name = stem


def _is_sequence_file_suffix(suffix: str) -> bool:
    """Return True for sequence file format and compression suffixes."""
    return suffix in _KNOWN_SEQUENCE_SUFFIXES or suffix in _KNOWN_COMPRESSION_SUFFIXES


def _is_sample_name_suffix(suffix: str) -> bool:
    """Return True for any suffix pixelator appends after a sample name."""
    return (
        _is_sequence_file_suffix(suffix)
        or suffix in _KNOWN_DATA_SUFFIXES
        or suffix in _KNOWN_STAGE_SUFFIXES
        or _PART_SUFFIX_PATTERN.fullmatch(suffix) is not None
    )


def strip_sequence_file_suffixes(name: str) -> str:
    """Remove trailing sequence file format and compression suffixes from a name.

    Args:
        name: the file name (without directory components) to strip

    Returns:
        the name without e.g. a trailing ``.fastq.gz`` or ``.fq.zst`` (str)
    """
    return _strip_known_suffixes(name, _is_sequence_file_suffix)


def get_sample_name(filename: PathType) -> str:
    """Extract the sample name from a sample's filename.

    All trailing suffixes that pixelator itself appends to a sample name are
    removed, i.e. the file format extension, any compression extension and the
    stage suffixes describing the content of the file. Everything that remains
    is the sample name, so sample names are allowed to contain dots::

        QC2504_sample01_downsample_0.365_S1.demux.part_000.parquet
        -> QC2504_sample01_downsample_0.365_S1

    Since the suffixes are stripped by name, a sample whose name ends in one of
    the pixelator stage suffixes (e.g. ``donor1.graph``) cannot be distinguished
    from a stage suffix and will be stripped as well.

    Args:
        filename: path to the file

    Returns:
        the sample name (str)
    """
    return _strip_known_suffixes(PurePath(filename).name, _is_sample_name_suffix)


def get_part_number(filename: PathType) -> int | None:
    """Extract the part number from the name of a file written in parts.

    Stages that split their output tag each file with a ``.part_<number>``
    suffix, e.g. ``<sample>.demux.m1.part_000.parquet``.

    Args:
        filename: path to the file

    Returns:
        the part number, or None if the name carries no part suffix (int | None)
    """
    for suffix in reversed(PurePath(filename).suffixes):
        match = _PART_SUFFIX_PATTERN.fullmatch(suffix)
        if match:
            return int(match.group(1))
    return None


def gz_size(filename: str) -> int:
    """Extract the size of a gzip compressed file.

    Args:
        filename: file name

    Returns:
        size of the file uncompressed (in bytes) (int)
    """
    with gzip.open(filename, "rb") as f:
        return f.seek(0, whence=2)


def sanity_check_inputs(
    input_files: Sequence[PathType] | PathType,
    allowed_extensions: Union[Sequence[str], Optional[str]] = None,
) -> None:
    """Perform basic sanity checking of input files.

    Args:
        input_files: the files to sanity check
        allowed_extensions: the expected file extension of the files, e.g. 'fastq.gz' or a tuple of
            allowed types eg. ('fastq.gz', 'fq.gz')

    Returns:
        None

    Raises:
        AssertionError: when any of validation fails
    """
    input_files_: list[PathType] = (
        input_files if not isinstance(input_files, PathType) else [input_files]  # type: ignore
    )

    for input_file in input_files_:
        input_file = Path(input_file)
        logger.debug("Sanity checking %s", input_file)

        if not input_file.is_file():
            raise AssertionError(f"{input_file} is not a file")

        if input_file.stat().st_size == 0:
            raise AssertionError(f"{input_file} is an empty file")

        if not isinstance(allowed_extensions, str) and isinstance(
            allowed_extensions, collections.abc.Sequence
        ):
            if not any(str(input_file).endswith(ext) for ext in allowed_extensions):
                raise AssertionError(
                    f"{input_file} does not have any of the "
                    f"extensions {', '.join(allowed_extensions)}"
                )
        elif allowed_extensions is not None and not str(input_file).endswith(
            allowed_extensions
        ):
            raise AssertionError(
                f"{input_file} does not have the extension {allowed_extensions}"
            )
