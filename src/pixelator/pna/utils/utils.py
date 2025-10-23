"""Utility functions.

Copyright © 2024 Pixelgen Technologies AB.
"""

import re
import time
import typing
from functools import wraps
from pathlib import Path, PurePath
from typing import Iterable, Literal

import duckdb as dd
import pandas as pd
import polars as pl

from pixelator.common.utils import (
    R1_REGEX,
    R2_REGEX,
    get_extension,
    get_sample_name,
    logger,
)

_KNOWN_SUFFIXES = {".fastq", ".fq", ".fasta", ".fa"}
_KNOWN_COMPRESSION = {".gz", ".bz2", ".zst", ".xz"}

R1_REGEX = R"(.[Rr]1$)|(_[Rr]?1$)|(_[Rr]?1)(?P<suffix>_[0-9]{3})$"
R2_REGEX = R"(.[Rr]2$)|(_[Rr]?2$)|(_[Rr]?2)(?P<suffix>_[0-9]{3})$"


def _check_extensions(read: str | Path):
    allowed_extensions = {
        ".fastq.gz",
        ".fq.gz",
        ".fastq",
        ".fq",
        ".fastq.zst",
        ".fq.zst",
    }

    read = str(read)

    if not any(read.endswith(e) for e in allowed_extensions):
        raise ValueError(
            "Invalid file extension: expected .fq or .fastq (with .gz or .zst compression)"
        )


def get_read_sample_name(read: str | Path) -> str:
    """Extract the sample name from a read file.

    Strip fq.gz or fastq.gz extension and remove R1/R2 suffixes.
    Supported R1 R2 identifieds are:

    _R1,_R2 | _r1, _r2 | _1, _2 | .R1, .R2 | .r1, .r2

    :param read: filename of a fastq read file
    :return str: sample name
    :raise ValueError: if the read file does not have a valid extension
    """
    # group input file by sample id and order reads by R1 and R2
    _check_extensions(read)

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

    :param read: filename of a fastq read file
    :param read_type: the read type to check for (r1 or r2)
    :return bool: True if the read file is a read 1 or 2 file
    :raise ValueError: if the read file does not have a valid extension
    :raise AssertionError: if the read_type is not 'r1' or 'r2'
    """
    read = Path(read).name

    if read_type not in ("r1", "r2"):
        raise AssertionError("Invalid read type: expected 'r1' or 'r2'")

    _check_extensions(read)

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


def clean_suffixes(path: PurePath) -> PurePath:
    """Remove known suffixes and compression extensions from a path.

    :param path: The path to clean
    :return: The path without fasta/fastq suffixes or compression extensions
    """
    while path.suffix in _KNOWN_COMPRESSION:
        path = path.with_suffix("")

    while path.suffix in _KNOWN_SUFFIXES:
        path = path.with_suffix("")

    return path


def get_demux_filename_info(filename: str | Path | PurePath) -> tuple[str, int]:
    """Extract the sample name and part for a `demux` output parquet file.

    The demux output file are expeted to use following schema:
    <sample_name>.demux.part_<part_number>.parquet

    :param filename: path to the file
    :returns str: the demux part
    """
    sample_name = get_sample_name(filename)
    filename_str = str(filename)

    if (".demux" not in filename_str) or (".part_" not in filename_str):
        raise ValueError("Invalid demux filename. Did not contain .demux or .part_")

    match = re.match(r".*\.part_(\d+)\.*", filename_str)
    if match:
        demux_part = int(match.group(1))
        return sample_name, demux_part

    raise ValueError("Invalid demux filename.")


def timer(command_name: str | None = None):
    """Time the different steps of a function."""

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwds):
            start_time = time.perf_counter()
            res = func(*args, **kwds)
            run_time = time.perf_counter() - start_time
            name = command_name or func.__name__
            logger.info("Finished pixelator %s in %.2fs", name, run_time)
            return res

        return inner

    return wrapper


def normalize_input_to_set(
    one_or_more_values: Iterable[str] | str | None,
) -> set[str] | None:
    """Normalize input to a set of strings."""
    if one_or_more_values is None:
        return None
    if isinstance(one_or_more_values, str):
        return {one_or_more_values}
    if isinstance(one_or_more_values, pd.Series):
        # For series return all truthy values from index
        return set(one_or_more_values[one_or_more_values].index)
    if isinstance(one_or_more_values, pl.Series):
        return set(one_or_more_values)
    if isinstance(one_or_more_values, pl.DataFrame):
        # if it is polars DataFrame with only one column get that
        if len(one_or_more_values.columns) == 1:
            return set(one_or_more_values.get_columns()[0])
        raise ValueError("If you pass a Polars DataFrame must have only one column")

    return {v for v in one_or_more_values}


def normalize_input_to_list(
    one_or_more_values: Iterable[str] | str | None,
) -> list[str] | None:
    """Normalize input to a list of strings."""
    if one_or_more_values is None:
        return None
    if isinstance(one_or_more_values, str):
        return [one_or_more_values]
    if isinstance(one_or_more_values, pd.Series):
        # For series return all truthy values from index
        return list(one_or_more_values[one_or_more_values].index)
    if isinstance(one_or_more_values, pl.Series):
        return one_or_more_values.to_list()
    if isinstance(one_or_more_values, pl.DataFrame):
        # if it is polars DataFrame with only one column get that
        if len(one_or_more_values.columns) == 1:
            return one_or_more_values.get_columns()[0].to_list()
        raise ValueError("If you pass a Polars DataFrame must have only one column")

    return [v for v in one_or_more_values]


def init_duckdb_conn(
    path: Path | str = ":memory:",
    read_only: bool = False,
    memory_limit: int | None = None,
    threads: int | None = None,
    temp_dir: str | Path | None = None,
    temp_dir_size_limit: str | None = None,
) -> dd.DuckDBPyConnection:
    """Initialize a duckdb connection with resource limits.

    Args:
        path: The path to the duckdb database file. Defaults to ":memory:" for in-memory database.
        read_only: Whether to open the database in read-only mode. Defaults to False.
        memory_limit: The memory limit in bytes. If None, no limit is set. Defaults to None.
        threads: The number of threads to use. If None, duckdb will decide. Defaults to None.
        temp_dir: The directory to use for temporary files. If None, duckdb will decide (defaults to /tmp). Defaults to None.
        temp_dir_size_limit: The maximum size of the temporary directory. If None, no limit is set. Defaults to None.

    Returns:
        A duckdb connection object.

    """
    conn = dd.connect(database=str(path), read_only=read_only)

    commands = []
    if memory_limit is not None:
        commands.append(f"SET memory_limit = '{memory_limit / 10**6}MiB';")
        logger.debug("Using DuckDB memory limit: %s MB", memory_limit / 10**6)
    if threads is not None:
        commands.append(f"SET threads = {threads};")
        logger.debug("Using DuckDB threads limit: %s", threads)
    if temp_dir is not None:
        temp_dir = str(Path(temp_dir).absolute())
        commands.append(f"SET temp_directory = '{temp_dir}';")
        logger.debug("Using DuckDB temp directory: %s", temp_dir)
    if temp_dir_size_limit is not None:
        commands.append(f"SET max_temp_directory_size = '{temp_dir_size_limit}';")
        logger.debug("Using DuckDB temp directory size limit: %s", temp_dir_size_limit)

    if commands:
        conn.execute("\n".join(commands))

    conn = dd.connect(":memory:")
    return conn
