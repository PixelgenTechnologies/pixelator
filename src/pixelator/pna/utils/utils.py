"""Utility functions.

Copyright © 2024 Pixelgen Technologies AB.
"""

import time
from functools import wraps
from pathlib import Path, PurePath
from typing import Iterable

import duckdb as dd
import pandas as pd
import polars as pl

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.common.utils import get_part_number, get_sample_name, logger

__all__ = [
    "get_demux_filename_info",
    "init_duckdb_conn",
    "normalize_input_to_list",
    "normalize_input_to_set",
    "timer",
]


def get_demux_filename_info(filename: str | Path | PurePath) -> tuple[str, int]:
    """Extract the sample name and part for a `demux` output parquet file.

    The demux output file are expeted to use following schema:
    <sample_name>.demux.part_<part_number>.parquet

    Args:
        filename: path to the file

    Returns:
        the sample name and the demux part (tuple[str, int])
    """
    if ".demux" not in PurePath(filename).name:
        raise ValueError("Invalid demux filename. Did not contain .demux")

    demux_part = get_part_number(filename)
    if demux_part is None:
        raise ValueError("Invalid demux filename. Did not contain .part_<number>")

    return get_sample_name(filename), demux_part


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
        temp_dir: The directory to use for temporary files. If None, defaults to
            ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp`` (never next to the database file).
        temp_dir_size_limit: The maximum size of the temporary directory. If None, defaults to
            ``PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE`` when set, otherwise no limit.

    Returns:
        A duckdb connection object.
    """
    conn = connect_duckdb(
        database=path,
        read_only=read_only,
        temp_dir=temp_dir,
        temp_dir_size_limit=temp_dir_size_limit,
    )

    commands = []
    if memory_limit is not None:
        commands.append(f"SET memory_limit = '{memory_limit / 10**6}MiB';")
        logger.debug("Using DuckDB memory limit: %s MB", memory_limit / 10**6)
    if threads is not None:
        commands.append(f"SET threads = {threads};")
        logger.debug("Using DuckDB threads limit: %s", threads)

    if commands:
        conn.execute("\n".join(commands))

    return conn
