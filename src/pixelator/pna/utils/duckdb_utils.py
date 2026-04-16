"""DuckDB configuration helpers for the pixelator.pna package.

Copyright (c) 2025 Pixelgen Technologies AB.
"""

import re

import duckdb

from pixelator.pna.cli.common import logger

_DUCKDB_MEMORY_LIMIT_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]*)?)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*$",
    re.IGNORECASE,
)

# DuckDB uses decimal (1000) units for KB–TB and binary (1024) for KiB–TiB.
_DUCKDB_MEMORY_UNIT_TO_BYTES: dict[str, int] = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

# Minimum per-thread DuckDB memory when splitting ``memory_limit`` across workers (1 MiB).
_MIN_PER_THREAD_DUCKDB_BYTES = 1024 * 1024


class DuckdbPerThreadMemoryError(ValueError):
    """Raised when DuckDB ``memory_limit`` cannot be split across the requested worker count."""


def parse_duckdb_memory_limit_to_bytes(setting: str) -> int:
    """Parse a DuckDB ``memory_limit`` setting string to a byte count (floor).

    Handles values returned by ``SELECT current_setting('memory_limit')``, for example
    ``49.5 GiB`` or ``953.6 MiB``.

    Args:
        setting: Raw setting string from DuckDB.

    Returns:
        Total limit in bytes (non-negative integer).

    Raises:
        ValueError: If the string does not match DuckDB's expected format.

    """
    m = _DUCKDB_MEMORY_LIMIT_RE.match(setting.strip())
    if not m:
        msg = f"Unrecognized DuckDB memory_limit format: {setting!r}"
        raise ValueError(msg)
    value = float(m.group(1))
    unit = m.group(2).lower()
    mult = _DUCKDB_MEMORY_UNIT_TO_BYTES[unit]
    return int(value * mult)


def get_single_thread_duckdb_config(n_threads: int) -> dict:
    """Get a DuckDB configuration that limits memory usage for multi-threaded processing.

    Args:
        n_threads (int): Number of threads to be used in the multi-threaded processing.

    Returns:
        dict: DuckDB configuration dictionary with memory limit and single thread setting.

    Raises:
        ValueError: If ``n_threads`` is invalid.
        DuckdbPerThreadMemoryError: If the configured memory split would give each thread
            less than 1 MiB.

    """
    if n_threads < 1:
        msg = f"n_threads must be >= 1, got {n_threads}"
        raise ValueError(msg)

    with duckdb.connect() as con:
        raw_limit = con.execute("SELECT current_setting('memory_limit');").fetchone()[0]  # type: ignore[index]
    if not isinstance(raw_limit, str):
        raw_limit = str(raw_limit)

    total_bytes = parse_duckdb_memory_limit_to_bytes(raw_limit)
    per_thread_bytes = total_bytes // n_threads
    if per_thread_bytes < _MIN_PER_THREAD_DUCKDB_BYTES:
        msg = (
            f"Not enough memory to share DuckDB work among {n_threads} threads: "
            f"per-thread limit would be {per_thread_bytes} bytes "
            f"(minimum {_MIN_PER_THREAD_DUCKDB_BYTES} bytes, 1 MiB). "
            f"DuckDB memory_limit is {raw_limit!r} ({total_bytes} bytes total)."
        )
        raise DuckdbPerThreadMemoryError(msg)

    logger.debug(
        "get_single_thread_duckdb_config: DuckDB memory_limit=%r -> %d bytes total, "
        "n_threads=%d, per_thread=%d bytes",
        raw_limit,
        total_bytes,
        n_threads,
        per_thread_bytes,
    )

    duckdb_single_config = {
        "memory_limit": f"{per_thread_bytes}B",
        "threads": "1",
    }

    logger.debug(
        "get_single_thread_duckdb_config: resulting config %s",
        duckdb_single_config,
    )

    return duckdb_single_config
