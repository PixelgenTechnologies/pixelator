"""Shared DuckDB connection helpers for Pixelator.

These helpers ensure DuckDB's spill (``temp_directory``) location is always set
explicitly. DuckDB otherwise defaults the spill location to a directory next to
the database file, which fails when the ``.pxl`` file lives on a networked
filesystem (e.g. Fusion/S3).

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_DUCKDB_TEMP_DIR = Path("/tmp")


def get_duckdb_temp_dir_from_env() -> Path | None:
    """Return ``PIXELATOR_DUCKDB_TEMP_DIR`` as a ``Path``, or ``None`` if unset."""
    value = os.environ.get("PIXELATOR_DUCKDB_TEMP_DIR")
    return Path(value) if value else None


def get_duckdb_max_temp_dir_size_from_env() -> str | None:
    """Return ``PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE``, or ``None`` if unset."""
    value = os.environ.get("PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE")
    return value.strip() if value else None


def resolve_duckdb_temp_dir(temp_dir: Path | str | None = None) -> Path:
    """Resolve the DuckDB spill directory.

    Uses the explicit argument, then ``PIXELATOR_DUCKDB_TEMP_DIR``, then ``/tmp``.

    Args:
        temp_dir: An explicit spill directory that takes precedence over the env var.

    Returns:
        The resolved spill directory.
    """
    if temp_dir is not None:
        return Path(temp_dir)
    return get_duckdb_temp_dir_from_env() or DEFAULT_DUCKDB_TEMP_DIR


def connect_duckdb(
    database: str | Path = ":memory:",
    *,
    read_only: bool = False,
    config: dict | None = None,
    temp_dir: str | Path | None = None,
    temp_dir_size_limit: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the spill (temp) directory always set.

    This prevents DuckDB from spilling next to the database file, which may be on a
    networked filesystem.

    Args:
        database: The database to connect to. Defaults to an in-memory database.
        read_only: Whether to open the database in read-only mode.
        config: A DuckDB configuration dict passed through to ``duckdb.connect``.
        temp_dir: The spill directory. Defaults to ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp``.
        temp_dir_size_limit: The spill directory size limit. Defaults to
            ``PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE`` when unset.

    Returns:
        A DuckDB connection with ``temp_directory`` configured.
    """
    conn = duckdb.connect(
        database=str(database), read_only=read_only, config=config or {}
    )

    resolved_temp_dir = str(resolve_duckdb_temp_dir(temp_dir).absolute())
    commands = [f"SET temp_directory = '{resolved_temp_dir}';"]

    resolved_size = temp_dir_size_limit or get_duckdb_max_temp_dir_size_from_env()
    if resolved_size is not None:
        commands.append(f"SET max_temp_directory_size = '{resolved_size}';")

    conn.execute("\n".join(commands))
    return conn
