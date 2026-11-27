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


def quote_identifier(name: str) -> str:
    """Return ``name`` as a quoted DuckDB identifier.

    Use it for every identifier that is not a literal in the code, such as the
    table and column names read from a file's catalog.

    Raises:
        ValueError: If ``name`` contains a NUL character, which DuckDB treats as
            the end of the SQL text.
    """
    if "\x00" in name:
        raise ValueError(f"Invalid SQL identifier {name!r}: contains NUL.")
    return '"' + name.replace('"', '""') + '"'


def qualified_name(*parts: str) -> str:
    """Return ``parts`` (e.g. database, schema, table) as a quoted dotted name."""
    return ".".join(quote_identifier(part) for part in parts)


def get_duckdb_temp_dir_from_env() -> Path | None:
    """Return ``PIXELATOR_DUCKDB_TEMP_DIR`` as a ``Path``, or ``None`` if unset."""
    value = os.environ.get("PIXELATOR_DUCKDB_TEMP_DIR")
    return Path(value) if value else None


def get_duckdb_max_temp_dir_size_from_env() -> str | None:
    """Return ``PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE``, or ``None`` if unset."""
    value = os.environ.get("PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE")
    return value.strip() if value else None


def get_duckdb_threads_from_env() -> str | None:
    """Return ``PIXELATOR_DUCKDB_THREADS``, or ``None`` if unset."""
    value = os.environ.get("PIXELATOR_DUCKDB_THREADS")
    return value.strip() if value else None


def resolve_duckdb_temp_dir(temp_dir: Path | str | None = None) -> Path:
    """Resolve the DuckDB spill directory.

    Uses the explicit argument, then ``PIXELATOR_DUCKDB_TEMP_DIR``, then ``TMPDIR``,
    then ``/tmp``. An unset ``TMPDIR`` leaves the ``/tmp`` fallback unchanged.

    Args:
        temp_dir: An explicit spill directory that takes precedence over the env var.

    Returns:
        The resolved spill directory.
    """
    if temp_dir is not None:
        return Path(temp_dir)
    from_env = get_duckdb_temp_dir_from_env()
    if from_env is not None:
        return from_env
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        return Path(tmpdir)
    return DEFAULT_DUCKDB_TEMP_DIR


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
            If it has no ``threads`` entry, ``PIXELATOR_DUCKDB_THREADS`` is used when set.
        temp_dir: The spill directory. Defaults to ``PIXELATOR_DUCKDB_TEMP_DIR``,
            ``TMPDIR``, or ``/tmp``.
        temp_dir_size_limit: The spill directory size limit. Defaults to
            ``PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE`` when unset.

    Returns:
        A DuckDB connection with ``temp_directory`` configured.
    """
    config = dict(config or {})
    env_threads = get_duckdb_threads_from_env()
    if env_threads is not None:
        config.setdefault("threads", env_threads)

    conn = duckdb.connect(database=str(database), read_only=read_only, config=config)

    resolved_temp_dir = str(resolve_duckdb_temp_dir(temp_dir).absolute())
    commands = [f"SET temp_directory = '{resolved_temp_dir}';"]

    resolved_size = temp_dir_size_limit or get_duckdb_max_temp_dir_size_from_env()
    if resolved_size is not None:
        commands.append(f"SET max_temp_directory_size = '{resolved_size}';")

    conn.execute("\n".join(commands))
    return conn
