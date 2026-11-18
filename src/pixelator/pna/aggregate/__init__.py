"""Collapse per-read edgelist parquet files into unique molecules.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from duckdb import DuckDBPyConnection

from pixelator.pna.utils import init_duckdb_conn

logger = logging.getLogger(__name__)

# Seconds between "still running" messages while a DuckDB query is executing.
_HEARTBEAT_INTERVAL_SECONDS = 30.0


@contextmanager
def _log_stage(stage: str) -> Iterator[None]:
    """Log the start and end of a stage, with periodic heartbeats in between.

    DuckDB's own progress bar is not usable here (it writes to the terminal and
    cannot be polled from another thread), so a background thread logs the
    elapsed time at a fixed interval to show the query is still running.
    """
    start = time.monotonic()
    done = threading.Event()

    def _heartbeat() -> None:
        while not done.wait(_HEARTBEAT_INTERVAL_SECONDS):
            logger.info(
                "Aggregate stage '%s' still running (%.0fs elapsed)",
                stage,
                time.monotonic() - start,
            )

    logger.info("Aggregate stage '%s' started", stage)
    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        yield
    finally:
        done.set()
        heartbeat.join()
    logger.info(
        "Aggregate stage '%s' finished in %.1fs", stage, time.monotonic() - start
    )


def _normalize_edgelist_paths(edgelist: Path | Sequence[Path]) -> list[Path]:
    """Normalize a single path or sequence of paths to a non-empty list."""
    if isinstance(edgelist, (str, Path)):
        paths = [Path(edgelist)]
    else:
        paths = [Path(path) for path in edgelist]
    if not paths:
        raise ValueError("At least one edgelist parquet file is required.")
    return paths


def _quote_sql_string(value: str | Path) -> str:
    """Quote a string as a DuckDB literal.

    Only used where DuckDB cannot bind a parameter (the ``COPY ... TO`` target).
    In DuckDB's standard strings a single quote is escaped by doubling it and
    backslashes are literal.
    """
    return "'" + str(value).replace("'", "''") + "'"


# Columns every edgelist must have. Other columns are allowed and are carried
# through to the output.
_REQUIRED_EDGELIST_SCHEMA = {
    "marker_1": "VARCHAR",
    "marker_2": "VARCHAR",
    "umi1": "UBIGINT",
    "umi2": "UBIGINT",
}


class InvalidEdgelistSchemaError(ValueError):
    """Raised when an input parquet does not have the edgelist schema."""


@dataclass(frozen=True)
class AggregateStats:
    """Statistics collected while aggregating an edgelist."""

    input_reads: int
    output_molecules: int
    umi1_degree_distribution: OrderedDict[int, int]
    umi2_degree_distribution: OrderedDict[int, int]


def _compute_degree_distribution(
    conn: DuckDBPyConnection, output_file: Path, umi_column: str
) -> OrderedDict[int, int]:
    """Histogram of molecule-edge degree for a UMI column."""
    result = conn.execute(
        f"""
        WITH umi_counts AS (
            SELECT {umi_column} AS umi, COUNT(*) AS degree
            FROM read_parquet(?)
            GROUP BY {umi_column}
        ),
        umi_degree_dist AS (
            SELECT degree, COUNT(*) AS count
            FROM umi_counts
            GROUP BY degree
        )
        SELECT degree, count FROM umi_degree_dist ORDER BY degree ASC
        """,
        [str(output_file)],
    )
    return OrderedDict((int(degree), int(count)) for degree, count in result.fetchall())


def _validate_edgelist_schema(
    conn: DuckDBPyConnection, edgelist: Path
) -> dict[str, str]:
    """Validate that a parquet file has the columns aggregate needs.

    Extra columns are allowed.

    Returns:
        The file's schema, as a mapping of column name to DuckDB type.

    Raises:
        InvalidEdgelistSchemaError: A required column is missing or has the
            wrong type.

    """
    rows = conn.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(edgelist)]
    ).fetchall()
    actual_schema = {str(row[0]): str(row[1]) for row in rows}

    missing = [
        column for column in _REQUIRED_EDGELIST_SCHEMA if column not in actual_schema
    ]
    invalid_types = [
        f"{column} expected {expected_type}, got {actual_schema[column]}"
        for column, expected_type in _REQUIRED_EDGELIST_SCHEMA.items()
        if column in actual_schema and actual_schema[column] != expected_type
    ]

    errors = []
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    if invalid_types:
        errors.append(f"invalid types: {'; '.join(invalid_types)}")

    if errors:
        expected = ", ".join(
            f"{column} {data_type}"
            for column, data_type in _REQUIRED_EDGELIST_SCHEMA.items()
        )
        raise InvalidEdgelistSchemaError(
            f"Invalid edgelist schema for '{edgelist}': {'; '.join(errors)}. "
            f"Expected: {expected}."
        )
    return actual_schema


def aggregate_edgelist(
    edgelist: Path | Sequence[Path],
    output: Path,
    threads: int | None = None,
    memory_limit: int | None = None,
    temp_directory: str | Path | None = None,
) -> AggregateStats:
    """Collapse identical molecules and count reads.

    Args:
        edgelist: Path or paths to the input per-read edgelist parquet file(s).
            Multiple files are unioned and collapsed together, and must all
            have the same columns. Reads collapse on ``marker_1``,
            ``marker_2``, ``umi1`` and ``umi2`` only. Any other column (such
            as ``marker_sequence_1`` and ``marker_sequence_2``) is carried
            through with an arbitrary read's value, so it should be constant
            per molecule.
        output: Path to the output aggregated parquet file. It has the input
            columns followed by ``read_count``.
        threads: Number of DuckDB threads. Defaults to DuckDB's own default.
        memory_limit: Soft memory limit in bytes. When set, DuckDB spills to
            disk once the limit is reached.
        temp_directory: DuckDB spill directory. Defaults to
            ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp``.

    Returns:
        AggregateStats with input/output counts and UMI degree distributions.

    """
    edgelist_paths = _normalize_edgelist_paths(edgelist)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with init_duckdb_conn(
        memory_limit=memory_limit,
        threads=max(threads, 1) if threads is not None else None,
        temp_dir=temp_directory,
    ) as conn:
        schemas = [_validate_edgelist_schema(conn, path) for path in edgelist_paths]
        # DuckDB takes the columns of the first file and silently drops
        # any that only later files have, so the files must agree.
        for path, schema in zip(edgelist_paths[1:], schemas[1:], strict=True):
            if schema != schemas[0]:
                raise InvalidEdgelistSchemaError(
                    f"Edgelist '{path}' has different columns or types than "
                    f"'{edgelist_paths[0]}'."
                )
        conn.execute("SET preserve_insertion_order = false")

        # Input and output paths must not be spliced into the SQL as-is: the inputs
        # are bound, and DuckDB cannot bind the COPY target, so it is quoted.
        with _log_stage("collapse and write molecules"):
            conn.execute(
                f"""
                COPY (
                    SELECT marker_1, marker_2, umi1, umi2,
                        any_value(COLUMNS(* EXCLUDE (marker_1, marker_2, umi1, umi2))),
                        count(*)::UINTEGER AS read_count
                    FROM read_parquet(?)
                    GROUP BY marker_1, marker_2, umi1, umi2
                    ORDER BY umi1, umi2, marker_1, marker_2
                ) TO {_quote_sql_string(output)}
                (FORMAT parquet, CODEC 'zstd', COMPRESSION_LEVEL 6);
                """,
                [[str(path) for path in edgelist_paths]],
            )

        # No filtering during aggregate, so sum(read_count) == input row count.
        with _log_stage("count molecules and reads"):
            counts = conn.execute(
                """
                SELECT coalesce(sum(read_count), 0)::BIGINT, count(*)
                FROM read_parquet(?)
                """,
                [str(output)],
            ).fetchone()
        # An aggregate query without GROUP BY always returns exactly one row.
        assert counts is not None
        input_reads, output_molecules = (int(value) for value in counts)
        with _log_stage("umi1 degree distribution"):
            umi1_degree_distribution = _compute_degree_distribution(
                conn, output, "umi1"
            )
        with _log_stage("umi2 degree distribution"):
            umi2_degree_distribution = _compute_degree_distribution(
                conn, output, "umi2"
            )

    return AggregateStats(
        input_reads=input_reads,
        output_molecules=output_molecules,
        umi1_degree_distribution=umi1_degree_distribution,
        umi2_degree_distribution=umi2_degree_distribution,
    )
