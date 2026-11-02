"""Collapse per-read edgelist parquet files into unique molecules.

Copyright © 2026 Pixelgen Technologies AB.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from duckdb import DuckDBPyConnection

from pixelator.pna.utils import init_duckdb_conn


def _normalize_edgelist_paths(edgelist: Path | Sequence[Path]) -> list[Path]:
    """Normalize a single path or sequence of paths to a non-empty list."""
    if isinstance(edgelist, (str, Path)):
        paths = [Path(edgelist)]
    else:
        paths = [Path(path) for path in edgelist]
    if not paths:
        raise ValueError("At least one edgelist parquet file is required.")
    return paths


def _parquet_list_literal(paths: Sequence[Path]) -> str:
    """Build a DuckDB list literal of parquet paths for ``read_parquet``."""
    quoted = ", ".join(f"'{path}'" for path in paths)
    return f"[{quoted}]"


EXPECTED_EDGELIST_SCHEMA = {
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
    result = conn.sql(
        f"""
        WITH umi_counts AS (
            SELECT {umi_column} AS umi, COUNT(*) AS degree
            FROM read_parquet('{output_file}')
            GROUP BY {umi_column}
        ),
        umi_degree_dist AS (
            SELECT degree, COUNT(*) AS count
            FROM umi_counts
            GROUP BY degree
        )
        SELECT degree, count FROM umi_degree_dist ORDER BY degree ASC
        """
    )
    return OrderedDict((int(degree), int(count)) for degree, count in result.fetchall())


def _validate_edgelist_schema(conn: DuckDBPyConnection, edgelist: Path) -> None:
    """Validate that a parquet file has the native edgelist schema."""
    rows = conn.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(edgelist)]
    ).fetchall()
    actual_schema = {str(row[0]): str(row[1]) for row in rows}

    missing = [
        column for column in EXPECTED_EDGELIST_SCHEMA if column not in actual_schema
    ]
    extra = [
        column for column in actual_schema if column not in EXPECTED_EDGELIST_SCHEMA
    ]
    invalid_types = [
        f"{column} expected {expected_type}, got {actual_schema[column]}"
        for column, expected_type in EXPECTED_EDGELIST_SCHEMA.items()
        if column in actual_schema and actual_schema[column] != expected_type
    ]

    errors = []
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected columns: {', '.join(extra)}")
    if invalid_types:
        errors.append(f"invalid types: {'; '.join(invalid_types)}")

    if errors:
        expected = ", ".join(
            f"{column} {data_type}"
            for column, data_type in EXPECTED_EDGELIST_SCHEMA.items()
        )
        raise InvalidEdgelistSchemaError(
            f"Invalid edgelist schema for '{edgelist}': {'; '.join(errors)}. "
            f"Expected: {expected}."
        )


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
            Multiple files are unioned and collapsed together.
        output: Path to the output aggregated parquet file.
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
    parquet_source = _parquet_list_literal(edgelist_paths)

    with init_duckdb_conn(
        memory_limit=memory_limit,
        threads=max(threads, 1) if threads is not None else None,
        temp_dir=temp_directory,
    ) as conn:
        for path in edgelist_paths:
            _validate_edgelist_schema(conn, path)
        conn.execute("SET preserve_insertion_order = false")

        conn.execute(
            f"""
            COPY (
                SELECT marker_1, marker_2, umi1, umi2,
                    count(*)::UINTEGER AS read_count
                FROM read_parquet({parquet_source})
                GROUP BY marker_1, marker_2, umi1, umi2
                ORDER BY umi1, umi2, marker_1, marker_2
            ) TO '{output}' (FORMAT parquet, CODEC 'zstd', COMPRESSION_LEVEL 6);
            """
        )

        # No filtering during aggregate, so sum(read_count) == input row count.
        input_reads, output_molecules = (
            int(value)
            for value in conn.execute(
                f"""
                SELECT coalesce(sum(read_count), 0)::BIGINT, count(*)
                FROM read_parquet('{output}')
                """
            ).fetchone()
        )
        umi1_degree_distribution = _compute_degree_distribution(conn, output, "umi1")
        umi2_degree_distribution = _compute_degree_distribution(conn, output, "umi2")

    return AggregateStats(
        input_reads=input_reads,
        output_molecules=output_molecules,
        umi1_degree_distribution=umi1_degree_distribution,
        umi2_degree_distribution=umi2_degree_distribution,
    )
