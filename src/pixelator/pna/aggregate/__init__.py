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


def _quote_sql_string(value: str | Path) -> str:
    """Quote a string as a DuckDB literal.

    Only used where DuckDB cannot bind a parameter (the ``COPY ... TO`` target).
    In DuckDB's standard strings a single quote is escaped by doubling it and
    backslashes are literal.
    """
    return "'" + str(value).replace("'", "''") + "'"


# Columns the collapse key needs. Other columns are allowed.
_REQUIRED_EDGELIST_SCHEMA = {
    "marker_1": "VARCHAR",
    "marker_2": "VARCHAR",
    "umi1": "UBIGINT",
    "umi2": "UBIGINT",
}

# Panel barcodes of marker_1 and marker_2. Optional, but only as a pair, and
# part of the output grouping key when present. DuckDB reads the dictionary
# encoding written by edgelist as VARCHAR.
_MARKER_SEQUENCE_COLUMNS = ("marker_sequence_1", "marker_sequence_2")
_MARKER_SEQUENCE_TYPE = "VARCHAR"


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


def _validate_edgelist_schema(conn: DuckDBPyConnection, edgelist: Path) -> bool:
    """Validate that a parquet file has the columns aggregate needs.

    Extra columns are allowed. ``marker_sequence_1`` and ``marker_sequence_2``
    are optional, but they must appear together and be ``VARCHAR``.

    Returns:
        True when both marker sequence columns are present.

    Raises:
        InvalidEdgelistSchemaError: A required column is missing, a required
            or marker-sequence column has the wrong type, or only one marker
            sequence column is present.

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

    present_sequences = [
        column for column in _MARKER_SEQUENCE_COLUMNS if column in actual_schema
    ]
    errors = []
    if len(present_sequences) == 1:
        absent = next(
            column for column in _MARKER_SEQUENCE_COLUMNS if column not in actual_schema
        )
        errors.append(
            "marker sequence columns must both be present or both be absent: "
            f"missing {absent}"
        )
    invalid_types.extend(
        f"{column} expected {_MARKER_SEQUENCE_TYPE}, got {actual_schema[column]}"
        for column in present_sequences
        if actual_schema[column] != _MARKER_SEQUENCE_TYPE
    )

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
    return len(present_sequences) == len(_MARKER_SEQUENCE_COLUMNS)


def _aggregate_copy_sql(output: Path, include_marker_sequences: bool) -> str:
    """Build the COPY statement that collapses reads into molecules.

    Marker sequence columns are selected and grouped when the inputs have them,
    so two reads collapse only when their panel barcodes agree.
    """
    sequence_columns = (
        "marker_sequence_1, marker_sequence_2, " if include_marker_sequences else ""
    )
    group_by = "marker_1, marker_2, umi1, umi2"
    if include_marker_sequences:
        group_by += ", marker_sequence_1, marker_sequence_2"
    return f"""
            COPY (
                SELECT marker_1, marker_2, umi1, umi2,
                    {sequence_columns}count(*)::UINTEGER AS read_count
                FROM read_parquet(?)
                GROUP BY {group_by}
                ORDER BY umi1, umi2, marker_1, marker_2
            ) TO {_quote_sql_string(output)}
            (FORMAT parquet, CODEC 'zstd', COMPRESSION_LEVEL 6);
            """


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
            Multiple files are unioned and collapsed together. When every file
            has ``marker_sequence_1`` and ``marker_sequence_2``, those columns
            are copied to the output. A mix of files with and without them is
            rejected. Other extra columns are ignored.
        output: Path to the output aggregated parquet file. With marker
            sequences, the columns are ``marker_1``, ``marker_2``, ``umi1``,
            ``umi2``, ``marker_sequence_1``, ``marker_sequence_2``, and
            ``read_count``.
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
        has_marker_sequences = [
            _validate_edgelist_schema(conn, path) for path in edgelist_paths
        ]
        if any(has_marker_sequences) and not all(has_marker_sequences):
            missing = [
                str(path)
                for path, present in zip(
                    edgelist_paths, has_marker_sequences, strict=True
                )
                if not present
            ]
            raise InvalidEdgelistSchemaError(
                "Marker sequence columns must be present on every edgelist or "
                f"on none. Missing from: {', '.join(missing)}."
            )
        conn.execute("SET preserve_insertion_order = false")

        # Input and output paths must not be spliced into the SQL as-is: the inputs
        # are bound, and DuckDB cannot bind the COPY target, so it is quoted.
        conn.execute(
            _aggregate_copy_sql(
                output, include_marker_sequences=all(has_marker_sequences)
            ),
            [[str(path) for path in edgelist_paths]],
        )

        # No filtering during aggregate, so sum(read_count) == input row count.
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
        umi1_degree_distribution = _compute_degree_distribution(conn, output, "umi1")
        umi2_degree_distribution = _compute_degree_distribution(conn, output, "umi2")

    return AggregateStats(
        input_reads=input_reads,
        output_molecules=output_molecules,
        umi1_degree_distribution=umi1_degree_distribution,
        umi2_degree_distribution=umi2_degree_distribution,
    )
