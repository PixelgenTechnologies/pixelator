"""Tests for collapsing per-read edgelists into unique molecules.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import polars as pl
import pytest

from pixelator.pna.aggregate import (
    AggregateStats,
    InvalidEdgelistSchemaError,
    aggregate_edgelist,
)

EXPECTED_SCHEMA = {
    "marker_1": pl.String,
    "marker_2": pl.String,
    "umi1": pl.UInt64,
    "umi2": pl.UInt64,
    "read_count": pl.UInt32,
}

EXPECTED_SCHEMA_WITH_SEQUENCES = {
    "marker_1": pl.String,
    "marker_2": pl.String,
    "umi1": pl.UInt64,
    "umi2": pl.UInt64,
    "marker_sequence_1": pl.String,
    "marker_sequence_2": pl.String,
    "read_count": pl.UInt32,
}


def _write_edgelist(
    path: Path,
    rows: list[tuple[str, str, int, int]],
    sequences: list[tuple[str, str]] | None = None,
    extra_columns: dict[str, list] | None = None,
) -> None:
    """Write a native-style per-read edgelist parquet from row tuples."""
    marker_1, marker_2, umi1, umi2 = zip(*rows, strict=True)
    columns: dict[str, list | pl.Series] = {
        "marker_1": list(marker_1),
        "marker_2": list(marker_2),
        "umi1": pl.Series(list(umi1), dtype=pl.UInt64),
        "umi2": pl.Series(list(umi2), dtype=pl.UInt64),
    }
    if sequences is not None:
        sequence_1, sequence_2 = zip(*sequences, strict=True)
        columns["marker_sequence_1"] = list(sequence_1)
        columns["marker_sequence_2"] = list(sequence_2)
    if extra_columns is not None:
        columns.update(extra_columns)
    pl.DataFrame(columns).write_parquet(path)


@pytest.mark.parametrize(
    ("data", "expected_message"),
    [
        (
            {
                "marker_1": ["CD3"],
                "marker_2": ["CD4"],
                "umi1": pl.Series([1], dtype=pl.UInt64),
            },
            "missing columns: umi2",
        ),
        (
            {
                "marker_1": ["CD3"],
                "marker_2": ["CD4"],
                "umi1": ["not-an-integer"],
                "umi2": pl.Series([2], dtype=pl.UInt64),
            },
            "invalid types: umi1 expected UBIGINT, got VARCHAR",
        ),
        (
            {
                "marker_1": ["CD3"],
                "marker_2": ["CD4"],
                "umi1": pl.Series([1], dtype=pl.UInt64),
                "umi2": pl.Series([2], dtype=pl.UInt64),
                "marker_sequence_1": pl.Series([1], dtype=pl.Int32),
                "marker_sequence_2": ["TTTT"],
            },
            "invalid types: marker_sequence_1 expected VARCHAR, got INTEGER",
        ),
    ],
)
def test_aggregate_rejects_invalid_input_schema(
    tmp_path: Path, data: dict[str, list | pl.Series], expected_message: str
):
    """Invalid edgelist columns and types produce a clear error."""
    input_path = tmp_path / "invalid.parquet"
    output_path = tmp_path / "out.parquet"
    pl.DataFrame(data).write_parquet(input_path)

    with pytest.raises(InvalidEdgelistSchemaError, match=expected_message):
        aggregate_edgelist(input_path, output_path)

    assert not output_path.exists()


def test_aggregate_collapses_duplicates_and_sums_reads(tmp_path: Path):
    """Identical molecules collapse to one row; read_count sums input rows."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"

    # Deliberately shuffled duplicates: (CD3, CD4, 1, 2) appears 3 times,
    # (CD8, CD19, 5, 6) twice, (CD3, CD8, 9, 10) once.
    _write_edgelist(
        input_path,
        [
            ("CD3", "CD4", 1, 2),
            ("CD8", "CD19", 5, 6),
            ("CD3", "CD4", 1, 2),
            ("CD3", "CD8", 9, 10),
            ("CD8", "CD19", 5, 6),
            ("CD3", "CD4", 1, 2),
        ],
    )

    stats = aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    assert result.height == 3
    assert result["read_count"].sum() == 6

    by_key = {
        (r["marker_1"], r["marker_2"], r["umi1"], r["umi2"]): r["read_count"]
        for r in result.to_dicts()
    }
    assert by_key[("CD3", "CD4", 1, 2)] == 3
    assert by_key[("CD8", "CD19", 5, 6)] == 2
    assert by_key[("CD3", "CD8", 9, 10)] == 1

    assert isinstance(stats, AggregateStats)
    assert stats.input_reads == 6
    assert stats.output_molecules == 3


def test_aggregate_merges_multiple_input_files(tmp_path: Path):
    """Molecules shared across input files collapse with summed read_count."""
    input_a = tmp_path / "part_a.parquet"
    input_b = tmp_path / "part_b.parquet"
    output_path = tmp_path / "out.parquet"

    _write_edgelist(
        input_a,
        [
            ("CD3", "CD4", 1, 2),
            ("CD3", "CD4", 1, 2),
            ("CD8", "CD19", 5, 6),
        ],
    )
    _write_edgelist(
        input_b,
        [
            ("CD3", "CD4", 1, 2),
            ("CD3", "CD8", 9, 10),
        ],
    )

    stats = aggregate_edgelist([input_a, input_b], output_path)

    result = pl.read_parquet(output_path)
    assert result.height == 3
    assert result["read_count"].sum() == 5

    by_key = {
        (r["marker_1"], r["marker_2"], r["umi1"], r["umi2"]): r["read_count"]
        for r in result.to_dicts()
    }
    assert by_key[("CD3", "CD4", 1, 2)] == 3
    assert by_key[("CD8", "CD19", 5, 6)] == 1
    assert by_key[("CD3", "CD8", 9, 10)] == 1

    assert stats.input_reads == 5
    assert stats.output_molecules == 3


def test_aggregate_degree_distributions(tmp_path: Path):
    """UMI degree histograms count molecule edges per unique UMI."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"

    # After collapse:
    #   umi1=1 appears in 2 molecules (degree 2), umi1=2 in 1 (degree 1)
    #   umi2=10 appears in 2 molecules (degree 2), umi2=20 in 1 (degree 1)
    _write_edgelist(
        input_path,
        [
            ("A", "B", 1, 10),
            ("A", "B", 1, 10),  # duplicate read of first molecule
            ("A", "C", 1, 20),
            ("D", "E", 2, 10),
        ],
    )

    stats = aggregate_edgelist(input_path, output_path)

    assert stats.input_reads == 4
    assert stats.output_molecules == 3
    assert dict(stats.umi1_degree_distribution) == {1: 1, 2: 1}
    assert dict(stats.umi2_degree_distribution) == {1: 1, 2: 1}


def test_aggregate_output_schema(tmp_path: Path):
    """Output columns and dtypes match the aggregate schema."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"
    _write_edgelist(input_path, [("CD3", "CD4", 1, 2), ("CD3", "CD4", 1, 2)])

    aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    assert result.schema == EXPECTED_SCHEMA


def test_aggregate_keeps_marker_sequences(tmp_path: Path):
    """Marker sequences are copied and only identical sequences collapse."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"
    rows = [
        ("CD3", "CD4", 1, 2),
        ("CD3", "CD4", 1, 2),
        ("CD3", "CD4", 1, 2),
        ("CD8", "CD19", 5, 6),
    ]
    sequences = [
        ("AAAA", "TTTT"),
        ("AAAA", "TTTT"),
        ("CCCC", "TTTT"),
        ("GGGG", "CCCC"),
    ]
    _write_edgelist(
        input_path,
        rows,
        sequences=sequences,
        extra_columns={"batch": ["a", "a", "a", "b"]},
    )

    stats = aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    assert result.schema == EXPECTED_SCHEMA_WITH_SEQUENCES
    assert "batch" not in result.columns
    assert stats.input_reads == 4
    assert stats.output_molecules == 3

    by_key = {
        (
            row["marker_1"],
            row["marker_2"],
            row["umi1"],
            row["umi2"],
            row["marker_sequence_1"],
            row["marker_sequence_2"],
        ): row["read_count"]
        for row in result.to_dicts()
    }
    assert by_key[("CD3", "CD4", 1, 2, "AAAA", "TTTT")] == 2
    assert by_key[("CD3", "CD4", 1, 2, "CCCC", "TTTT")] == 1
    assert by_key[("CD8", "CD19", 5, 6, "GGGG", "CCCC")] == 1


def test_aggregate_merges_marker_sequences_across_files(tmp_path: Path):
    """Sequence columns from every input file collapse together."""
    input_a = tmp_path / "part_a.parquet"
    input_b = tmp_path / "part_b.parquet"
    output_path = tmp_path / "out.parquet"
    _write_edgelist(
        input_a,
        [("CD3", "CD4", 1, 2), ("CD8", "CD19", 5, 6)],
        sequences=[("AAAA", "TTTT"), ("GGGG", "CCCC")],
    )
    _write_edgelist(
        input_b,
        [("CD3", "CD4", 1, 2)],
        sequences=[("AAAA", "TTTT")],
    )

    stats = aggregate_edgelist([input_a, input_b], output_path)

    result = pl.read_parquet(output_path)
    assert result.schema == EXPECTED_SCHEMA_WITH_SEQUENCES
    assert stats.input_reads == 3
    assert stats.output_molecules == 2
    counts = {
        (row["umi1"], row["marker_sequence_1"]): row["read_count"]
        for row in result.to_dicts()
    }
    assert counts[(1, "AAAA")] == 2
    assert counts[(5, "GGGG")] == 1


def test_aggregate_keeps_a_single_marker_sequence_column(tmp_path: Path):
    """Each marker sequence column is optional on its own."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"
    _write_edgelist(
        input_path,
        [("CD3", "CD4", 1, 2), ("CD3", "CD4", 1, 2), ("CD3", "CD4", 1, 2)],
        extra_columns={"marker_sequence_2": ["TTTT", "TTTT", "GGGG"]},
    )

    aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    assert result.schema == {
        "marker_1": pl.String,
        "marker_2": pl.String,
        "umi1": pl.UInt64,
        "umi2": pl.UInt64,
        "marker_sequence_2": pl.String,
        "read_count": pl.UInt32,
    }
    counts = {row["marker_sequence_2"]: row["read_count"] for row in result.to_dicts()}
    assert counts == {"TTTT": 2, "GGGG": 1}


def test_aggregate_rejects_mixed_marker_sequence_files(tmp_path: Path):
    """Every input file must have the same marker sequence columns."""
    with_sequences = tmp_path / "with_sequences.parquet"
    without_sequences = tmp_path / "without_sequences.parquet"
    output_path = tmp_path / "out.parquet"
    _write_edgelist(
        with_sequences,
        [("CD3", "CD4", 1, 2)],
        sequences=[("AAAA", "TTTT")],
    )
    _write_edgelist(without_sequences, [("CD3", "CD4", 1, 2)])

    with pytest.raises(
        InvalidEdgelistSchemaError, match="must be the same on every edgelist"
    ) as exc_info:
        aggregate_edgelist([with_sequences, without_sequences], output_path)

    assert f"{without_sequences}: none" in str(exc_info.value)
    assert not output_path.exists()


def test_aggregate_ignores_unrelated_extra_columns(tmp_path: Path):
    """Columns other than the marker sequences are not required and not copied."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"
    _write_edgelist(
        input_path,
        [("CD3", "CD4", 1, 2), ("CD3", "CD4", 1, 2)],
        extra_columns={"batch": ["a", "b"]},
    )

    aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    assert result.schema == EXPECTED_SCHEMA
    assert result["read_count"].to_list() == [2]


def test_aggregate_sorts_by_umi_then_marker(tmp_path: Path):
    """Output rows are ordered by (umi1, umi2, marker_1, marker_2)."""
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "out.parquet"

    # Inserted out of sort order so a missing ORDER BY fails the assertion.
    _write_edgelist(
        input_path,
        [
            ("CD8", "CD19", 5, 1),
            ("CD3", "CD4", 1, 9),
            ("B7", "A1", 1, 2),
            ("A1", "B7", 1, 2),
            ("CD3", "CD4", 2, 1),
        ],
    )

    aggregate_edgelist(input_path, output_path)

    result = pl.read_parquet(output_path)
    expected_order = result.sort(["umi1", "umi2", "marker_1", "marker_2"])
    assert result.to_dicts() == expected_order.to_dicts()


@pytest.mark.slow
def test_aggregate_respects_memory_limit(tmp_path: Path):
    """Large input completes under a tight memory limit via DuckDB spilling."""
    input_path = tmp_path / "large.parquet"
    output_path = tmp_path / "out.parquet"

    n_rows = 5_000_000
    n_unique = 2_000_000
    # Cycle through n_unique keys so most rows are duplicates and the
    # hash aggregate + ORDER BY must spill under a 200 MB cap.
    idx = pl.arange(0, n_rows, eager=True)
    pl.DataFrame(
        {
            "marker_1": (idx % 50).cast(pl.Utf8),
            "marker_2": ((idx // 50) % 50).cast(pl.Utf8),
            "umi1": (idx % n_unique).cast(pl.UInt64),
            "umi2": ((idx * 7) % n_unique).cast(pl.UInt64),
        }
    ).write_parquet(input_path)

    aggregate_edgelist(input_path, output_path, memory_limit=200 * 1024**2, threads=2)

    result = pl.read_parquet(output_path)
    assert result["read_count"].sum() == n_rows
    assert result.height == n_unique
    assert result.schema == EXPECTED_SCHEMA
    assert result.equals(result.sort(["umi1", "umi2", "marker_1", "marker_2"]))


@pytest.mark.parametrize(
    "dir_name",
    ["it's", 'say "hi"', "back\\slash", "tab\\tnew\\nline", "100%_{braces}"],
)
def test_aggregate_handles_special_characters_in_paths(tmp_path: Path, dir_name: str):
    """Quotes, backslashes and braces in input and output paths are not interpreted."""
    work_dir = tmp_path / dir_name
    work_dir.mkdir()
    input_a = work_dir / "part'a.parquet"
    input_b = work_dir / "part\\b.parquet"
    output_path = work_dir / "out'put" / "agg.parquet"

    _write_edgelist(input_a, [("A", "B", 1, 10), ("A", "B", 1, 10)])
    _write_edgelist(input_b, [("A", "B", 1, 10), ("C", "D", 2, 20)])

    stats = aggregate_edgelist([input_a, input_b], output_path)

    assert stats.input_reads == 4
    assert stats.output_molecules == 2
    result = pl.read_parquet(output_path).sort("umi1")
    assert result["read_count"].to_list() == [3, 1]
