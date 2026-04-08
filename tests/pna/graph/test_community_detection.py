"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pixelator.pna.graph.community_detection import (
    DuckdbPerThreadMemoryError,
    calculate_post_recovery_component_statistics,
    get_single_thread_duckdb_config,
    parse_duckdb_memory_limit_to_bytes,
)
from pixelator.pna.graph.report import GraphStatistics


@pytest.mark.parametrize(
    ("setting", "expected_bytes"),
    [
        ("2.0 GiB", 2 * 1024**3),
        ("1024 MiB", 1024 * 1024**2),
        ("1.0 MiB", 1024**2),
        ("512 KiB", 512 * 1024),
        ("1000 B", 1000),
        ("1 GB", 1000**3),
        ("1 TB", 1000**4),
    ],
)
def test_parse_duckdb_memory_limit_to_bytes(setting: str, expected_bytes: int) -> None:
    assert parse_duckdb_memory_limit_to_bytes(setting) == expected_bytes


def test_parse_duckdb_memory_limit_to_bytes_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_duckdb_memory_limit_to_bytes("12.5 XY")


@patch("pixelator.pna.graph.community_detection.duckdb.connect")
def test_get_single_thread_duckdb_config_raises_when_per_thread_below_1mib(
    mock_connect: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_con
    mock_connect.return_value.__exit__.return_value = None
    mock_con.execute.return_value.fetchone.return_value = ("1000 B",)

    with pytest.raises(
        DuckdbPerThreadMemoryError, match="Not enough memory to share DuckDB work"
    ):
        get_single_thread_duckdb_config(10_000)


@patch("pixelator.pna.graph.community_detection.duckdb.connect")
def test_get_single_thread_duckdb_config_allows_exactly_1_mib_per_thread(
    mock_connect: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_con
    mock_connect.return_value.__exit__.return_value = None
    mock_con.execute.return_value.fetchone.return_value = ("2.0 MiB",)

    cfg = get_single_thread_duckdb_config(2)

    assert cfg["threads"] == "1"
    assert cfg["memory_limit"] == f"{1024**2}B"


@patch("pixelator.pna.graph.community_detection.duckdb.connect")
def test_get_single_thread_duckdb_config_splits_total_bytes(
    mock_connect: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_con
    mock_connect.return_value.__exit__.return_value = None
    mock_con.execute.return_value.fetchone.return_value = ("1024 MiB",)

    cfg = get_single_thread_duckdb_config(4)

    assert cfg["memory_limit"] == f"{1024 * 1024**2 // 4}B"


def test_get_single_thread_duckdb_config_rejects_zero_threads() -> None:
    with pytest.raises(ValueError, match="n_threads"):
        get_single_thread_duckdb_config(0)


def test_calculate_post_recovery_component_statistics():
    """Edgelist stats: n_umi = n_unique(umi1) + n_unique(umi2) per component."""
    # Component A: two edges sharing umi1=1 and distinct umi2 → n_umi = 1 + 2 = 3, n_edges = 2
    # Component B: one edge → n_umi = 1 + 1 = 2, n_edges = 1
    edgelist = pl.DataFrame(
        {
            "umi1": [1, 1, 100],
            "umi2": [10, 11, 200],
            "component": ["A", "A", "B"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        edgelist.write_parquet(path)
        stats = GraphStatistics()
        out = calculate_post_recovery_component_statistics(path, stats)
    finally:
        path.unlink(missing_ok=True)

    assert out.component_count_post_recovery == 2
    assert out.edge_count_post_recovery == 3
    assert out.node_count_post_recovery == 5  # 3 + 2
    assert out.fraction_nodes_in_largest_component_post_recovery == pytest.approx(3 / 5)
