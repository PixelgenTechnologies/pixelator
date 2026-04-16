"""Copyright © 2025 Pixelgen Technologies AB."""

from unittest.mock import MagicMock, patch

import pytest

from pixelator.pna.utils.duckdb_utils import (
    DuckdbPerThreadMemoryError,
    get_single_thread_duckdb_config,
    parse_duckdb_memory_limit_to_bytes,
)


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


@patch("pixelator.pna.utils.duckdb_utils.duckdb.connect")
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


@patch("pixelator.pna.utils.duckdb_utils.duckdb.connect")
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


@patch("pixelator.pna.utils.duckdb_utils.duckdb.connect")
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
