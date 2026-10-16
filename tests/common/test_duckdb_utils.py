"""Copyright © 2026 Pixelgen Technologies AB."""

import duckdb
import pytest

from pixelator.common.duckdb_utils import connect_duckdb


def _threads(con) -> int:
    return con.execute("SELECT current_setting('threads')").fetchone()[0]


@pytest.mark.parametrize(
    "env_threads, config, expected",
    [
        ("1", None, 1),
        ("2", {"threads": "3"}, 3),
    ],
)
def test_connect_duckdb_threads_from_env(monkeypatch, env_threads, config, expected):
    """PIXELATOR_DUCKDB_THREADS sets the default; an explicit config wins."""
    monkeypatch.setenv("PIXELATOR_DUCKDB_THREADS", env_threads)
    with connect_duckdb(config=config) as con:
        assert _threads(con) == expected


def test_connect_duckdb_threads_unset_keeps_duckdb_default(monkeypatch):
    """Without the env var, DuckDB keeps its own default thread count."""
    monkeypatch.delenv("PIXELATOR_DUCKDB_THREADS", raising=False)
    with connect_duckdb() as con, duckdb.connect() as reference:
        assert _threads(con) == _threads(reference)
