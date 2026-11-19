"""Copyright © 2026 Pixelgen Technologies AB."""

from pathlib import Path

import duckdb
import pytest

from pixelator.common.duckdb_utils import connect_duckdb, resolve_duckdb_temp_dir


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


def test_resolve_duckdb_temp_dir_fallback_order(monkeypatch, tmp_path):
    """Explicit path, then PIXELATOR_DUCKDB_TEMP_DIR, then TMPDIR, then /tmp."""
    explicit = tmp_path / "explicit"
    pixelator = tmp_path / "pixelator"
    tmpdir = tmp_path / "tmpdir"

    monkeypatch.setenv("PIXELATOR_DUCKDB_TEMP_DIR", str(pixelator))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    assert resolve_duckdb_temp_dir(explicit) == explicit
    assert resolve_duckdb_temp_dir() == pixelator

    monkeypatch.delenv("PIXELATOR_DUCKDB_TEMP_DIR")
    assert resolve_duckdb_temp_dir() == tmpdir

    monkeypatch.delenv("TMPDIR")
    assert resolve_duckdb_temp_dir() == Path("/tmp")


def test_connect_duckdb_threads_unset_keeps_duckdb_default(monkeypatch):
    """Without the env var, DuckDB keeps its own default thread count."""
    monkeypatch.delenv("PIXELATOR_DUCKDB_THREADS", raising=False)
    with connect_duckdb() as con, duckdb.connect() as reference:
        assert _threads(con) == _threads(reference)
