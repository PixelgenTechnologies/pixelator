"""Smoke test that the frozen 5-cell PNA PBMC fixture opens.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.pna.pixeldataset import read


def test_minimal_pna_pbmc_fixture_opens(minimal_pna_pbmc_pxl_file: Path):
    """Verify the frozen 5-cell PNA PBMC fixture opens."""
    dataset = read(minimal_pna_pbmc_pxl_file)
    assert dataset.components() == {
        "0a45497c6bfbfb22",
        "2708240b908e2eba",
        "c3c393e9a17c1981",
        "d4074c845bb62800",
        "efe0ed189cb499fc",
    }
    assert not dataset.proximity().is_empty()

    with connect_duckdb(minimal_pna_pbmc_pxl_file, read_only=True) as con:
        tables = set(con.sql("SHOW ALL TABLES").to_df()["name"])
    assert {"edgelist", "proximity", "layouts"}.issubset(tables)
