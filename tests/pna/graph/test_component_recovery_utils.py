"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

import polars as pl

from pixelator.pna.graph.component_recovery_utils import get_count_statistics


def test_get_count_statistics(tmp_path: Path) -> None:
    """get_count_statistics yields correct edge, read, molecule, and distinct-UMI counts from a Parquet edgelist."""
    path = tmp_path / "edgelist.parquet"
    pl.DataFrame(
        {
            "umi1": ["a", "b", "a"],
            "umi2": ["d", "c", "d"],
            "read_count": [10, 20, 30],
            "uei_count": [1, 2, 3],
        }
    ).write_parquet(path)

    stats = get_count_statistics(path)

    assert stats == {
        "n_edges": 3,
        "n_reads": 60,
        "n_molecules": 6,
        "n_umi": 4,
    }
