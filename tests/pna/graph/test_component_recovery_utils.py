"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

import polars as pl

from pixelator.pna.graph.component_recovery_utils import (
    get_count_statistics,
    write_hive_partitioned_edgelist_without_small_components,
)


def test_get_count_statistics(tmp_path: Path) -> None:
    """Test count statistics from a Parquet edgelist.

    Args:
        tmp_path: Tmp path.
    """
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


def test_write_hive_partitioned_edgelist_without_small_components_prunes(
    tmp_path: Path,
) -> None:
    """Test that small components are omitted and listed as discarded.

    Args:
        tmp_path: Tmp path.
    """
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["keep", "keep", "keep", "drop"],
            "umi1": ["a", "c", "e", "x"],
            "umi2": ["b", "d", "f", "y"],
        }
    ).write_parquet(partitioned)

    out_path, discarded = write_hive_partitioned_edgelist_without_small_components(
        input_edgelist_path=partitioned,
        min_component_size_to_prune=3,
        working_dir=tmp_path,
    )

    assert out_path == tmp_path / "hive_partitioned_edgelist.parquet"
    kept = pl.scan_parquet(out_path, hive_schema={"component": pl.String}).collect()
    assert kept["component"].unique().to_list() == ["keep"]
    assert kept.height == 3

    discarded_sorted = discarded.sort("component")
    assert discarded_sorted["component"].to_list() == ["drop"]
    assert discarded_sorted["n_umi"].to_list() == [2]


def test_write_hive_partitioned_edgelist_without_small_components_nothing_discarded(
    tmp_path: Path,
) -> None:
    """When every component meets the threshold, discarded frame is empty and all rows are kept.

    Args:
        tmp_path: Tmp path.
    """
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["a", "a", "b"],
            "umi1": ["u1", "u3", "w1"],
            "umi2": ["u2", "u4", "w2"],
        }
    ).write_parquet(partitioned)

    _, discarded = write_hive_partitioned_edgelist_without_small_components(
        input_edgelist_path=partitioned,
        min_component_size_to_prune=2,
        working_dir=tmp_path,
    )

    assert discarded.height == 0
    kept = pl.scan_parquet(
        tmp_path / "hive_partitioned_edgelist.parquet",
        hive_schema={"component": pl.String},
    ).collect()
    assert kept.height == 3
