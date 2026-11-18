"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pixelator.pna.graph.community_detection import (
    RefinementOptions,
    StagedRefinementOptions,
    calculate_post_recovery_component_statistics,
    refine_component,
    run_leiden_refinement,
)
from pixelator.pna.graph.component_recovery_utils import (
    write_hive_partitioned_edgelist_without_out_of_size_bound_components,
)
from pixelator.pna.graph.report import GraphStatistics
from pixelator.pna.utils.duckdb_utils import DuckdbPerThreadMemoryError


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


def test_calculate_post_recovery_component_statistics_includes_discarded_large_components():
    """Components discarded early for being too large still count toward "largest component"."""
    edgelist = pl.DataFrame(
        {
            "umi1": [1, 1, 100],
            "umi2": [10, 11, 200],
            "component": ["A", "A", "B"],
        }
    )
    # Discarded upstream for being too large; its edges are gone, only its size is known.
    discarded_large_components = pl.DataFrame(
        {
            "component": ["huge"],
            "n_umi": pl.Series([1000], dtype=pl.UInt32),
            "n_edges": pl.Series([5000], dtype=pl.UInt32),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        edgelist.write_parquet(path)
        stats = GraphStatistics()
        out = calculate_post_recovery_component_statistics(
            path, stats, discarded_large_components=discarded_large_components
        )
    finally:
        path.unlink(missing_ok=True)

    assert out.component_count_post_recovery == 3
    assert out.fraction_nodes_in_largest_component_post_recovery == pytest.approx(
        1000 / 1005
    )


def test_calculate_post_recovery_component_statistics_with_real_discarded_frame(
    tmp_path: Path,
):
    """The discarded-components frame produced upstream must concat with n_edges from pl.len().

    Regression test for a schema mismatch (Int64 vs UInt32) between the two `n_edges`
    columns being concatenated.
    """
    edgelist = pl.DataFrame(
        {
            "umi1": [1, 1, 100],
            "umi2": [10, 11, 200],
            "component": ["A", "A", "B"],
        }
    )
    kept_path = tmp_path / "edgelist.parquet"
    edgelist.write_parquet(kept_path)

    # A component large enough to be pruned upstream, producing the real
    # discarded-components frame (as returned by community detection).
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["huge", "huge"],
            "umi1": ["a", "c"],
            "umi2": ["b", "d"],
        }
    ).write_parquet(partitioned)
    _, discarded_large_components = (
        write_hive_partitioned_edgelist_without_out_of_size_bound_components(
            input_edgelist_path=partitioned,
            min_component_size_to_prune=0,
            max_component_size_to_prune=3,
            working_dir=tmp_path,
        )
    )

    stats = GraphStatistics()
    out = calculate_post_recovery_component_statistics(
        kept_path, stats, discarded_large_components=discarded_large_components
    )

    assert out.component_count_post_recovery == 3
    assert out.edge_count_post_recovery == 5


def test_refine_component_splits_on_crossing_edge_and_keeps_all_columns(
    tmp_path: Path,
):
    """Two dense blocks joined by one bridge edge are split, and no column is dropped."""
    block_a = [(u1, u2) for u1 in range(1, 21) for u2 in range(101, 121)]
    block_b = [(u1, u2) for u1 in range(21, 41) for u2 in range(121, 141)]
    bridge = [(1, 121)]
    edges = block_a + block_b + bridge
    edgelist = pl.DataFrame(
        {
            "marker_1": ["CD3"] * len(edges),
            "marker_2": ["CD4"] * len(edges),
            "umi1": pl.Series([u1 for u1, _ in edges], dtype=pl.UInt64),
            "umi2": pl.Series([u2 for _, u2 in edges], dtype=pl.UInt64),
            "marker_sequence_1": [f"SEQ1_{u1}" for u1, _ in edges],
            "marker_sequence_2": [f"SEQ2_{u2}" for _, u2 in edges],
            "read_count": pl.Series([1] * len(edges), dtype=pl.UInt32),
        }
    )
    partitions = tmp_path / "edgelists"
    (partitions / "component=comp").mkdir(parents=True)
    edgelist.write_parquet(partitions / "component=comp" / "data_0.parquet")

    comp_sizes, n_removed_edges, discard_sizes = refine_component(
        component_id="comp",
        component_edgelists_path=partitions,
        refinement_options=RefinementOptions(
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=1.0,
            min_component_size_to_prune=1,
        ),
    )

    assert n_removed_edges == 1
    assert sorted(comp_sizes.to_list()) == [40, 40]
    assert discard_sizes.empty
    assert not (partitions / "component=comp").exists()

    result = pl.read_parquet(partitions, hive_partitioning=True)
    assert set(result.columns) == set(edgelist.columns) | {"component"}
    assert result["component"].n_unique() == 2
    assert result["component"].str.starts_with("comp_").all()
    expected = edgelist.filter(~((pl.col("umi1") == 1) & (pl.col("umi2") == 121)))
    assert_frame_equal(
        result.drop("component").select(edgelist.columns),
        expected,
        check_row_order=False,
    )


@patch("pixelator.pna.utils.duckdb_utils.duckdb.connect")
def test_run_leiden_refinement_raises_when_not_enough_memory_for_duckdb_workers(
    mock_connect: MagicMock,
) -> None:
    """DuckDB memory split across workers must leave at least 1 MiB per thread.

    Args:
        mock_connect: Mock connect.
    """
    mock_con = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_con
    mock_connect.return_value.__exit__.return_value = None
    mock_con.execute.return_value.fetchone.return_value = ("1000 B",)

    component_sizes = pl.DataFrame(
        {
            "component": pl.Series(dtype=pl.Utf8),
            "n_umi": pl.Series(dtype=pl.UInt32),
        }
    )

    with pytest.raises(
        DuckdbPerThreadMemoryError, match="Not enough memory to share DuckDB work"
    ):
        run_leiden_refinement(
            component_edgelists_path=Path("/nonexistent/edgelists"),
            refinement_options=StagedRefinementOptions(),
            component_stats=GraphStatistics(),
            component_sizes=component_sizes,
            max_workers=10_000,
        )
