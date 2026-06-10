"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pixelator.pna.graph.community_detection import (
    StagedRefinementOptions,
    calculate_post_recovery_component_statistics,
    map_working_to_original_umi_names,
    run_leiden_refinement,
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


def test_map_working_to_original_umi_names(tmp_path: Path) -> None:
    """Working UMI names in umi1 and umi2 are replaced with original names."""
    node_map_path = tmp_path / "node_map.parquet"
    edgelist_path = tmp_path / "edgelist.parquet"
    pl.DataFrame(
        {
            "working_name": ["w1", "w2", "w3", "w4"],
            "original_name": ["o1", "o2", "o3", "o4"],
        }
    ).write_parquet(node_map_path)
    pl.DataFrame(
        {
            "component": ["c1", "c1"],
            "umi1": ["w1", "w3"],
            "umi2": ["w2", "w4"],
            "read_count": [1, 2],
        }
    ).write_parquet(edgelist_path)

    mapped_path = map_working_to_original_umi_names(
        edgelist_path, node_map_path, tmp_path
    )
    mapped = pl.read_parquet(mapped_path)

    assert mapped["umi1"].to_list() == ["o1", "o3"]
    assert mapped["umi2"].to_list() == ["o2", "o4"]
    assert mapped["component"].to_list() == ["c1", "c1"]
    assert mapped["read_count"].to_list() == [1, 2]


def test_map_working_to_original_umi_names_raises_for_missing_mapping(
    tmp_path: Path,
) -> None:
    """Mapping fails when an UMI in input is missing in the node map."""
    node_map_path = tmp_path / "node_map.parquet"
    edgelist_path = tmp_path / "edgelist.parquet"
    pl.DataFrame(
        {
            "working_name": ["w1"],
            "original_name": ["o1"],
        }
    ).write_parquet(node_map_path)
    pl.DataFrame(
        {
            "component": ["c1"],
            "umi1": ["w1"],
            "umi2": ["w_missing"],
        }
    ).write_parquet(edgelist_path)

    with pytest.raises(ValueError, match="Missing UMI mapping"):
        map_working_to_original_umi_names(edgelist_path, node_map_path, tmp_path)


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
