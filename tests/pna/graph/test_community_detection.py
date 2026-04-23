"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pixelator.pna.graph.community_detection import (
    StagedRefinementOptions,
    calculate_post_recovery_component_statistics,
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


@patch("pixelator.pna.utils.duckdb_utils.duckdb.connect")
def test_run_leiden_refinement_raises_when_not_enough_memory_for_duckdb_workers(
    mock_connect: MagicMock,
) -> None:
    """DuckDB memory split across workers must leave at least 1 MiB per thread."""
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
