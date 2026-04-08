"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from pixelator.pna.graph.community_detection import (
    calculate_post_recovery_component_statistics,
)
from pixelator.pna.graph.report import GraphStatistics


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
