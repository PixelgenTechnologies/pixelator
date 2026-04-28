"""Tests for adaptive_core_expansion.

Modified to match the structure and intent of the R test suite.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pathlib import Path

import networkx as nx
import pytest

from pixelator import read_pna as read
from pixelator.common.graph.adaptive_core_expansion import adaptive_core_expansion

PXL_FILE = (
    Path(__file__).parents[2]
    / "tests"
    / "pna"
    / "data"
    / "PNA055_Sample07_S7.layout.pxl"
)
COMPONENT_ID = "0a45497c6bfbfb22"


@pytest.fixture(scope="module")
def graph_from_pxl():
    """Load a single graph from the minimal PNA PBMC pxl file."""
    pg_data = read(str(PXL_FILE))
    g = pg_data.filter(components=COMPONENT_ID).edgelist().iterator().__next__().graph
    return g


@pytest.mark.slow
def test_adaptive_core_expansion_works_as_expected(graph_from_pxl):
    """Test that adaptive_core_expansion works for various valid inputs."""
    # Basic execution and exact count check
    res = adaptive_core_expansion(graph_from_pxl)
    partitions = list(nx.get_node_attributes(res.raw, "partition").values())
    partition_counts = {
        "high": partitions.count("high"),
        "low": partitions.count("low"),
    }
    # Expected counts for COMPONENT_ID in PNA055_Sample07_S7.layout.pxl
    assert partition_counts == {"high": 43144, "low": 399}

    # Test with different valid parameter combinations (expect no errors)
    adaptive_core_expansion(graph_from_pxl, k=2)
    adaptive_core_expansion(graph_from_pxl, max_iter=10)
    adaptive_core_expansion(graph_from_pxl, min_seed_pct=0.2)
    adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold=100)
    adaptive_core_expansion(graph_from_pxl, select_LCC=False)


def test_adaptive_core_expansion_fails_with_invalid_input(graph_from_pxl):
    """Test that adaptive_core_expansion raises errors for invalid inputs."""
    # Invalid graph input
    with pytest.raises((AttributeError, TypeError)):
        adaptive_core_expansion("Invalid")

    # Invalid types for numerical parameters
    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, k="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, max_iter="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, min_seed_pct="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold="Invalid")

    # Out of bounds numerical values
    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, max_iter=0)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, max_iter=1001)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, min_seed_pct=2.0)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold=-1)
