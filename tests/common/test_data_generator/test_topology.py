"""Tests for the graph topology generation.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pixelator.common.utils.test_data_generator.topology import generate_cell_graph


@pytest.mark.parametrize("rng", [0, 1, 42])
def test_generate_cell_graph(rng):
    """Verify node count, edge count and bipartite structure of the edge list."""
    n_nodes = 20
    min_neighbors = 10
    n_edges = 50

    edgelist = generate_cell_graph(n_nodes, n_edges, min_neighbors, rng=rng)

    # correct number of edges
    assert edgelist.height == n_edges

    # no duplicate edges
    assert edgelist.n_unique() == edgelist.height

    node1 = set(edgelist["node1"].to_list())
    node2 = set(edgelist["node2"].to_list())

    # node1 and node2 are mutually exclusive (the graph is bipartite)
    assert node1.isdisjoint(node2)

    # correct number of nodes
    assert len(node1 | node2) == n_nodes
    assert node1 | node2 == set(range(n_nodes))


def test_generate_cell_graph_reproducible():
    """Verify the edge list is deterministic for a given rng."""
    params = dict(n_nodes=20, n_edges=50, min_neighbors=10)

    # the same integer seed yields an identical edge list
    first = generate_cell_graph(**params, rng=0)
    assert_frame_equal(first, generate_cell_graph(**params, rng=0))

    # a Generator seeded identically yields the same edge list as the int seed
    assert_frame_equal(first, generate_cell_graph(**params, rng=np.random.default_rng(0)))

    # a different seed yields a different edge list
    assert not first.equals(generate_cell_graph(**params, rng=1))
