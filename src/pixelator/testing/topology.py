"""Unit sphere graph topology generation for test data.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import KDTree


def generate_cell_graph(
    n_nodes: int, n_edges: int, min_neighbors: int, strategy="subsample", rng=None
) -> pl.DataFrame:
    """Generate a cell graph with nodes distributed on a unit sphere.

    This function generates a graph where:
    - Nodes are points distributed on the surface of a unit sphere.
    - Edges are created based on the `min_neighbors` nearest neighbors for each node.
    - The graph is made bipartite by removing edges between nodes of the same parity.
    - The number of edges is subsampled to match `n_edges` if necessary.

    Args:
        n_nodes: The number of nodes in the graph.
        n_edges: The target number of edges in the graph.
        min_neighbors: The minimum number of neighbors for each node.
        strategy: strategy to use match degree distribution
        rng: a seed or numpy Generator for the random number generator

    Returns:
        A polars DataFrame representing the edge list of the graph.

    Raises:
        ValueError: If `strategy` is not one of `subsample` or `negative-binomial`.
    """
    # Generate point cloud
    rng = np.random.default_rng(rng)
    points = rng.standard_normal((n_nodes, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Generate edge list
    tree = KDTree(points)
    edgelist_df = pl.DataFrame(
        [
            (i, j) if i % 2 else (j, i)
            for i in range(n_nodes)
            for j in tree.query(points[i], k=min_neighbors + 1)[1][1:]
            if (i % 2) != (j % 2)
        ],
        schema=["node1", "node2"],
        orient="row",
    ).unique(maintain_order=True)

    # Subsample degree distribution
    match strategy:
        case "subsample":
            # polars sample needs an int seed, so derive one from the generator
            edgelist_df = edgelist_df.sample(n_edges, seed=int(rng.integers(1 << 32)))
        case "negative-binomial":
            raise NotImplementedError
        case _:
            raise ValueError("Unknown strategy `%s`" % strategy)

    return edgelist_df
