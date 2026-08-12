"""Robustness tests for the coarsened PMDS layout.

Copyright © 2026 Pixelgen Technologies AB.

The layout is produced by a Leiden coarsening followed by a PMDS
eigendecomposition, so the exact coordinates depend on the numeric backend
(scipy/BLAS) and are only defined up to rotation, reflection and scaling.
Instead of pinning coordinates, these tests assert backend-independent
properties on a synthetic cell graph: determinism for a fixed seed, structural
validity, and that the layout preserves the graph neighborhood (adjacent nodes
end up closer than non-adjacent ones).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from pixelator.common.graph.backends.implementations._networkx import (
    coarsened_pmds_layout,
)
from tests.common.data_generator.topology import generate_cell_graph

WEIGHT_MODES = ["tp", "crossing_edges"]


@pytest.fixture(scope="module")
def layout_graph() -> nx.Graph:
    """A connected synthetic cell graph with nodes on a unit sphere.

    ``coarsened_pmds_layout`` requires a connected, undirected graph with at
    least ``pivots`` (default 200) nodes, so we generate a larger graph and keep
    its largest connected component.
    """
    edgelist = generate_cell_graph(n_nodes=800, n_edges=2500, min_neighbors=25, rng=42)
    graph = nx.Graph()
    graph.add_edges_from(edgelist.select(["node1", "node2"]).iter_rows())
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()


def _mean_pair_distance(layout: dict, pairs: list[tuple[int, int]]) -> float:
    """Mean Euclidean distance in layout space over a list of node pairs."""
    deltas = np.array([layout[int(a)] - layout[int(b)] for a, b in pairs])
    return float(np.linalg.norm(deltas, axis=1).mean())


def _sample_non_edges(
    graph: nx.Graph, n_samples: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Sample ``n_samples`` node pairs that are not connected by an edge."""
    nodes = list(graph.nodes())
    non_edges: list[tuple[int, int]] = []
    while len(non_edges) < n_samples:
        a, b = (int(x) for x in rng.choice(nodes, size=2, replace=False))
        if not graph.has_edge(a, b):
            non_edges.append((a, b))
    return non_edges


@pytest.mark.parametrize("weight_edges_by", WEIGHT_MODES)
def test_coarsened_pmds_layout_is_deterministic(layout_graph, weight_edges_by):
    """The same seed produces an identical layout across runs."""
    first = coarsened_pmds_layout(
        layout_graph, weight_edges_by=weight_edges_by, seed=42
    )
    second = coarsened_pmds_layout(
        layout_graph, weight_edges_by=weight_edges_by, seed=42
    )

    assert first.keys() == second.keys()
    for node in first:
        np.testing.assert_array_equal(first[node], second[node])


@pytest.mark.parametrize("weight_edges_by", WEIGHT_MODES)
def test_coarsened_pmds_layout_returns_valid_coordinates(layout_graph, weight_edges_by):
    """Every node gets a finite 3D coordinate."""
    layout = coarsened_pmds_layout(
        layout_graph, weight_edges_by=weight_edges_by, seed=42
    )

    assert set(layout.keys()) == set(layout_graph.nodes())
    coords = np.vstack([layout[node] for node in layout_graph.nodes()])
    assert coords.shape == (layout_graph.number_of_nodes(), 3)
    assert np.isfinite(coords).all()


@pytest.mark.parametrize("weight_edges_by", WEIGHT_MODES)
def test_coarsened_pmds_layout_preserves_neighborhood(layout_graph, weight_edges_by):
    """Adjacent nodes are placed much closer than non-adjacent ones.

    This is invariant to rotation, reflection and scaling of the embedding, so
    it holds regardless of the scipy/BLAS backend that produced the coordinates.
    """
    layout = coarsened_pmds_layout(
        layout_graph, weight_edges_by=weight_edges_by, seed=42
    )

    edges = list(layout_graph.edges())
    non_edges = _sample_non_edges(layout_graph, len(edges), np.random.default_rng(0))

    mean_edge_distance = _mean_pair_distance(layout, edges)
    mean_non_edge_distance = _mean_pair_distance(layout, non_edges)

    assert mean_edge_distance < 0.5 * mean_non_edge_distance
