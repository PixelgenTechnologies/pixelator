"""Tests for the spectral layout algorithm.

Copyright © 2026 Pixelgen Technologies AB.
"""

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

from pixelator.common.graph.backends.implementations._networkx import spectral_layout
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset
from tests.common.data_generator.topology import generate_cell_graph

METHOD_CONFIGS = [("eigen", True), ("eigen", False), ("psvd", True)]


@pytest.fixture(scope="module")
def layout_graph() -> nx.Graph:
    edgelist = generate_cell_graph(n_nodes=800, n_edges=2500, min_neighbors=25, rng=42)
    graph = nx.Graph()
    graph.add_edges_from(edgelist.select(["node1", "node2"]).iter_rows())
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    # bipartite by parity -> assign pixel_type for the psvd method
    nx.set_node_attributes(
        graph, {n: ("A" if n % 2 == 0 else "B") for n in graph.nodes()}, "pixel_type"
    )
    return graph


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


def _sample_node_pairs(n_nodes: int, n_pairs: int, rng: np.random.Generator):
    """Sample distinct-node index pairs, resampling only the rare self-pairs."""
    if n_nodes < 2:
        raise ValueError(f"Cannot sample distinct node pairs with n_nodes={n_nodes}")
    i = rng.integers(0, n_nodes, size=n_pairs)
    j = rng.integers(0, n_nodes, size=n_pairs)
    self_pairs = i == j
    while self_pairs.any():
        j[self_pairs] = rng.integers(0, n_nodes, size=self_pairs.sum())
        self_pairs = i == j
    return i, j


@pytest.mark.slow
def test_spectral_layout_correlates_with_cpmds_on_real_component(
    pna_pxl_dataset: PNAPixelDataset,
):
    """Compare the spectral layout against the default CPMDS layout.

    We compare pairwise Euclidean distances between the same pairs of nodes,
    which is invariant to rotation and scaling. Pairwise distances from
    spectral_3d should rank-correlate with coarsened_pmds_3d.
    """
    n_pairs = 10_000
    seed = 42

    obs = pna_pxl_dataset.adata().obs
    component_id = obs["n_edges"].idxmax()

    edgelist = (
        pna_pxl_dataset.filter(components=[component_id]).edgelist().to_record_batches()
    )
    graph = PNAGraph.from_record_batches(edgelist)

    cpmds = graph.layout_coordinates(
        layout_algorithm="coarsened_pmds_3d",
        get_node_marker_matrix=False,
        random_seed=seed,
    ).set_index("index")
    spectral = graph.layout_coordinates(
        layout_algorithm="spectral_3d", get_node_marker_matrix=False, random_seed=seed
    ).set_index("index")

    assert set(cpmds.index) == set(spectral.index)
    nodes = cpmds.index
    coords_cpmds = cpmds.loc[nodes, ["x", "y", "z"]].to_numpy()
    coords_spectral = spectral.loc[nodes, ["x", "y", "z"]].to_numpy()

    rng = np.random.default_rng(seed)
    i, j = _sample_node_pairs(len(nodes), n_pairs, rng)

    dist_cpmds = np.linalg.norm(coords_cpmds[i] - coords_cpmds[j], axis=1)
    dist_spectral = np.linalg.norm(coords_spectral[i] - coords_spectral[j], axis=1)

    correlation = spearmanr(dist_cpmds, dist_spectral)

    assert correlation.statistic > 0.5


@pytest.mark.parametrize("method, normalize", METHOD_CONFIGS)
def test_spectral_layout_is_deterministic(layout_graph, method, normalize):
    """The same seed produces an identical layout across runs."""
    first = spectral_layout(layout_graph, method=method, normalize=normalize, seed=42)
    second = spectral_layout(layout_graph, method=method, normalize=normalize, seed=42)

    assert first.keys() == second.keys()
    for node in first:
        np.testing.assert_array_equal(first[node], second[node])


@pytest.mark.parametrize("method, normalize", METHOD_CONFIGS)
def test_spectral_layout_returns_valid_coordinates(layout_graph, method, normalize):
    """Every node gets a finite 3D coordinate."""
    layout = spectral_layout(layout_graph, method=method, normalize=normalize, seed=42)

    assert set(layout.keys()) == set(layout_graph.nodes())
    coords = np.vstack([layout[node] for node in layout_graph.nodes()])
    assert coords.shape == (layout_graph.number_of_nodes(), 3)
    assert np.isfinite(coords).all()


@pytest.mark.parametrize("method, normalize", METHOD_CONFIGS)
def test_spectral_layout_preserves_neighborhood(layout_graph, method, normalize):
    """Adjacent nodes are placed much closer than non-adjacent ones.

    This is invariant to rotation, reflection and scaling of the embedding, so
    it holds regardless of the backend that produced the coordinates.
    """
    layout = spectral_layout(layout_graph, method=method, normalize=normalize, seed=42)

    edges = list(layout_graph.edges())
    non_edges = _sample_non_edges(layout_graph, len(edges), np.random.default_rng(0))

    mean_edge_distance = _mean_pair_distance(layout, edges)
    mean_non_edge_distance = _mean_pair_distance(layout, non_edges)

    assert mean_edge_distance < 0.5 * mean_non_edge_distance
