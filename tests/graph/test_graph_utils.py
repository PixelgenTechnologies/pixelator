"""Tests for the graph utils module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pixelator.graph import (
    Graph,
    components_metrics,
    create_node_markers_counts,
    edgelist_metrics,
    update_edgelist_membership,
)


def add_random_names_to_vertexes(graph: Graph) -> None:
    """Add some random names to vertices on the graph."""
    for vertex in graph.vs:
        vertex["name"] = random_sequence(21)


@pytest.fixture(name="pentagram_graph")
def pentagram_graph_fixture():
    """Build a graph in the shape of a five pointed star."""
    # Construct a graph in the shape of a five pointed
    # star with a single marker in each point
    edges = [
        (0, 2),
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 4),
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 2),
    ]
    edgelist = pd.DataFrame(edges, columns=["upia", "upib"])
    g = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=False,
        simplify=True,
        use_full_bipartite=True,
    )

    default_marker = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    g.vs[0]["markers"] = dict(default_marker, A=1)
    g.vs[1]["markers"] = dict(default_marker, B=1)
    g.vs[2]["markers"] = dict(default_marker, C=1)
    g.vs[3]["markers"] = dict(default_marker, D=1)
    g.vs[4]["markers"] = dict(default_marker, E=1)
    return g


def random_sequence(size: int) -> str:
    """Create a random sequence of size (size)."""
    return "".join(random.choices("CGTA", k=size))


def test_build_graph_full_bipartite(full_graph_edgelist: pd.DataFrame):
    """Build full-bipartite graph."""
    graph = Graph.from_edgelist(
        edgelist=full_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    assert graph.vcount() == 50 + 50
    assert graph.ecount() == 50 * 50
    assert "markers" in graph.vs.attributes()
    assert sorted(list(graph.vs[0]["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == ["name", "markers", "type", "pixel_type"]


def test_build_graph_a_node_projected(full_graph_edgelist: pd.DataFrame):
    """Build an A-node projected graph."""
    graph = Graph.from_edgelist(
        edgelist=full_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=False,
    )
    assert graph.vcount() == 50
    assert graph.ecount() == ((50 * 50) / 2) - (50 / 2)
    assert "markers" in graph.vs.attributes()
    assert sorted(list(graph.vs[0]["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == ["name", "markers", "type", "pixel_type"]


def test_components_metrics(full_graph_edgelist: pd.DataFrame):
    """Test generating component metrics."""
    # test component metrics
    metrics = components_metrics(edgelist=full_graph_edgelist)
    assert_array_equal(
        metrics.to_numpy(),
        np.array([[100, 2500, 2, 50, 50, 1, 2500, 1, 1, 50, 50, 50, 50, 1]]),
    )
    assert sorted(metrics.columns) == sorted(
        [
            "vertices",
            "edges",
            "antibodies",
            "upia",
            "upib",
            "umi",
            "reads",
            "mean_reads",
            "median_reads",
            "mean_upia_degree",
            "median_upia_degree",
            "mean_umi_per_upia",
            "median_umi_per_upia",
            "upia_per_upib",
        ]
    )


def test_create_node_markers_counts_k_eq_0(pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 0."""
    result = create_node_markers_counts(graph=pentagram_graph, k=0)

    expected = pd.DataFrame(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected.columns.name = "markers"
    assert_frame_equal(result, expected)


def test_create_node_markers_counts_k_eq_1(pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 1."""
    result = create_node_markers_counts(graph=pentagram_graph, k=1)

    expected = pd.DataFrame(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected.columns.name = "markers"
    assert_frame_equal(result, expected)


def test_create_node_markers_counts_k_eq_2(pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 2."""
    result = create_node_markers_counts(graph=pentagram_graph, k=2)

    expected = pd.DataFrame(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected.columns.name = "markers"
    assert_frame_equal(result, expected)


def test_create_node_markers_counts_k_eq_2_with_mean(pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 2, mean values."""
    result = create_node_markers_counts(
        graph=pentagram_graph, k=2, normalization="mean"
    )

    expected = pd.DataFrame(
        [
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected.columns.name = "markers"
    assert_frame_equal(result, expected)


def test_create_node_markers_counts(random_graph_edgelist: pd.DataFrame):
    """Test build a node marker matrix with a neigbourhood of 0."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=0)
    assert counts.shape == (graph.vcount(), 2)
    # it is a fully connected graph so each antibody should cover all edges
    assert counts["A"].sum() == graph.ecount() + 1
    assert counts["B"].sum() == graph.ecount() + 1


def test_create_node_markers_counts_with_neighbourhood_1_with_mean_normalization(
    random_graph_edgelist: pd.DataFrame,
):
    """Test build a node marker matrix with a neigbourhood of 1, with the mean value."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=1, normalization="mean")
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == pytest.approx(715.79, abs=0.01)
    assert counts["B"].sum() == pytest.approx(706.57, abs=0.01)


def test_create_node_markers_counts_with_neighbourhood_2(
    random_graph_edgelist: pd.DataFrame,
):
    """Test build a node marker matrix with a neigbourhood of 2."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=2)
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == 8027
    assert counts["B"].sum() == 7870


def test_create_node_markers_counts_column_order(
    random_graph_edgelist: pd.DataFrame,
):
    """Columns should always be returned in alphabetical sort order."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=0)
    assert counts.columns.to_list() == ["A", "B"]


def test_create_node_markers_counts_k_eq_1_with_mean(pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 1, with the mean value."""
    result = create_node_markers_counts(
        graph=pentagram_graph, k=1, normalization="mean"
    )

    expected = pd.DataFrame(
        [
            [1 / 3, 0, 1 / 3, 1 / 3, 0],
            [0, 1 / 3, 0, 1 / 3, 1 / 3],
            [1 / 3, 0, 1 / 3, 0, 1 / 3],
            [1 / 3, 1 / 3, 0, 1 / 3, 0],
            [0, 1 / 3, 1 / 3, 0, 1 / 3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected.columns.name = "markers"
    assert_frame_equal(result, expected)


def test_create_node_markers_counts_with_neighbourhood_1(
    random_graph_edgelist: pd.DataFrame,
):
    """Test build a node marker matrix with a neigbourhood of 1."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=1)
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == 2803
    assert counts["B"].sum() == 2759


def test_edgelist_metrics(full_graph_edgelist: pd.DataFrame):
    """Test generating edgelist metrics."""
    metrics = edgelist_metrics(full_graph_edgelist)
    assert metrics == {
        "components": 1,
        "components_modularity": 0.0,
        "edges": 2500,
        "frac_largest_edges": 1.0,
        "frac_largest_vertices": 1.0,
        "markers": 2,
        "vertices": 100,
        "total_upia": 50,
        "total_upib": 50,
        "total_umi": 1,
        "total_upi": 100,
        "frac_upib_upia": 1.0,
        "upia_degree_mean": 50.0,
        "upia_degree_median": 50.0,
    }


def test_update_edgelist_membership(data_root):
    """Test updating the edgelist membership."""
    edgelist = pd.read_csv(str(data_root / "test_edge_list.csv"))
    result = update_edgelist_membership(edgelist.copy(), prefix="PXLCMP")

    assert "component" not in edgelist.columns
    assert set(result["component"].unique()) == {
        "PXLCMP0000000",
        "PXLCMP0000001",
        "PXLCMP0000002",
        "PXLCMP0000003",
        "PXLCMP0000004",
    }
