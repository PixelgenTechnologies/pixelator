"""Test configuration generally useful in the graph module.


Copyright © 2023 Pixelgen Technologies AB.
"""

import networkx as nx
import pandas as pd
import pytest

from pixelator.graph import Graph
from pixelator.graph.backends.implementations import graph_backend
from tests.graph.networkx.test_tools import add_random_names_to_vertexes


@pytest.fixture(name="output_dir")
def output_dir_fixture(tmp_path):
    """Fix an output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir


@pytest.fixture(name="metrics_file")
def metrics_file_fixture(tmp_path):
    """Fix a metrics file."""
    metrics_file = tmp_path / "metrics.json"
    yield metrics_file


@pytest.fixture(name="input_edgelist")
def input_edgelist_fixture(tmp_path, edgelist_with_communities: pd.DataFrame):
    """Fix an input edgelist."""
    input_edgelist = tmp_path / "tmp_edgelist.parquet"
    edgelist_with_communities.to_parquet(
        input_edgelist, engine="fastparquet", compression="zstd", index=False
    )
    yield input_edgelist


@pytest.fixture(name="graph_with_communities")
def graph_with_communities_fixture(edgelist_with_communities: pd.DataFrame):
    """Fix a bipartite multi-graph with communities and no marker counts."""
    # build the graph from the edge list
    graph = Graph.from_edgelist(
        edgelist=edgelist_with_communities,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )

    return graph


@pytest.fixture(name="graph_without_communities")
def graph_without_communities_fixture():
    """Fix a full graph with random names in vertexes."""

    graph = Graph.from_raw(nx.fast_gnp_random_graph(100, p=0.1, seed=10))
    # Remove any unattached nodes, since that messes up the community
    # detection
    graph = graph.connected_components().giant()
    add_random_names_to_vertexes(graph)
    return graph


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
    edgelist = pd.DataFrame(
        edges, columns=["upia", "upib"], index=[str(i) for i in range(len(edges))]
    )
    GraphBackend = graph_backend()
    g = Graph(
        backend=GraphBackend.from_edgelist(
            edgelist=edgelist,
            add_marker_counts=False,
            simplify=True,
            use_full_bipartite=True,
        )
    )

    default_marker = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    g.vs.get_vertex(0)["markers"] = dict(default_marker, A=1)
    g.vs.get_vertex(1)["markers"] = dict(default_marker, B=1)
    g.vs.get_vertex(2)["markers"] = dict(default_marker, C=1)
    g.vs.get_vertex(3)["markers"] = dict(default_marker, D=1)
    g.vs.get_vertex(4)["markers"] = dict(default_marker, E=1)

    g.vs.get_vertex(0)["pixel_type"] = "A"
    g.vs.get_vertex(1)["pixel_type"] = "B"
    g.vs.get_vertex(2)["pixel_type"] = "A"
    g.vs.get_vertex(3)["pixel_type"] = "B"
    g.vs.get_vertex(4)["pixel_type"] = "A"

    g.vs.get_vertex(0)["name"] = "AAAA"
    g.vs.get_vertex(1)["name"] = "TTTT"
    g.vs.get_vertex(2)["name"] = "CCCC"
    g.vs.get_vertex(3)["name"] = "GGGG"
    g.vs.get_vertex(4)["name"] = "AATT"
    return g
