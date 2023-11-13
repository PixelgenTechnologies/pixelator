"""Test configuration generally useful in the graph module.


Copyright (c) 2023 Pixelgen Technologies AB.
"""

import os

import networkx as nx
import pandas as pd
import pytest
from pixelator.graph import Graph

from tests.graph.igraph.test_tools import full_graph
from tests.graph.test_graph_utils import add_random_names_to_vertexes
from tests.test_tools import enforce_edgelist_types_for_tests


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
    input_edgelist = tmp_path / "tmp_edgelist.csv"
    edgelist_with_communities.to_csv(
        input_edgelist,
        header=True,
        index=False,
    )
    assert len(edgelist_with_communities["component"].unique()) == 1
    edgelist_with_communities = enforce_edgelist_types_for_tests(
        edgelist_with_communities
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

    # Somewhat hacky solution make sure this works with both
    # the igraph and networkx tests

    if os.environ.get("ENABLE_NETWORKX_BACKEND"):
        graph = Graph.from_raw(nx.fast_gnp_random_graph(100, p=0.1, seed=10))
        # Remove any unattached nodes, since that messes up the community
        # detection
        graph = graph.connected_components().giant()
        add_random_names_to_vertexes(graph)
        return graph
    graph = full_graph(n=100)
    add_random_names_to_vertexes(graph)
    return graph


@pytest.fixture
def enable_backend(request):
    previous_environment = os.environ
    if request.param == "networkx":
        new_environment = previous_environment.copy()
        new_environment["ENABLE_NETWORKX_BACKEND"] = True
        os.environ = new_environment
    yield
    os.environ = previous_environment
