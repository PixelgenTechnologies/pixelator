"""Test the graph module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import os
import random

import pandas as pd
import pytest
from pixelator.graph import Graph

from tests.test_tools import enforce_edgelist_types_for_tests
from tests.graph.test_graph_utils import random_sequence


def create_simple_edge_list_from_graph(
    graph: Graph, random_markers: bool = False
) -> pd.DataFrame:
    """Convert a graph to edge list (dataframe)."""
    df = graph.get_edge_dataframe()
    df_vert = graph.get_vertex_dataframe()
    df["source"].replace(df_vert["name"], inplace=True)
    df["target"].replace(df_vert["name"], inplace=True)

    # rename source/target columns
    df = df.rename(columns={"source": "upib", "target": "upia"})

    # add attributes
    n_row = df.shape[0]
    df["count"] = 1
    df["umi_unique_count"] = 1
    df["upi_unique_count"] = 1
    if random_markers:
        df["marker"] = random.choices(
            ["A", "B", "C", "D", "E", "F", "G"], weights=[4, 2, 3, 1, 1, 1, 1], k=n_row
        )
    else:
        df["marker"] = "B"
        df.iloc[0 : int(n_row / 2), 5] = "A"
    df["umi"] = [random_sequence(6) for _ in range(len(df))]
    df["upib"] = df["upib"].astype(str)
    df["upia"] = df["upia"].astype(str)
    marker_to_seq = {
        "A": "ACTG",
        "B": "CTGA",
        "C": "TGAC",
        "D": "GACT",
        "E": "GTCA",
        "F": "TCAG",
        "G": "CAGT",
    }
    df["sequence"] = df["marker"].map(marker_to_seq)
    df = enforce_edgelist_types_for_tests(df)
    return df


@pytest.fixture
def enable_backend(request):
    previous_environment = os.environ
    if request.param == "networkx":
        new_environment = previous_environment.copy()
        new_environment["ENABLE_NETWORKX_BACKEND"] = True
        os.environ = new_environment
    yield
    os.environ = previous_environment


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_full_bipartite(enable_backend, full_graph_edgelist: pd.DataFrame):
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
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_full_bipartite_do_not_add_marker_counts(
    enable_backend,
    full_graph_edgelist: pd.DataFrame,
):
    graph = Graph.from_edgelist(
        edgelist=full_graph_edgelist,
        add_marker_counts=False,
        simplify=True,
        use_full_bipartite=True,
    )
    assert graph.vcount() == 100
    assert graph.ecount() == 2500
    assert graph.vs.attributes() == {"name", "type", "pixel_type"}


@pytest.mark.test_this
@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_full_bipartite_do_not_add_marker_counts_benchmark(
    benchmark,
    enable_backend,
    full_graph_edgelist: pd.DataFrame,
):
    graph = benchmark(
        Graph.from_edgelist,
        edgelist=full_graph_edgelist,
        add_marker_counts=False,
        simplify=True,
        use_full_bipartite=True,
    )
    assert graph.vcount() == 100
    assert graph.ecount() == 2500
    assert graph.vs.attributes() == {"name", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_full_bipartite_do_not_simplify(
    enable_backend,
    full_graph_edgelist: pd.DataFrame,
):
    edgelist_with_multiedges = full_graph_edgelist.copy()
    # Duplicate one row to create a multiedge
    one_row = edgelist_with_multiedges.iloc[0].to_frame().T
    one_row["umi"] = random_sequence(6)
    edgelist_with_multiedges = pd.concat(
        [one_row, edgelist_with_multiedges],
        axis=0,
        ignore_index=True,
    )

    # When not simplifying all edges should be kept
    graph = Graph.from_edgelist(
        edgelist=edgelist_with_multiedges,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )
    assert graph.vcount() == 100
    assert graph.ecount() == 2501
    assert graph.vs.attributes() == {"name", "type", "pixel_type"}

    # And the duplicate edge should disappear when we simplify
    graph = Graph.from_edgelist(
        edgelist=edgelist_with_multiedges,
        add_marker_counts=False,
        simplify=True,
        use_full_bipartite=True,
    )
    assert graph.vcount() == 100
    assert graph.ecount() == 2500
    assert graph.vs.attributes() == {"name", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_a_node_projected(
    enable_backend, full_graph_edgelist: pd.DataFrame
):
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
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_a_node_projected_benchmark(
    benchmark, enable_backend, full_graph_edgelist: pd.DataFrame
):
    """Build an A-node projected graph."""
    graph = benchmark(
        Graph.from_edgelist,
        edgelist=full_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=False,
    )
    assert graph.vcount() == 50
    assert graph.ecount() == ((50 * 50) / 2) - (50 / 2)
    assert "markers" in graph.vs.attributes()
    assert sorted(list(graph.vs[0]["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["igraph", "networkx"], indirect=True)
def test_build_graph_a_node_projected_without_simplifying(
    enable_backend,
    full_graph_edgelist: pd.DataFrame,
):
    def _test():
        # The A-node projection disregards any multiedges, so running it with
        # or with out simplification should yield the same result
        graph = Graph.from_edgelist(
            edgelist=full_graph_edgelist,
            add_marker_counts=True,
            simplify=False,
            use_full_bipartite=False,
        )
        assert graph.vcount() == 50
        assert graph.ecount() == ((50 * 50) / 2) - (50 / 2)
        assert "markers" in graph.vs.attributes()
        assert sorted(list(graph.vs[0]["markers"].keys())) == ["A", "B"]
        assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}

    # We want to warn when a-node projection is requested without simplification.
    if os.environ.get("ENABLE_NETWORKX_BACKEND"):
        with pytest.warns(UserWarning):
            _test()
    else:
        _test()
