"""Test the graph module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import random

import pandas as pd

from pixelator.graph import Graph
from tests.test_tools import enforce_edgelist_types_for_tests


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
    df["umi"] = "umi"
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
