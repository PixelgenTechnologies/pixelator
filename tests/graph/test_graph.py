"""Test the graph module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import random

import pandas as pd
import pytest

from pixelator.graph import (
    Graph,
    connect_components,
    community_detection_crossing_edges,
    detect_edges_to_remove,
    recover_technical_multiplets,
)

from tests.graph.test_graph_utils import add_random_names_to_vertexes
from tests.graph.igraph.test_tools import full_graph


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
    return df


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
    yield input_edgelist


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
    graph = full_graph(n=100)
    add_random_names_to_vertexes(graph)
    return graph


def test_connect_components(input_edgelist, output_dir, metrics_file):
    """Test connect components function."""
    connect_components(
        input=input_edgelist,
        output=output_dir,
        output_prefix="test",
        metrics_file=metrics_file,
        multiplet_recovery=True,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.csv.gz"))
    result = pd.read_csv(result_pixel_data_file)
    assert len(result["component"].unique()) == 2


def test_connect_components_no_recovery(input_edgelist, output_dir, metrics_file):
    """Test connect components with no recovery function."""
    connect_components(
        input=input_edgelist,
        output=output_dir,
        output_prefix="test",
        metrics_file=metrics_file,
        multiplet_recovery=False,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.csv.gz"))
    result = pd.read_csv(result_pixel_data_file)
    assert len(result["component"].unique()) == 1


def test_recovery_technical_multiplets(
    edgelist_with_communities: pd.DataFrame,
):
    """Test recovery of technical multiplet components."""
    assert len(edgelist_with_communities["component"].unique()) == 1

    result, info = recover_technical_multiplets(
        edgelist=edgelist_with_communities.copy(),
    )
    assert len(result["component"].unique()) == 2
    assert info.keys() == {"PXLCMP0000000"}
    assert sorted(list(info.values())[0]) == ["RCVCMP0000000", "RCVCMP0000001"]


def test_community_detection_crossing_edges(graph_with_communities):
    """Test discovery of crossing edges from graph with communities."""
    result = community_detection_crossing_edges(
        graph=graph_with_communities,
        leiden_iterations=2,
    )
    assert result == [{"CTCGTACCTGGGACTGATACT", "TGTAAGTCAGTTGCAGGTTGG"}]


def test_community_detection_crossing_edges_no_communities(graph_without_communities):
    """Test discovery of crossing edges from graph with no communities."""
    result = community_detection_crossing_edges(
        graph=graph_without_communities,
        leiden_iterations=2,
    )
    assert result == []


def test_detect_edges_to_remove(edgelist_with_communities):
    """Test discovery of edges to remove from edgelist."""
    result = detect_edges_to_remove(edgelist_with_communities, leiden_iterations=2)
    assert result == [{"CTCGTACCTGGGACTGATACT", "TGTAAGTCAGTTGCAGGTTGG"}]
