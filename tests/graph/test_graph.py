"""Test the graph module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import random
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from graspologic.match import graph_match
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pixelator.graph import Graph

from tests.graph.networkx.test_tools import random_sequence
from tests.test_tools import enforce_edgelist_types_for_tests


def create_simple_edge_list_from_graph(
    graph: Graph, random_markers: bool = False
) -> pd.DataFrame:
    """Convert a graph to edge list (dataframe)."""
    random.seed(7319)

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


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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
    assert sorted(list(graph.vs.get_vertex(0)["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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
    assert sorted(list(graph.vs.get_vertex(0)["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
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
    assert sorted(list(graph.vs.get_vertex(0)["markers"].keys())) == ["A", "B"]
    assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_build_graph_a_node_projected_without_simplifying(
    enable_backend,
    full_graph_edgelist: pd.DataFrame,
):
    with pytest.warns(UserWarning):
        # The A-node projection disregards any multiedges, so running it with
        # or without simplification should yield the same result
        graph = Graph.from_edgelist(
            edgelist=full_graph_edgelist,
            add_marker_counts=True,
            simplify=False,
            use_full_bipartite=False,
        )
        assert graph.vcount() == 50
        assert graph.ecount() == ((50 * 50) / 2) - (50 / 2)
        assert "markers" in graph.vs.attributes()
        assert sorted(list(graph.vs.get_vertex(0)["markers"].keys())) == ["A", "B"]
        assert graph.vs.attributes() == {"name", "markers", "type", "pixel_type"}


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connected_components(enable_backend, edgelist):
    graph = Graph.from_edgelist(
        edgelist, add_marker_counts=False, simplify=False, use_full_bipartite=True
    )
    result = graph.connected_components()
    assert len(result) == 5
    vertex_cluster_sizes = {len(c) for c in result}
    assert vertex_cluster_sizes == {1996, 1995, 1998, 1996, 1995}
    assert len(result.giant().vs) == 1998
    subgraphs = list(result.subgraphs())
    graph_sizes = {len(g.vs) for g in subgraphs}
    assert len(subgraphs) == 5
    assert graph_sizes == {1996, 1995, 1998, 1996, 1995}


def test_community_leiden_raises_for_invalid_options(edgelist):
    graph = Graph.from_edgelist(
        edgelist, add_marker_counts=False, simplify=False, use_full_bipartite=True
    )

    with pytest.raises(AssertionError):
        _ = graph.community_leiden(beta=-1.0)


def test_connected_components_caches_results(edgelist):
    graph = Graph.from_edgelist(
        edgelist, add_marker_counts=False, simplify=False, use_full_bipartite=True
    )
    mock_func = MagicMock()
    graph._backend.connected_components = mock_func

    # The backend connected component should only be called once, since it caches
    graph.connected_components()
    graph.connected_components()
    mock_func.assert_called_once()


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connected_components_benchmark(benchmark, enable_backend, edgelist):
    graph = benchmark(
        Graph.from_edgelist,
        edgelist,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )
    result = graph.connected_components()
    assert len(result) == 5
    vertex_cluster_sizes = {len(c) for c in result}
    assert vertex_cluster_sizes == {1996, 1995, 1998, 1996, 1995}
    assert len(result.giant().vs) == 1998
    subgraphs = list(result.subgraphs())
    graph_sizes = {len(g.vs) for g in subgraphs}
    assert len(subgraphs) == 5
    assert graph_sizes == {1996, 1995, 1998, 1996, 1995}


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_get_adjacency_sparse(enable_backend, pentagram_graph):
    # This is a little bit involved. Since different network backends might
    # use different internal indexing schemes, they are not guaranteed to generate
    # the same order of nodes in the adjacency matrix
    #
    # What this test does is to generate the sparse adjacency matrix
    # and then try to find a rotation (i.e. an ordering of the nodes)
    # for that and the expected adjacency matrix under which they are identical.
    #
    # Finally it tests for the equality of these rotated matrices.

    expected = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ]
    )

    result = pentagram_graph.get_adjacency_sparse()
    results_dense = np.array(result.todense())
    expected_idx_permutations, result_idx_permutations, *_ = graph_match(
        expected, results_dense, rng=1
    )

    results_dense_permuted = results_dense[
        np.ix_(result_idx_permutations, result_idx_permutations)
    ]
    expected_permuted = expected[
        np.ix_(expected_idx_permutations, expected_idx_permutations)
    ]

    assert_array_equal(expected_permuted, results_dense_permuted)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_layout_coordinates_2d_networkx(enable_backend, pentagram_graph):
    result = pentagram_graph.layout_coordinates(
        layout_algorithm="fruchterman_reingold",
        get_node_marker_matrix=True,
        cache=False,
        only_keep_a_pixels=False,
        random_seed=1234,
    )
    assert_frame_equal(
        result.sort_index(),
        pd.DataFrame.from_dict(
            data={
                "0": {
                    "x": -0.19130137780050335,
                    "y": -0.995853333686976,
                    "A": 1,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                    "E": 0,
                },
                "2": {
                    "x": 0.8830198712248385,
                    "y": -0.4852152436940987,
                    "A": 0,
                    "B": 0,
                    "C": 1,
                    "D": 0,
                    "E": 0,
                },
                "3": {
                    "x": -0.9999999999999999,
                    "y": -0.12342422456086793,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 1,
                    "E": 0,
                },
                "1": {
                    "x": -0.42674182538070177,
                    "y": 0.9138447918386549,
                    "A": 0,
                    "B": 1,
                    "C": 0,
                    "D": 0,
                    "E": 0,
                },
                "4": {
                    "x": 0.7350233319563663,
                    "y": 0.6906480101032882,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                    "E": 1,
                },
            },
            orient="index",
        ).sort_index(),
    )


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_layout_coordinates_for_all_algorithms(enable_backend, pentagram_graph):
    # Just making sure all existing algorithms get exercised

    algorithms = [
        "fruchterman_reingold",
        "fruchterman_reingold_3d",
        "kamada_kawai",
        "kamada_kawai_3d",
    ]
    for algorithm in algorithms:
        _ = pentagram_graph.layout_coordinates(
            layout_algorithm=algorithm,
            get_node_marker_matrix=False,
            cache=False,
            only_keep_a_pixels=False,
            random_seed=1234,
        )


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_layout_coordinates_3d_networkx(enable_backend, pentagram_graph):
    result = pentagram_graph.layout_coordinates(
        layout_algorithm="fruchterman_reingold_3d",
        get_node_marker_matrix=True,
        cache=False,
        only_keep_a_pixels=False,
        random_seed=1234,
    )
    assert_frame_equal(
        result.sort_index(),
        pd.DataFrame.from_dict(
            {
                "0": {
                    "x": -0.9648627593518555,
                    "y": 0.6407664038966624,
                    "z": -0.033615468664985694,
                    "x_norm": -0.8326847714045886,
                    "y_norm": 0.5529868588884579,
                    "z_norm": -0.029010435494229957,
                    "A": 1,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                    "E": 0,
                },
                "2": {
                    "x": 0.06231005082947333,
                    "y": 0.6879998987122531,
                    "z": -0.9306229217434148,
                    "x_norm": 0.05376181897086785,
                    "y_norm": 0.5936141202608003,
                    "z_norm": -0.8029520178989151,
                    "A": 0,
                    "B": 0,
                    "C": 1,
                    "D": 0,
                    "E": 0,
                },
                "3": {
                    "x": -0.6574427161103169,
                    "y": -0.29032243099867855,
                    "z": 0.9067564430700126,
                    "x_norm": -0.5682143336015969,
                    "y_norm": -0.25091975713946313,
                    "z_norm": 0.7836911040497817,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 1,
                    "E": 0,
                },
                "1": {
                    "x": 0.5599954246326997,
                    "y": -0.8243431466963642,
                    "z": 0.5981413573809647,
                    "x_norm": 0.4818050386972553,
                    "y_norm": -0.7092427263211374,
                    "z_norm": 0.5146247757798849,
                    "A": 0,
                    "B": 1,
                    "C": 0,
                    "D": 0,
                    "E": 0,
                },
                "4": {
                    "x": 0.9999999999999999,
                    "y": -0.21410072491387208,
                    "z": -0.5406594100425767,
                    "x_norm": 0.8644648158917496,
                    "y_norm": -0.18508254374496058,
                    "z_norm": -0.46738103736259806,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                    "E": 1,
                },
            },
            orient="index",
        ).sort_index(),
    )


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_layout_coordinates_3d_benchmark(enable_backend, benchmark, pentagram_graph):
    benchmark(
        pentagram_graph.layout_coordinates,
        layout_algorithm="fruchterman_reingold_3d",
        get_node_marker_matrix=True,
        cache=False,
        only_keep_a_pixels=False,
        random_seed=1234,
    )


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_layout_coordinates_3d_networkx_only_a_pixels(enable_backend, pentagram_graph):
    result = pentagram_graph.layout_coordinates(
        layout_algorithm="fruchterman_reingold_3d",
        get_node_marker_matrix=True,
        cache=False,
        only_keep_a_pixels=True,
        random_seed=1234,
    )
    # there are 3 nodes with type A in the pentagram graph
    assert len(result) == 3


def test_layout_coordinates_caches(pentagram_graph):
    mock_layout_method = MagicMock()
    pentagram_graph._backend.layout_coordinates = mock_layout_method

    _ = pentagram_graph.layout_coordinates(
        layout_algorithm="fruchterman_reingold",
        get_node_marker_matrix=True,
        cache=True,
        only_keep_a_pixels=False,
    )

    _ = pentagram_graph.layout_coordinates(
        layout_algorithm="fruchterman_reingold",
        get_node_marker_matrix=True,
        cache=True,
        only_keep_a_pixels=False,
    )

    # If caching works as intended the backend should only be
    # hit once.
    mock_layout_method.assert_called_once()
