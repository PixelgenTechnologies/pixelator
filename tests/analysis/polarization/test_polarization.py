"""
Tests for the polarization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pixelator.analysis.polarization import (
    polarization_scores,
    polarization_scores_component_df,
    polarization_scores_component_graph,
)

from tests.graph.networkx.test_tools import create_randomly_connected_bipartite_graph


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_polarization(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)

    scores = polarization_scores(
        edgelist=full_graph_edgelist,
        n_permutations=10,
        use_full_bipartite=True,
        random_seed=1,
    )

    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": -0.058418204066991254,
                "morans_z": -7.844216486178604,
                "morans_p_value": 2.1783248596911517e-15,
                "morans_p_adjusted": 4.356649719382303e-15,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "B",
                "morans_i": -0.05841820406699129,
                "morans_z": -7.747319374160002,
                "morans_p_value": 4.692634773999269e-15,
                "morans_p_adjusted": 4.692634773999269e-15,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )

    # test polarization scores
    assert_frame_equal(scores.sort_index(axis=1), expected.sort_index(axis=1))


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_permuted_polarization(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)
    scores = polarization_scores(
        edgelist=full_graph_edgelist,
        n_permutations=10,
        use_full_bipartite=True,
        random_seed=1,
    )

    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": -0.058418204066991254,
                "morans_z": -7.844216486178604,
                "morans_p_value": 2.1783248596911517e-15,
                "morans_p_adjusted": 4.356649719382303e-15,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "B",
                "morans_i": -0.05841820406699129,
                "morans_z": -7.747319374160002,
                "morans_p_value": 4.692634773999269e-15,
                "morans_p_adjusted": 4.692634773999269e-15,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_polarization_log1p(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)

    scores = polarization_scores(
        edgelist=full_graph_edgelist,
        n_permutations=10,
        normalization="log1p",
        use_full_bipartite=True,
        random_seed=1,
    )

    # test polarization scores
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": -0.1776437812217308,
                "morans_z": -25.810281029712225,
                "morans_p_value": 3.3990578613561664e-147,
                "morans_p_adjusted": 6.798115722712333e-147,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "B",
                "morans_i": -0.17764378122173088,
                "morans_z": -25.399463214025243,
                "morans_p_value": 1.2782689040520306e-142,
                "morans_p_adjusted": 1.2782689040520306e-142,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected)


def test_polarization_with_differentially_polarized_markers():
    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )

    rng = np.random.default_rng(1)

    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0}
    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["C"] = 10

    scores = polarization_scores_component_graph(
        graph, component_id="PXLCMP0000000", n_permutations=10, random_seed=1
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 8.855302575253617,
                "morans_p_value": 4.1728555387292095e-19,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": 0.33792334994637657,
                "morans_p_value": 0.3677104754311888,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)


def test_polarization_with_min_marker_count():
    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )

    rng = np.random.default_rng(1)

    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0, "D": 0}
    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["C"] = 10

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["D"] = 4

    scores = polarization_scores_component_graph(
        graph,
        component_id="PXLCMP0000000",
        n_permutations=10,
        random_seed=1,
        min_marker_count=5,
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # We don't expect to get a value for D, since the marker is filtered due to
    # low counts
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 6.6115290465881955,
                "morans_p_value": 1.9018523168814586e-11,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -0.11473061339664403,
                "morans_p_value": 0.4543293240814379,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)


def test_permuted_polarization_with_differentially_polarized_markers():
    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )

    rng = np.random.default_rng(1)

    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0}
    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["C"] = 10

    scores = polarization_scores_component_graph(
        graph,
        component_id="PXLCMP0000000",
        n_permutations=10,
        random_seed=1,
        min_marker_count=0,
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 8.855302575253617,
                "morans_p_value": 4.1728555387292095e-19,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": 0.33792334994637657,
                "morans_p_value": 0.3677104754311888,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)


def test_polarization_transformation():
    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )

    rng = np.random.default_rng(1)

    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0, "D": 0}
    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["C"] = 10

    random_vertex = graph.vs.get_vertex(
        rng.integers(low=0, high=graph.vcount(), size=1)[0]
    )
    random_vertex["markers"]["D"] = 4

    scores_1 = polarization_scores_component_graph(
        graph,
        component_id="PXLCMP0000000",
        n_permutations=10,
        random_seed=1,
        min_marker_count=5,
    )

    scores_2 = polarization_scores_component_graph(
        graph,
        component_id="PXLCMP0000000",
        n_permutations=10,
        random_seed=1,
        min_marker_count=0,
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # We don't expect to get a value for D, since the marker is filtered due to
    # low counts
    expected_1 = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 6.6115290465881955,
                "morans_p_value": 1.9018523168814586e-11,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -0.11473061339664403,
                "morans_p_value": 0.4543293240814379,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # We DO expect to get a value for D, since it is not filtered this time
    expected_2 = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 6.6115290465881955,
                "morans_p_value": 1.9018523168814586e-11,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -0.11473061339664403,
                "morans_p_value": 0.4543293240814379,
                "component": "PXLCMP0000000",
            },
            2: {
                "marker": "D",
                "morans_i": -0.0031455494542742875,
                "morans_z": 0.8264801996063112,
                "morans_p_value": 0.204265872789349,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )

    # test polarization scores
    assert_frame_equal(scores_1, expected_1, check_exact=False, atol=1e-3)
    assert_frame_equal(scores_2, expected_2, check_exact=False, atol=1e-3)

    # we also expect the scores to be the same for A and B
    assert_frame_equal(scores_1, scores_2.head(2), check_exact=False, atol=1e-3)
