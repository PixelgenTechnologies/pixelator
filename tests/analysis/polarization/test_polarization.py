"""
Tests for the polarization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

import random

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


def test_polarization_with_differentially_polarized_markers():
    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )

    random.seed(1)
    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0}
    random_vertex = graph.vs.get_vertex(random.randint(0, graph.vcount()))
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(random.randint(0, graph.vcount()))
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
                "morans_i": 0.37307837882553935,
                "morans_z": 4.095594525493692,
                "morans_p_value": 2.1054318495343033e-05,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.0037658463832960483,
                "morans_z": -2.1799059127756326,
                "morans_p_value": 0.014632218279985308,
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

    random.seed(1)
    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 1, "C": 0}
    random_vertex = graph.vs.get_vertex(random.randint(0, graph.vcount()))
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs.get_vertex(random.randint(0, graph.vcount()))
    random_vertex["markers"]["C"] = 10

    np.random.seed(1)
    scores = polarization_scores_component_graph(
        graph, component_id="PXLCMP0000000", n_permutations=10, random_seed=1
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.37307837882553935,
                "morans_z": 4.095594525493692,
                "morans_p_value": 2.1054318495343033e-05,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.0037658463832960483,
                "morans_z": -2.1799059127756326,
                "morans_p_value": 0.014632218279985308,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)
