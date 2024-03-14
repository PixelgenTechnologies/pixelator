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
    polarization_scores_component,
)

from tests.graph.networkx.test_tools import create_randomly_connected_bipartite_graph


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_polarization(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)

    scores = polarization_scores(
        edgelist=full_graph_edgelist, permutations=0, use_full_bipartite=True
    )

    expected = pd.DataFrame(
        data={
            "morans_i": [-0.058418204066991254, -0.05841820406699129],
            "morans_p_value": [0.0006289931974147407, 0.0006289931974147351],
            "morans_p_adjusted": [0.0006289931974147407, 0.0006289931974147407],
            "morans_z": [-3.418794027777849, -3.4187940277778512],
            "marker": ["A", "B"],
            "component": ["PXLCMP0000000", "PXLCMP0000000"],
        },
        index=pd.Index([0, 1]),
    )
    # test polarization scores
    assert_frame_equal(scores, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_permuted_polarization(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)
    np.random.seed(1)

    scores = polarization_scores(
        edgelist=full_graph_edgelist, permutations=999, use_full_bipartite=True
    )

    expected = pd.DataFrame(
        data={
            "morans_i": [-0.058418204066991254, -0.05841820406699129],
            "morans_p_value": [0.0006289931974147407, 0.0006289931974147351],
            "morans_p_adjusted": [0.0006289931974147407, 0.0006289931974147407],
            "morans_z": [-3.418794027777849, -3.4187940277778512],
            "morans_p_value_sim": {0: 0.02, 1: 0.01},
            "morans_z_sim": {0: -3.2672829000275176, 1: -3.9734152894187327},
            "marker": ["A", "B"],
            "component": ["PXLCMP0000000", "PXLCMP0000000"],
        },
        index=pd.Index([0, 1]),
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

    scores = polarization_scores_component(graph, component_id="PXLCMP0000000")

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            "morans_i": {0: 0.37307837882553935, 1: -0.0037658463832960496},
            "morans_p_value": {0: 1.317601591958555e-17, 1: 0.5910333410276116},
            "morans_z": {0: 8.54213868598478, 1: 0.537339187874427},
            "marker": {0: "A", 1: "C"},
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000"},
        }
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
    scores = polarization_scores_component(
        graph, component_id="PXLCMP0000000", permutations=999
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            "morans_i": {0: 0.37307837882553935, 1: -0.0037658463832960496},
            "morans_p_value": {0: 1.317601591958555e-17, 1: 0.5910333410276116},
            "morans_z": {0: 8.54213868598478, 1: 0.537339187874427},
            "morans_p_value_sim": {0: 0.001, 1: 0.411},
            "morans_z_sim": {0: 8.91274668296225, 1: 0.5422508893213437},
            "marker": {0: "A", 1: "C"},
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000"},
        }
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)
