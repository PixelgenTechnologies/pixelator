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

    el = full_graph_edgelist

    clr_scores = polarization_scores(
        edgelist=el,
        n_permutations=10,
        normalization="clr_neg",
        use_full_bipartite=True,
        random_seed=1,
    )

    log1p_scores = polarization_scores(
        edgelist=el,
        n_permutations=10,
        normalization="log1p",
        use_full_bipartite=True,
        random_seed=1,
    )

    # test polarization scores
    assert_frame_equal(clr_scores, log1p_scores)


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
                "morans_z": 3.6409882113983114,
                "morans_p_value": 0.00013579678673041425,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -2.558088675046134,
                "morans_p_value": 0.005262462500942811,
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
                "morans_z": 3.6409882113983114,
                "morans_p_value": 0.00013579678673041425,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -2.558088675046134,
                "morans_p_value": 0.005262462500942811,
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
        graph, component_id="PXLCMP0000000", n_permutations=10, random_seed=1
    )

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "A",
                "morans_i": 0.3009109262187527,
                "morans_z": 3.6409882113983114,
                "morans_p_value": 0.00013579678673041425,
                "component": "PXLCMP0000000",
            },
            1: {
                "marker": "C",
                "morans_i": -0.001864280387770322,
                "morans_z": -2.558088675046134,
                "morans_p_value": 0.005262462500942811,
                "component": "PXLCMP0000000",
            },
        },
        orient="index",
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)
