"""
Tests for the polarization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import random

import pandas as pd
from pandas.testing import assert_frame_equal
from pixelator.analysis.polarization import (
    polarization_scores,
    polarization_scores_component,
)

from tests.graph.igraph.test_tools import create_randomly_connected_bipartite_graph


def test_polarization(full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (sparse and clustered patterns)
    scores = polarization_scores(edgelist=full_graph_edgelist)

    expected = pd.DataFrame(
        data={
            "morans_i": [-0.05841822339465653, -0.05841818713383407],
            "morans_p_value": [0.0006289900162095327, 0.0006289959783166696],
            "morans_p_adjusted": [0.0006289959783166696, 0.0006289959783166696],
            "morans_z": [-3.41879540404497, -3.4187928246970136],
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
        n1=50, n2=100, p=0.1, random_seed=1477
    )

    graph.vs["markers"] = [{"A": 0, "B": 1, "C": 0} for _ in range(graph.vcount())]
    random_vertex = graph.vs[random.randint(0, graph.vcount())]
    random_vertex["markers"]["A"] = 5
    neighbors = random_vertex.neighbors()
    for n in neighbors:
        n["markers"]["A"] = 2

    random_vertex = graph.vs[random.randint(0, graph.vcount())]
    random_vertex["markers"]["C"] = 10

    scores = polarization_scores_component(graph, component_id="PXLCMP0000000")

    # We don't expect to get a value for B, since it has only one value in it.
    # Hence it is filtered out.
    expected = pd.DataFrame.from_dict(
        {
            "morans_i": {0: 0.41048695076280795, 1: -0.007230744646852031},
            "morans_p_value": {0: 0.0, 1: 0.9231131857481375},
            "morans_z": {0: 11.54324998113701, 1: -0.09651295450603938},
            "marker": {0: "A", 1: "C"},
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000"},
        }
    )
    # test polarization scores
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-3)
