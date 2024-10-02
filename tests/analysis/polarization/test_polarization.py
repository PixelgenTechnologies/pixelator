"""
Tests for the polarization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pixelator.analysis.polarization import (
    PolarizationAnalysis,
    get_differential_polarity,
    polarization_scores,
    polarization_scores_component_graph,
)
from pixelator.graph import Graph
from pixelator.pixeldataset import PixelDataset
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
                "morans_i": -0.17764378122173094,
                "morans_z": -25.810281029712247,
                "morans_p_value": 3.399057861353845e-147,
                "morans_p_adjusted": 6.79811572270769e-147,
                "component": "23885f346392ff2c",
            },
            1: {
                "marker": "B",
                "morans_i": -0.17764378122173102,
                "morans_z": -25.39946321402526,
                "morans_p_value": 1.2782689040515938e-142,
                "morans_p_adjusted": 1.2782689040515938e-142,
                "component": "23885f346392ff2c",
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
                "morans_i": -0.17764378122173094,
                "morans_z": -25.810281029712247,
                "morans_p_value": 3.399057861353845e-147,
                "morans_p_adjusted": 6.79811572270769e-147,
                "component": "23885f346392ff2c",
            },
            1: {
                "marker": "B",
                "morans_i": -0.17764378122173102,
                "morans_z": -25.39946321402526,
                "morans_p_value": 1.2782689040515938e-142,
                "morans_p_adjusted": 1.2782689040515938e-142,
                "component": "23885f346392ff2c",
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
        transformation="log1p",
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
                "component": "23885f346392ff2c",
            },
            1: {
                "marker": "B",
                "morans_i": -0.17764378122173088,
                "morans_z": -25.399463214025243,
                "morans_p_value": 1.2782689040520306e-142,
                "morans_p_adjusted": 1.2782689040520306e-142,
                "component": "23885f346392ff2c",
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
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-1)


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
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-1)


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
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-1)


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
    assert_frame_equal(scores_1, expected_1, check_exact=False, atol=1e-1)
    assert_frame_equal(scores_2, expected_2, check_exact=False, atol=1e-1)

    # we also expect the scores to be the same for A and B
    assert_frame_equal(scores_1, scores_2.head(2), check_exact=False, atol=1e-1)


def test_polarization_backward_compatibility():
    # This tests that we get the same polarity score as the original
    # polarization score where markers were filtered prior to transformation
    # transformation.

    # Set seed to get same graph every time
    graph = create_randomly_connected_bipartite_graph(
        n1=50, n2=100, p=0.1, random_seed=2
    )
    rng = np.random.default_rng(1)

    for v in graph.vs:
        v["markers"] = {"A": 0, "B": 0, "C": 0}
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
    assert_frame_equal(scores, expected, check_exact=False, atol=1e-1)


class TestPolarizationAnalysis:
    analysis = PolarizationAnalysis(
        transformation_type="log1p",
        n_permutations=10,
        min_marker_count=2,
        random_seed=1,
    )

    def test_run_on_component(self, full_graph_edgelist):
        component_id = "23885f346392ff2c"
        component = Graph.from_edgelist(
            full_graph_edgelist[full_graph_edgelist["component"] == component_id],
            add_marker_counts=True,
            simplify=True,
            use_full_bipartite=True,
        )

        result = self.analysis.run_on_component(component, component_id)

        expected = pd.DataFrame.from_dict(
            {
                0: {
                    "marker": "A",
                    "morans_i": -0.17764378122173094,
                    "morans_z": -25.810281029712247,
                    "morans_p_value": 3.399057861353845e-147,
                    "component": component_id,
                },
                1: {
                    "marker": "B",
                    "morans_i": -0.17764378122173102,
                    "morans_z": -25.39946321402526,
                    "morans_p_value": 1.2782689040515938e-142,
                    "component": component_id,
                },
            },
            orient="index",
        )

        assert_frame_equal(result, expected, check_exact=False, atol=1e-3)

    def test_post_process_data(self):
        data = pd.DataFrame.from_dict(
            {
                "morans_p_value": [0.01, 0.02, 0.03],
                "component": ["PXLCMP0000000"] * 3,
            }
        )

        result = self.analysis.post_process_data(data)

        assert_series_equal(
            result["morans_p_adjusted"],
            pd.Series([0.03, 0.03, 0.03], name="morans_p_adjusted"),
        )

    def test_add_to_pixel_dataset(self):
        pxl_dataset = create_autospec(PixelDataset, instance=True)
        mock_data = pd.DataFrame(
            {
                "data": [0.01, 0.02, 0.03],
                "component": ["PXLCMP0000000"] * 3,
            }
        )

        pxl_dataset = self.analysis.add_to_pixel_dataset(mock_data, pxl_dataset)
        assert_frame_equal(pxl_dataset.polarization, mock_data)


def test_get_differential_polarity(setup_basic_pixel_dataset):
    pxl_data, *_ = setup_basic_pixel_dataset
    result = get_differential_polarity(
        polarity_data=pxl_data.polarization,
        targets="701ec72d3bda62d5",
        reference="bec92437d668cfa1",
        contrast_column="component",
        value_column="morans_i",
    )
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker": "CD19",
                "stat": 0.0,
                "p_value": 1.0,
                "median_difference": 0.2,
                "p_adj": 1.0,
                "target": "701ec72d3bda62d5",
            },
            1: {
                "marker": "CD3",
                "stat": 0.0,
                "p_value": 1.0,
                "median_difference": 0.0,
                "p_adj": 1.0,
                "target": "701ec72d3bda62d5",
            },
            2: {
                "marker": "CD45",
                "stat": 0.0,
                "p_value": 1.0,
                "median_difference": 0.0,
                "p_adj": 1.0,
                "target": "701ec72d3bda62d5",
            },
        },
        orient="index",
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
