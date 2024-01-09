"""
Tests for the colocalization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pixelator.analysis.colocalization import (
    colocalization_from_component_edgelist,
    colocalization_scores,
)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_from_component_edgelist(
    enable_backend, full_graph_edgelist: pd.DataFrame
):
    # TODO we should test more scenarios here (no overlapping and half overlapping)
    result = colocalization_from_component_edgelist(
        edgelist=full_graph_edgelist,
        component_id="PXLCMP0000000",
        transformation="raw",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
    )

    expected = pd.DataFrame.from_dict(
        {
            "marker_1": {0: "A", 1: "A", 2: "B"},
            "marker_2": {0: "A", 1: "B", 2: "B"},
            "pearson": {0: 1.0, 1: -1.0, 2: 1.0},
            "pearson_mean": {0: 1.0, 1: -1.0, 2: 1.0},
            "pearson_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "pearson_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "pearson_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_mean": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "jaccard_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "component": {
                0: "PXLCMP0000000",
                1: "PXLCMP0000000",
                2: "PXLCMP0000000",
            },
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores(enable_backend, full_graph_edgelist: pd.DataFrame):
    # TODO we should test more scenarios here (no overlapping and half overlapping)
    result = colocalization_scores(
        edgelist=full_graph_edgelist,
        use_full_bipartite=True,
        transformation="raw",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
    )

    expected = pd.DataFrame.from_dict(
        {
            "marker_1": {0: "A", 1: "A", 2: "B"},
            "marker_2": {0: "A", 1: "B", 2: "B"},
            "pearson": {0: 1.0, 1: -1.0, 2: 1.0},
            "pearson_mean": {0: 1.0, 1: -1.0, 2: 1.0},
            "pearson_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "pearson_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "pearson_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "pearson_p_value_adjusted": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_mean": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "jaccard_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value_adjusted": {0: np.nan, 1: np.nan, 2: np.nan},
            "component": {
                0: "PXLCMP0000000",
                1: "PXLCMP0000000",
                2: "PXLCMP0000000",
            },
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores_log1p(enable_backend, full_graph_edgelist: pd.DataFrame):
    result = colocalization_scores(
        edgelist=full_graph_edgelist,
        use_full_bipartite=True,
        transformation="log1p",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
        random_seed=1477,
    )

    expected = pd.DataFrame.from_dict(
        {
            "marker_1": {0: "A", 1: "A", 2: "B"},
            "marker_2": {0: "A", 1: "B", 2: "B"},
            "pearson": {0: 1.0, 1: -0.9999040260400158, 2: 1.0},
            "pearson_mean": {0: 1.0, 1: -0.9996137575786639, 2: 1.0},
            "pearson_stdev": {0: 0.0, 1: 0.00010716932087924583, 2: 0.0},
            "pearson_z": {0: np.nan, 1: -2.708503319504645, 2: np.nan},
            "pearson_p_value": {0: np.nan, 1: 0.0033793717998496305, 2: np.nan},
            "pearson_p_value_adjusted": {
                0: np.nan,
                1: 0.010138115400118594,
                2: np.nan,
            },
            "jaccard": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_mean": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "jaccard_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value_adjusted": {0: np.nan, 1: np.nan, 2: np.nan},
            "component": {
                0: "PXLCMP0000000",
                1: "PXLCMP0000000",
                2: "PXLCMP0000000",
            },
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores_should_not_fail_when_one_component_has_single_node(
    enable_backend,
    full_graph_edgelist,
):
    edgelist = full_graph_edgelist.copy()
    artificial_single_node_component = edgelist["component"].astype("str")
    artificial_single_node_component.iloc[0] = "PXLCMP0000001"
    edgelist["component"] = pd.Categorical(artificial_single_node_component)
    colocalization_scores(
        edgelist=edgelist,
        use_full_bipartite=False,
        transformation="log1p",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
        random_seed=1477,
    )


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores_should_warn_when_no_data(
    enable_backend, full_graph_edgelist, caplog
):
    with pytest.raises(ValueError):
        edgelist = full_graph_edgelist.copy()
        edgelist = edgelist.iloc[[0]]
        colocalization_scores(
            edgelist=edgelist,
            use_full_bipartite=False,
            transformation="log1p",
            neighbourhood_size=1,
            n_permutations=50,
            min_region_count=0,
            random_seed=1477,
        )
    assert (
        "No data was found to compute colocalization, probably "
        "because all components only had a single node."
    ) in caplog.text
