"""
Tests for the colocalization modules

Copyright © 2023 Pixelgen Technologies AB.
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
            0: {
                "marker_1": "A",
                "marker_2": "B",
                "pearson": -1.0,
                "pearson_mean": -1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
                "component": "PXLCMP0000000",
            }
        },
        orient="index",
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
            0: {
                "marker_1": "A",
                "marker_2": "B",
                "pearson": -1.0,
                "pearson_mean": -1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "pearson_p_value_adjusted": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
                "jaccard_p_value_adjusted": np.nan,
                "component": "PXLCMP0000000",
            }
        },
        orient="index",
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
            0: {
                "marker_1": "A",
                "marker_2": "B",
                "pearson": -0.999904026040017,
                "pearson_mean": -0.9996137575786639,
                "pearson_stdev": 0.00010716932087924583,
                "pearson_z": -2.7085033195295076,
                "pearson_p_value": 0.00337937179980743,
                "pearson_p_value_adjusted": 0.00337937179980743,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
                "jaccard_p_value_adjusted": np.nan,
                "component": "PXLCMP0000000",
            }
        },
        orient="index",
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
