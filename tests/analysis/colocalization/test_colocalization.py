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
    get_differential_colocalization,
)


def test_get_differential_colocalization(setup_basic_pixel_dataset):
    pxl_data, *_ = setup_basic_pixel_dataset
    result = get_differential_colocalization(
        colocalization_data_frame=pxl_data.colocalization,
        target="PXLCMP0000002",
        reference="PXLCMP0000003",
        contrast_column="component",
        use_z_score=False,
    )
    expected = pd.DataFrame.from_dict(
        {0: {"marker_1": "CD19", "marker_2": "CD45", "pearson": 0.1}},
        orient="index",
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


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
                "pearson_stdev": 0.0001071693208792342,
                "pearson_z": -2.7085033195298016,
                "pearson_p_value": 0.0033793717998044336,
                "pearson_p_value_adjusted": 0.0033793717998044336,
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
