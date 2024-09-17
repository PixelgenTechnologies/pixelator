"""
Tests for the colocalization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pixelator.analysis.colocalization import (
    ColocalizationAnalysis,
    colocalization_from_component_edgelist,
    colocalization_scores,
    get_differential_colocalization,
)
from pixelator.graph import Graph
from pixelator.pixeldataset import PixelDataset


def test_get_differential_colocalization(setup_basic_pixel_dataset):
    pxl_data, *_ = setup_basic_pixel_dataset
    result = get_differential_colocalization(
        colocalization_data_frame=pxl_data.colocalization,
        targets="701ec72d3bda62d5",
        reference="bec92437d668cfa1",
        contrast_column="component",
        value_column="pearson",
    ).iloc[:1, :]
    expected = pd.DataFrame.from_dict(
        {
            0: {
                "marker_1": "CD19",
                "marker_2": "CD45",
                "stat": 0.0,
                "p_value": 1.0,
                "median_difference": 0.1,
                "p_adj": 1.0,
                "target": "701ec72d3bda62d5",
            }
        },
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
        min_marker_count=0,
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
        min_marker_count=0,
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
                "component": "23885f346392ff2c",
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
        min_marker_count=0,
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
                "component": "23885f346392ff2c",
            }
        },
        orient="index",
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores_ratediff(
    enable_backend, full_graph_edgelist: pd.DataFrame
):
    result = colocalization_scores(
        edgelist=full_graph_edgelist,
        use_full_bipartite=True,
        transformation="rate-diff",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
        min_marker_count=0,
        random_seed=1477,
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
                "jaccard": 0.0,
                "jaccard_mean": 0.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
                "jaccard_p_value_adjusted": np.nan,
                "component": "23885f346392ff2c",
            }
        },
        orient="index",
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_colocalization_scores_low_marker_removed(
    enable_backend, full_random_graph_edgelist: pd.DataFrame
):
    component_edges = full_random_graph_edgelist.loc[
        full_random_graph_edgelist["component"] == "05639f32aabd2c23", :
    ]
    marker_counts = component_edges.groupby("marker").count()["count"]
    second_highest_marker_count = marker_counts.sort_values()[-2]
    result2 = colocalization_scores(
        edgelist=component_edges,
        use_full_bipartite=True,
        transformation="rate-diff",
        neighbourhood_size=0,
        n_permutations=50,
        min_region_count=0,
        min_marker_count=2 * second_highest_marker_count - 1,
        random_seed=1477,
    )

    assert result2.shape[0] == 1

    third_highest_marker_count = marker_counts.sort_values()[-3]
    result3 = colocalization_scores(
        edgelist=component_edges,
        use_full_bipartite=True,
        transformation="rate-diff",
        neighbourhood_size=0,
        n_permutations=50,
        min_region_count=0,
        min_marker_count=2 * third_highest_marker_count - 1,
        random_seed=1477,
    )

    assert result3.shape[0] == 3


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
        min_marker_count=0,
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
            min_marker_count=0,
            random_seed=1477,
        )
    assert (
        "No data was found to compute colocalization, probably "
        "because all components only had a single node."
    ) in caplog.text


class TestColocalizationAnalysis:
    analysis = ColocalizationAnalysis(
        transformation_type="raw",
        n_permutations=50,
        min_region_count=0,
        min_marker_count=0,
        neighbourhood_size=1,
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
                    "component": component_id,
                }
            },
            orient="index",
        )

        assert_frame_equal(result, expected)

    def test_post_process_data(self):
        data = pd.DataFrame.from_dict(
            {
                "pearson_p": [0.01, 0.02, 0.03],
                "jaccard_p": [0.01, 0.02, 0.03],
                "component": ["PXLCMP0000000"] * 3,
            }
        )

        result = self.analysis.post_process_data(data)

        assert_series_equal(
            result["pearson_p_adjusted"],
            pd.Series([0.03, 0.03, 0.03], name="pearson_p_adjusted"),
        )
        assert_series_equal(
            result["jaccard_p_adjusted"],
            pd.Series([0.03, 0.03, 0.03], name="jaccard_p_adjusted"),
        )

    def test_add_to_pixel_dataset(self):
        pxl_dataset = create_autospec(PixelDataset, instance=True)
        mock_data = pd.DataFrame(
            {
                "pearson_p": [0.01, 0.02, 0.03],
                "jaccard_p": [0.01, 0.02, 0.03],
                "component": ["PXLCMP0000000"] * 3,
            }
        )

        pxl_dataset = self.analysis.add_to_pixel_dataset(mock_data, pxl_dataset)
        assert_frame_equal(pxl_dataset.colocalization, mock_data)
