"""
Tests for the colocalization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pixelator.analysis.colocalization import (
    colocalization_from_component_edgelist,
    colocalization_scores,
)


def test_colocalization_from_component_edgelist(full_graph_edgelist: pd.DataFrame):
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
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000", 2: "PXLCMP0000000"},
        }
    )

    assert_frame_equal(result, expected)


def test_colocalization_scores(full_graph_edgelist: pd.DataFrame):
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
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000", 2: "PXLCMP0000000"},
        }
    )

    assert_frame_equal(result, expected)


def test_colocalization_clr_transformed(full_graph_edgelist: pd.DataFrame):
    result = colocalization_from_component_edgelist(
        edgelist=full_graph_edgelist,
        component_id="PXLCMP0000000",
        transformation="clr",
        neighbourhood_size=1,
        n_permutations=50,
        min_region_count=0,
        random_seed=42,
    )

    expected = pd.DataFrame.from_dict(
        {
            "marker_1": {0: "A", 1: "A", 2: "B"},
            "marker_2": {0: "A", 1: "B", 2: "B"},
            "pearson": {0: 1.0, 1: -0.9999759868297136, 2: 1.0},
            "pearson_mean": {0: 1.0, 1: -0.9999067349924343, 2: 1.0},
            "pearson_stdev": {0: 0.0, 1: 2.9090630498626518e-05, 2: 0.0},
            "pearson_z": {0: np.nan, 1: -2.3805547041171207, 2: np.nan},
            "pearson_p_value": {0: np.nan, 1: 0.008643297077543004, 2: np.nan},
            "jaccard": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_mean": {0: 1.0, 1: 1.0, 2: 1.0},
            "jaccard_stdev": {0: 0.0, 1: 0.0, 2: 0.0},
            "jaccard_z": {0: np.nan, 1: np.nan, 2: np.nan},
            "jaccard_p_value": {0: np.nan, 1: np.nan, 2: np.nan},
            "component": {0: "PXLCMP0000000", 1: "PXLCMP0000000", 2: "PXLCMP0000000"},
        }
    )

    assert_frame_equal(result, expected, check_exact=False, rtol=0.1)
