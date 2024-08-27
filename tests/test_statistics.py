"""
Copyright © 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_almost_equal
from pandas.testing import assert_frame_equal

from pixelator.statistics import (
    clr_transformation,
    correct_pvalues,
    log1p_transformation,
    rate_diff_transformation,
    rel_normalization,
    wilcoxon_test,
)


def test_correct_p_values_basic():
    x = np.array([0.001, 0.001, 0.02, 0.01])
    result = correct_pvalues(x)
    assert_array_almost_equal(result, np.array([0.002, 0.002, 0.02, 0.013333]))


def test_correct_p_values_only_ties():
    x = np.array([0.001, 0.001])
    result = correct_pvalues(x)
    assert_array_almost_equal(result, np.array([0.001, 0.001]))


def test_correct_p_values_with_nan_values():
    x = np.array([0.001, 0.001, np.nan, 0.02, 0.01])
    result = correct_pvalues(x)
    assert_array_almost_equal(
        result, np.array([0.0025, 0.0025, np.nan, 0.025, 0.016667])
    )


def test_log1p_transformation():
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = log1p_transformation(antibody_counts)
    expected = pd.DataFrame(
        [[2.079442, 1.386294, 2.397895], [2.397895, 1.098612, 1.791759]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_frame_equal(norm_counts, expected)


def test_clr_transformation():
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = clr_transformation(antibody_counts, axis=0)
    expected = pd.DataFrame(
        [[0.557443, 0.623811, 0.802412], [0.725616, 0.455746, 0.479618]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_frame_equal(norm_counts, expected)

    norm_counts = clr_transformation(antibody_counts, axis=1)
    expected = pd.DataFrame(
        [[0.688840, 0.354093, 0.882234], [0.999055, 0.295012, 0.619424]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_frame_equal(norm_counts, expected)


def test_clr_standard_transformation_axis_0():
    """
    This tests standard definition clr (as opposed to the non-negative clr)
    """
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = clr_transformation(antibody_counts, axis=0, non_negative=False)
    expected = pd.DataFrame(
        [
            [-0.15922686555926768, 0.1438410362258904, 0.3030679017851581],
            [0.15922686555926724, -0.1438410362258904, -0.3030679017851581],
        ],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_allclose(norm_counts.sum(axis=0), 0.0, atol=1e-12)
    assert_frame_equal(norm_counts, expected)


def test_clr_standard_transformation_axis_1():
    """
    This tests standard definition clr (as opposed to the non-negative clr)
    """
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = clr_transformation(antibody_counts, axis=1, non_negative=False)
    expected = pd.DataFrame(
        [
            [0.12489781648047016, -0.568249364079475, 0.443352],
            [0.635139595900192, -0.6641433882300689, 0.029004],
        ],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_allclose(norm_counts.sum(axis=1), 0, atol=1e-12)
    assert_frame_equal(norm_counts, expected)


def test_rate_diff_transformation():
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = rate_diff_transformation(antibody_counts)
    expected = pd.DataFrame(
        [[-2.189189, 0.2972973, 1.89189189], [2.189189, -0.2972973, -1.89189189]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )
    assert_frame_equal(norm_counts, expected)


def test_rel_normalization():
    antibody_counts = pd.DataFrame(
        [[7.0, 3.0, 10.0], [10.0, 2.0, 5.0]],
        columns=["A", "B", "C"],
        index=["0000000", "0000001"],
    )

    norm_counts = rel_normalization(antibody_counts, axis=0)
    assert_frame_equal(
        norm_counts,
        pd.DataFrame(
            [[0.411765, 0.6, 0.666667], [0.588235, 0.4, 0.333333]],
            columns=["A", "B", "C"],
            index=["0000000", "0000001"],
        ),
    )

    norm_counts = rel_normalization(antibody_counts, axis=1)
    assert_frame_equal(
        norm_counts,
        pd.DataFrame(
            [[0.350000, 0.150000, 0.500000], [0.588235, 0.117647, 0.294118]],
            columns=["A", "B", "C"],
            index=["0000000", "0000001"],
        ),
    )


def test_wilcoxon_test():
    same_dist_df = pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50,
            "value": np.random.normal(size=100),
        }
    )
    same_dist_result = wilcoxon_test(same_dist_df, "A", "B", "group", "value")
    assert same_dist_result["p_value"] > 0.05
    diff_dist_df = pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50,
            "value": np.concatenate(
                [np.random.normal(size=50), np.random.normal(loc=1, size=50)]
            ),
        }
    )
    diff_dist_result = wilcoxon_test(diff_dist_df, "A", "B", "group", "value")
    assert diff_dist_result["p_value"] < 0.05
    assert diff_dist_result["median_difference"] > 0
