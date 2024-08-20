"""Tests for the colocalization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pixelator.analysis.colocalization.statistics import (
    Jaccard,
    Pearson,
    apply_multiple_stats,
    jaccard,
    pearson,
)


def _assert_df_equal(results, expected):
    expected.set_index(["marker1", "marker2"], inplace=True)
    assert_frame_equal(
        results.sort_index(),
        expected.sort_index(),
        check_dtype=False,
        check_names=False,
    )


def test_pearson():
    df = pd.DataFrame(
        [[1, 2, 3, 4], [2, 3, 5, 6], [10, 5, 3, 8]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    results = pearson(df)
    # Check that the data conforms to the expected shape, i.e. the upper-
    # triagonal matrix of all vs all comparissons
    cols = ["marker1", "marker2", "pearson"]
    data = [
        ["marker1", "marker2", 0.973223],
        ["marker1", "marker3", -0.409644],
        ["marker2", "marker3", -0.188982],
        ["marker1", "marker4", 0.912245],
        ["marker2", "marker4", 0.981981],
        ["marker3", "marker4", 0.000000],
    ]
    expected = pd.DataFrame(data, columns=cols)
    _assert_df_equal(results, expected)


def test_pearson_no_variation():
    # Pearson will return NaN if a column has no variation, i.e.
    # we set marker 3, 4 to 3 and 4 for all observations here to test that
    df = pd.DataFrame(
        [[1, 2, 3, 4], [2, 3, 3, 4], [10, 5, 3, 4]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    results = pearson(df)
    # Check that the data conforms to the expected shape, i.e. the upper-
    # triagonal matrix of all vs all comparisons
    cols = ["marker1", "marker2", "pearson"]
    data = [
        ["marker1", "marker2", 0.973223],
        ["marker1", "marker3", np.nan],
        ["marker2", "marker3", np.nan],
        ["marker1", "marker4", np.nan],
        ["marker2", "marker4", np.nan],
        ["marker3", "marker4", np.nan],
    ]

    expected = pd.DataFrame(data, columns=cols)
    _assert_df_equal(results, expected)


def test_jaccard():
    df = pd.DataFrame(
        [[0, -1, 3, 4], [2, 3, 0, 6], [0, 5, 3, 8]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    results = jaccard(df)

    cols = ["marker1", "marker2", "jaccard"]
    data = [
        ["marker1", "marker2", 0.500000],
        ["marker1", "marker3", 0.000000],
        ["marker2", "marker3", 0.333333],
        ["marker1", "marker4", 0.333333],
        ["marker2", "marker4", 0.666667],
        ["marker3", "marker4", 0.666667],
    ]

    _assert_df_equal(results, pd.DataFrame(data, columns=cols))


def test_apply_multiple_stats():
    df = pd.DataFrame(
        [[0, 0, 3, 4], [2, 3, 0, 6], [0, 5, 3, 8]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )

    result = apply_multiple_stats(df, [Pearson, Jaccard])
    assert result.columns.to_list() == ["pearson", "jaccard"]
