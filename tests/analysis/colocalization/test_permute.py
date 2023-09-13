"""
Tests for the colocalization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import pandas as pd
from numpy.random import default_rng
from pandas.testing import assert_series_equal

from pixelator.analysis.colocalization.permute import (
    permutations,
    permute,
)

random_number_generator = default_rng(seed=747)


def test_permute():
    df = pd.DataFrame(
        random_number_generator.integers(0, 100, size=(200, 4)),
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    result = permute(df)

    assert df.shape == result.shape
    # We want to preserve the number counts in each region
    assert_series_equal(df.sum(axis="columns"), result.sum(axis="columns"))

    # We want to make sure that the proporitions of the counts
    # are preserved (give a large enough sample)
    original_probs = df.sum(axis="index") / df.sum(axis="index").sum()
    result_probs = result.sum(axis="index") / result.sum(axis="index").sum()
    assert_series_equal(original_probs, result_probs, check_exact=False, atol=0.01)


def test_permutations():
    df = pd.DataFrame(
        random_number_generator.integers(0, 100, size=(200, 4)),
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    result = permutations(df, n=100)
    assert len(list(result)) == 100
