"""
Tests for the colocalization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas.testing import assert_frame_equal

from pixelator.analysis.colocalization.estimate import (
    estimate_observation_statistics,
    permutation_analysis_results,
)
from pixelator.analysis.colocalization.permute import permutations
from pixelator.analysis.colocalization.statistics import (
    Jaccard,
    Pearson,
    apply_multiple_stats,
)


def test_estimate_observation_statistics():
    random_number_generator = default_rng(seed=12)
    df = pd.DataFrame(
        random_number_generator.negative_binomial(n=50, p=0.9, size=(4, 4)),
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    functions_of_interest = [Pearson, Jaccard]
    observations = apply_multiple_stats(df=df, funcs=functions_of_interest)
    permutation_results = permutation_analysis_results(
        df, funcs=[Pearson, Jaccard], permuter=permutations, n=50, random_seed=42
    )
    result = estimate_observation_statistics(
        funcs=functions_of_interest,
        observations=observations,
        permutation_results=permutation_results,
    )

    expected = pd.DataFrame.from_records(
        [
            {
                "marker_1": "marker1",
                "marker_2": "marker1",
                "pearson": 1.0,
                "pearson_mean": 1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
            },
            {
                "marker_1": "marker1",
                "marker_2": "marker2",
                "pearson": -0.3146459267857001,
                "pearson_mean": 0.21041134618906188,
                "pearson_stdev": 0.5063861813973255,
                "pearson_z": -1.0368712501709967,
                "pearson_p_value": 0.1498979321056746,
                "jaccard": 1.0,
                "jaccard_mean": 0.975,
                "jaccard_stdev": 0.0757614408414158,
                "jaccard_z": 0.3299831645537225,
                "jaccard_p_value": 0.3707063415200079,
            },
            {
                "marker_1": "marker2",
                "marker_2": "marker2",
                "pearson": 1.0,
                "pearson_mean": 1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
            },
            {
                "marker_1": "marker1",
                "marker_2": "marker3",
                "pearson": 0.3157348151855431,
                "pearson_mean": 0.32192169771907236,
                "pearson_stdev": 0.4871352637641674,
                "pearson_z": -0.012700543347493076,
                "pearson_p_value": 0.494933352486564,
                "jaccard": 1.0,
                "jaccard_mean": 0.995,
                "jaccard_stdev": 0.035355339059327376,
                "jaccard_z": 0.14142135623730964,
                "jaccard_p_value": 0.44376854199085747,
            },
            {
                "marker_1": "marker2",
                "marker_2": "marker3",
                "pearson": -0.31889640207164033,
                "pearson_mean": 0.3193760950621657,
                "pearson_stdev": 0.48759510657118094,
                "pearson_z": -1.3090215396585994,
                "pearson_p_value": 0.09526352728128246,
                "jaccard": 1.0,
                "jaccard_mean": 0.98,
                "jaccard_stdev": 0.06851187890446743,
                "jaccard_z": 0.2919201796799049,
                "jaccard_p_value": 0.3851738270115038,
            },
            {
                "marker_1": "marker3",
                "marker_2": "marker3",
                "pearson": 1.0,
                "pearson_mean": 1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
            },
            {
                "marker_1": "marker1",
                "marker_2": "marker4",
                "pearson": 0.6139601294045423,
                "pearson_mean": 0.20440656372928964,
                "pearson_stdev": 0.4610642811164378,
                "pearson_z": 0.8882786683963995,
                "pearson_p_value": 0.18719543562085778,
                "jaccard": 0.75,
                "jaccard_mean": 0.99,
                "jaccard_stdev": 0.049487165930539354,
                "jaccard_z": -4.849742261192856,
                "jaccard_p_value": 6.181100019742473e-07,
            },
            {
                "marker_1": "marker2",
                "marker_2": "marker4",
                "pearson": -0.3758230140014145,
                "pearson_mean": 0.4050255456168266,
                "pearson_stdev": 0.4091554248187237,
                "pearson_z": -1.908439952773927,
                "pearson_p_value": 0.028167188592744995,
                "jaccard": 0.75,
                "jaccard_mean": 0.975,
                "jaccard_stdev": 0.07576144084141581,
                "jaccard_z": -2.969848480983499,
                "jaccard_p_value": 0.0014897333281664957,
            },
            {
                "marker_1": "marker3",
                "marker_2": "marker4",
                "pearson": 0.9428090415820632,
                "pearson_mean": 0.4795788827845511,
                "pearson_stdev": 0.39468083140460875,
                "pearson_z": 1.173682940590114,
                "pearson_p_value": 0.12026102289968166,
                "jaccard": 0.75,
                "jaccard_mean": 0.995,
                "jaccard_stdev": 0.035355339059327376,
                "jaccard_z": -6.929646455628165,
                "jaccard_p_value": 2.10946826200288e-12,
            },
            {
                "marker_1": "marker4",
                "marker_2": "marker4",
                "pearson": 1.0,
                "pearson_mean": 1.0,
                "pearson_stdev": 0.0,
                "pearson_z": np.nan,
                "pearson_p_value": np.nan,
                "jaccard": 1.0,
                "jaccard_mean": 1.0,
                "jaccard_stdev": 0.0,
                "jaccard_z": np.nan,
                "jaccard_p_value": np.nan,
            },
        ]
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


def test_permutation_analysis_results():
    random_number_generator = default_rng(seed=12)
    for _ in range(0, 5):
        df = pd.DataFrame(
            random_number_generator.negative_binomial(n=50, p=0.9, size=(4, 4)),
            columns=["marker1", "marker2", "marker3", "marker4"],
        )
        result = permutation_analysis_results(
            df, funcs=[Pearson], permuter=permutations, n=50
        )
        assert np.round(result.pearson_perm.max(), decimals=9) <= 1
        assert np.round(result.pearson_perm.min(), decimals=9) >= -1

        assert result.index.names == ["marker_1", "marker_2", "permutation"]
        assert result.columns.to_list() == ["pearson_perm"]
        assert result.shape[0] == 10 * 50


def test_permutation_analysis_results_example():
    df = pd.DataFrame().from_records(
        [[6, 3, 7, 4], [4, 4, 6, 3], [8, 8, 6, 10], [6, 2, 3, 8]]
    )
    df.columns = ["markerA", "markerB", "markerC", "markerD"]
    perm_n = 10
    result = permutation_analysis_results(
        df, funcs=[Pearson], permuter=permutations, n=perm_n
    )

    assert result.index.names == ["marker_1", "marker_2", "permutation"]
    assert result.columns.to_list() == ["pearson_perm"]
    assert result.shape[0] == 10 * perm_n


def test_permutation_analysis_results_multiple_functions():
    random_number_generator = default_rng(seed=12)
    for _ in range(0, 5):
        df = pd.DataFrame(
            random_number_generator.negative_binomial(n=50, p=0.9, size=(4, 4)),
            columns=["marker1", "marker2", "marker3", "marker4"],
        )
        result = permutation_analysis_results(
            df, funcs=[Pearson, Jaccard], permuter=permutations, n=50
        )
        assert np.round(result.pearson_perm.max(), decimals=9) <= 1
        assert np.round(result.pearson_perm.min(), decimals=9) >= -1
        assert np.round(result.jaccard_perm.min(), decimals=9) >= 0
        assert np.round(result.jaccard_perm.max(), decimals=9) <= 1

        assert result.index.names == ["marker_1", "marker_2", "permutation"]
        assert result.columns.to_list() == ["pearson_perm", "jaccard_perm"]
        assert result.shape[0] == 10 * 50
