"""Module for computing colocalization statistics.

Copyright © 2023 Pixelgen Technologies AB.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from pixelator.mpx.analysis.colocalization.types import (
    CoLocalizationFunction,
    MarkerColocalizationResults,
)
from pixelator.mpx.analysis.types import RegionByCountsDataFrame


def _wide_correlation_matrix_to_long_correlation_results(
    df: pd.DataFrame, stat_name: str
) -> pd.DataFrame:
    # We want to keep the lower triangle, but also any value that has
    # a NaN value
    lower_triangle = np.tril(np.ones(df.shape)).astype(bool)
    df_lower_tri = df.mask(~lower_triangle)
    df_lower_tri.index.set_names(["marker_cols"], inplace=True)
    df_lower_tri.columns.set_names(["marker_rows"], inplace=True)
    correlation_values = df_lower_tri.stack(future_stack=True).dropna().reset_index()
    correlation_values.columns = ["marker_1", "marker_2", stat_name]
    correlation_values.set_index(["marker_1", "marker_2"], inplace=True)
    correlation_values.index.rename(["marker_1", "marker_2"])

    # We want to keep the NaN values in the output.
    # This adds them back again to the output (as they are dropped)
    # in the stack operation above. It's not pretty, but it's the only
    # way I've figured out to solve this.
    dx = (df.isna() & lower_triangle).values
    nn = pd.DataFrame(dx, index=df.index, columns=df.columns, dtype="object").stack()
    nn = nn[nn]
    nn[:] = pd.NA

    correlation_values_with_nan = pd.concat([correlation_values, nn], axis=0).drop(
        columns=[0]
    )
    correlation_values_with_nan.index.rename(["marker_1", "marker_2"], inplace=True)

    return correlation_values_with_nan


def _alphanumeric_sort_marker_columns(
    data: MarkerColocalizationResults,
) -> MarkerColocalizationResults:
    """Make sure that the markers are always sorted in the same order."""
    data.index = pd.MultiIndex.from_tuples(
        map(sorted, data.index.values), names=data.index.names
    )
    return data


def _drop_self_correlation(
    data: MarkerColocalizationResults,
) -> MarkerColocalizationResults:
    """Drop the self-correlation values from the data."""
    return data[data.index.get_level_values(0) != data.index.get_level_values(1)]


def pearson(df: RegionByCountsDataFrame) -> MarkerColocalizationResults:
    """Calculate the Pearson correlation between all vs all markers.

    Calculate the Pearson correlation between all vs all markers
    in the RegionByCountsDataFrame. Since these values are symmetrical only
    one of the combination of each marker pair is returned

    :param df: the RegionByCountsDataFrame to compute Pearson correlation on
    :rtype: MarkerColocalizationResults
    :return: MarkerColocalizationResults with Pearson correlations
    """
    pearson_matrix = df.corr(method="pearson")
    pearson_values = _alphanumeric_sort_marker_columns(
        _drop_self_correlation(
            _wide_correlation_matrix_to_long_correlation_results(
                pearson_matrix, "pearson"
            )
        )
    )
    return pearson_values


Pearson = CoLocalizationFunction(name="pearson", func=pearson)


def jaccard(df: RegionByCountsDataFrame) -> MarkerColocalizationResults:
    """Calculate the Jaccard index between all vs all markers.

    Calculate the Jaccard index between all vs all markers
    in the RegionByCountsDataFrame. Since these values are symmetrical only
    one of the combination of each marker pair is returned

    :param df: the RegionByCountsDataFrame to compute Jaccard indexes on
    :rtype: MarkerColocalizationResults
    :return: MarkerColocalizationResults with Jaccard indexes
    """
    jaccard_matrix = pd.DataFrame(
        1 - pairwise_distances((df.T > 0).to_numpy(dtype=bool), metric="jaccard"),
        index=df.columns.copy(),
        columns=df.columns.copy(),
    )
    jaccard_values = _alphanumeric_sort_marker_columns(
        _drop_self_correlation(
            _wide_correlation_matrix_to_long_correlation_results(
                jaccard_matrix, "jaccard"
            )
        )
    )
    return jaccard_values


Jaccard = CoLocalizationFunction(name="jaccard", func=jaccard)


def apply_multiple_stats(
    df: RegionByCountsDataFrame, funcs: Tuple[CoLocalizationFunction, ...]
) -> pd.DataFrame:
    """Compute multiple statistics on the same dataframe.

    :param df: data to compute statistics on
    :param funcs: a list of functions to use to compute the
                  statistics
    :return: a dataframe with all the statistics computed for the dataframe
    :rtype: pd.DataFrame
    """
    return pd.concat([func.func(df) for func in funcs], axis=1)
