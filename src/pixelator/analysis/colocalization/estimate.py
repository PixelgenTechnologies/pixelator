"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from pixelator.analysis.colocalization.types import (
    CoLocalizationFunction,
    RegionByCountsDataFrame,
)

logger = logging.getLogger(__name__)


def estimate_observation_statistics(
    observations: pd.DataFrame,
    permutation_results: pd.DataFrame,
    funcs: Tuple[CoLocalizationFunction, ...],
):
    def estimates():
        for func in funcs:
            func_name = func.name
            # Since we are using a normal distribution we can just calculate the
            # mean and stdev directly
            permutation_stats = permutation_results.groupby(
                ["marker_1", "marker_2"]
            ).agg(
                **{
                    f"{func_name}_mean": pd.NamedAgg(f"{func_name}_perm", np.nanmean),
                    f"{func_name}_stdev": pd.NamedAgg(f"{func_name}_perm", np.nanstd),
                }
            )

            merged = pd.merge(
                observations[func_name], permutation_stats, on=["marker_1", "marker_2"]
            )
            merged[f"{func_name}_z"] = (
                merged[func_name] - merged[f"{func_name}_mean"]
            ) / merged[f"{func_name}_stdev"]
            merged[f"{func_name}_p_value"] = norm.sf(np.abs(merged[f"{func_name}_z"]))
            yield merged

    result = pd.concat(estimates(), axis=1)
    return result.reset_index()


def permutation_analysis_results(
    data: RegionByCountsDataFrame,
    funcs: Tuple[CoLocalizationFunction, ...],
    permuter: Callable[
        [RegionByCountsDataFrame, int], Iterable[RegionByCountsDataFrame]
    ],
    transformer: Optional[
        Callable[[RegionByCountsDataFrame], RegionByCountsDataFrame]
    ] = None,
    n=50,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    def constuct_permutation_data(data, n):
        for idx, permuted_df in enumerate(permuter(data, n=n, random_seed=random_seed)):
            df_for_comp = transformer(permuted_df) if transformer else permuted_df
            res = pd.concat(
                [
                    func.func(df_for_comp).rename(
                        columns={func.name: f"{func.name}_perm"}
                    )
                    for func in funcs
                ],
                axis=1,
            )
            res["permutation"] = idx
            res.set_index(["permutation"], inplace=True, append=True)
            res.index.names = ["marker_1", "marker_2", "permutation"]
            yield res

    permutations = pd.concat(constuct_permutation_data(data, n))
    return permutations
