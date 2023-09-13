"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

import time
from typing import Generator, Optional

import pandas as pd
from numpy.random import Generator as RandomNumberGenerator
from numpy.random import default_rng

from pixelator.analysis.colocalization.types import (
    RegionByCountsDataFrame,
)


def _get_random_number_generator(
    random_seed: Optional[int] = None,
) -> RandomNumberGenerator:
    if not random_seed:
        random_seed = int(time.time())
    return default_rng(seed=random_seed)


def permutations(
    df: RegionByCountsDataFrame, n=50, random_seed: Optional[int] = None
) -> Generator[RegionByCountsDataFrame, None, None]:
    """
    Generate `n` permutatinos of the data provided in `df`

    :param df: dataframe to use as basis of permutations
    :param n: number of permutations to generate, defaults to 50
    :param random_seed: set a seed to the random number generator
                        needed to make results deterministic, defaults to None
    :yield: n RegionByCountsDataFrames
    """
    random_number_generator = _get_random_number_generator(random_seed)
    for _ in range(n):
        yield permute(df, random_number_generator)


def permute(
    df: RegionByCountsDataFrame,
    random_number_generator: Optional[RandomNumberGenerator] = None,
) -> RegionByCountsDataFrame:
    """
    Permute the given dataframe in a way that preserves the number of
    counts in each region. The proportions of each marker is kept
    approximately by sampling from a multinomial distribution of the
    original counts. Note that especially for low count markers this
    might mean that some markers that have counts in the original
    dataframe might not have it counts in the permuted dataframe.

    :param df: input dataframe to permute
    :param random_number_generator: Set a random number generator
                                    to make results reproducible, defaults to None
    :return: a permuted dataframe
    """
    if not random_number_generator:
        random_number_generator = _get_random_number_generator()

    marker_counts = df.sum(axis="index")
    marker_sum = marker_counts.sum()
    marker_probs = marker_counts / marker_sum
    marker_probs_sum = marker_probs.sum()
    assert (
        -1e-10 < 1 - marker_probs_sum and marker_probs_sum - 1 < 1e-10
    ), f"{marker_probs_sum}"
    region_counts = df.sum(axis="columns")

    sampling = random_number_generator.multinomial(
        n=region_counts.to_numpy().reshape((len(region_counts), 1)), pvals=marker_probs
    ).reshape(df.shape)

    return pd.DataFrame(
        sampling,
        columns=df.columns.copy(),
        index=df.index.copy(),
    )
