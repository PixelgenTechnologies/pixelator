"""Module with functions for created permuted data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from time import time
from typing import Generator, Optional

import pandas as pd
import polars as pl
from numpy.random import Generator as RandomNumberGenerator
from numpy.random import default_rng


def _get_random_number_generator(
    random_seed: Optional[int] = None,
) -> RandomNumberGenerator:
    if not random_seed:
        random_seed = int(time())
    return default_rng(seed=random_seed)


def edgelist_permutations(
    edgelist_df: pl.DataFrame, n: int = 50, random_seed: Optional[int] = None
) -> Generator[pl.DataFrame, None, None]:
    """Generate `n` permutations of the markers in `edgelist_df`.

    :param edgelist_df: dataframe to use as basis of permutations
    :param n: number of permutations to generate, defaults to 50
    :param random_seed: set a seed to the random number generator
                        needed to make results deterministic, defaults to None
    :return: a generator that yields `n` permutations of the input dataframe
    :rtype: Generator[pl.DataFrame, None, None]
    """
    random_number_generator = _get_random_number_generator(random_seed)
    for _ in range(n):
        yield permute_edgelist(edgelist_df, random_number_generator)


def permute_edgelist(
    edgelist: pl.DataFrame,
    random_number_generator: Optional[RandomNumberGenerator] = None,
):
    """Permute markers in an edgelist.

    This function permutes the edgelist by shuffling the corresponding
    markers to umi1 and umi2 columns.

    :param edgelist: A DataFrame representing the edgelist
    :param n_permutations: The number of permutations to perform
    :returns: A DataFrame containing the permuted edgelist
    """
    if random_number_generator is None:
        random_number_generator = _get_random_number_generator()

    marker1_permute = (
        edgelist.select(["umi1", "marker_1"])
        .unique()
        .group_by("umi1")
        .first()
        .sort("umi1")
        .with_columns(
            pl.col("marker_1").shuffle(random_number_generator.integers(0, int(1e10)))  # type: ignore
        )
    )
    marker2_permute = (
        edgelist.select(["umi2", "marker_2"])
        .unique()
        .group_by("umi2")
        .first()
        .sort("umi2")
        .with_columns(
            pl.col("marker_2").shuffle(random_number_generator.integers(0, int(1e10)))  # type: ignore
        )
    )

    permuted_edgelist = edgelist.with_columns(
        marker_1=pl.col("umi1").replace_strict(
            marker1_permute["umi1"], marker1_permute["marker_1"]
        ),
        marker_2=pl.col("umi2").replace_strict(
            marker2_permute["umi2"], marker2_permute["marker_2"]
        ),
    )
    return permuted_edgelist


def permute_node_markers(
    node_markers: pd.DataFrame,
    random_number_generator: Optional[RandomNumberGenerator] = None,
    node_a_rows: Optional[pd.Series] = None,
):
    """Permute markers in a node_markers DataFrame.

    This function permutes the node_markers by shuffling the corresponding
    markers to umi column. If node_a_rows is provided, markers from COa nodes
    and COb nodes will be shuffled separately.

    :param node_markers: A DataFrame representing the node_markers (nodes as rows and markers as columns)
    :param random_number_generator: A RandomNumberGenerator instance
    :param node_a_rows: A boolean Series indicating which rows are COa nodes
    :returns: A DataFrame containing the permuted node_markers
    """
    if random_number_generator is None:
        random_number_generator = _get_random_number_generator()

    if node_a_rows is None:
        node_a_rows = pd.Series(False, index=node_markers.index)

    permuted_node_markers = node_markers.copy()

    permuted_node_markers.loc[node_a_rows, :] = (
        node_markers.loc[node_a_rows, :].sample(frac=1).values
    )
    permuted_node_markers.loc[~node_a_rows, :] = (
        node_markers.loc[~node_a_rows, :].sample(frac=1).values
    )
    return permuted_node_markers
