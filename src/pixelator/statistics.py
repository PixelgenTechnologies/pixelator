"""Functions for statistics/math.

This module contains functions related to various useful statistics/math

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


def clr_transformation(
    df: pd.DataFrame,
    axis: Literal[0, 1] = 0,
    non_negative: bool = True,
) -> pd.DataFrame:
    """Transform antibody counts data with CLR (centered log ratio).

    This function will perform CLR (centered log ratio) transformation
    on the dataframe that is passed containing antibody counts.

    A description of CLR transformation can be found at:
    https://en.wikipedia.org/wiki/Compositional_data#Center_logratio_transform
    Essentially, the counts are divided by the geometric mean and then log-
    transformed. An alternate version consists of log-transforming the counts
    first and then subtracting the geometric mean (log). This makes the
    transformed counts centered around zero (include negative values).
    Use `axis=0` to apply the transformation by column (antibody) and `axis=1`
    to apply the transformation by row (component).

    :param df: the dataframe of antibody counts.
    :param axis: on which axis to apply the transformation. axis=0 means
                 by columns (antibody), and axis=1 means by row (component).
    :param non_negative: if `True` the non-negative CLR transform will be used.
                         if `False` the zero-centered CLR transformation will be used.
    :raises AssertionError: when the input axis is not valid
    :return: a dataframe with the antibody counts transformed
    :rtype: pd.DataFrame
    """
    if axis not in [0, 1]:
        raise AssertionError("Axis is required to be 0 or 1")

    logger.debug(
        "Computing CLR transformation for antibody counts with %i nodes and %i markers",
        df.shape[0],
        df.shape[1],
    )

    geometric_mean_log = np.mean(np.log1p(df), axis=axis)
    dim = 1 if axis == 0 else 0
    if non_negative:
        # using the definition of clr from Wikipedia:
        # clr(X) = [log(x1 / g(X)), ..., log(xn / g(X))]
        # where X is a vector, and g(X) is the geometric mean
        # this is equivalent to clr(X) = log(X / g(X))
        # this definition is used below with a pseudocount of 1:
        clr_df = np.log1p(df.div(np.exp(geometric_mean_log), axis=dim))
    else:
        # since log(A/B) = log(A) - log(B) we can rewrite the term above to:
        # clr(X) = [log(x1) - log(g(X)), ..., log(xn) - log(g(X))]
        # this definition is used below with a pseudocount of 1:
        clr_df = np.log1p(df).subtract(geometric_mean_log, axis=dim)

    logger.debug("CLR transformation computed")
    return clr_df


def correct_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Correct a series of p-values using the Benjamini-Hochberg method.

    It returns the corrected p-values as `np.ndarray` in the same order
    as the original array.

    An outline of the method can be found here:
    https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure

    :param pvalues: an array of p-values to adjust
    :return: the array of adjusted p-values in the same order as the input array
    :rtype: np.ndarray
    """
    # Most descriptions of the BH method states that p-values should
    # first be ordered in ascending order an ranked, however doing so
    # requires reversing the sort-order multiple times. For this reason
    # we sort them in descending order instead.
    descending_order_idx = pvalues.argsort()[::-1]
    original_order_idx = descending_order_idx.argsort()

    p_values_in_desc_order = pvalues[descending_order_idx]
    total_nbr_of_p_values = float(len(pvalues))
    descending_rank = np.arange(len(pvalues), 0, -1)

    q = np.where(
        np.isnan(p_values_in_desc_order),
        np.nan,
        np.fmin.accumulate(
            (p_values_in_desc_order * total_nbr_of_p_values / descending_rank)
        ),
    )
    q[q > 1] = 1
    return q[original_order_idx]


def log1p_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a antibody counts using the log1p function.

    A helper function that takes as input a dataframe of antibody
    counts and transforms it using log1p function (natural logarithm
    of 1 + x) on the count of each marker or component, element-wise.

    :param df: the dataframe of antibody counts (antibodies as columns)
    :returns: a dataframe with the counts normalized
    :rtype: pd.DataFrame
    """
    logger.debug(
        (
            "Computing LOG1P normalization for antibody counts"
            " with %i nodes and %i markers"
        ),
        df.shape[0],
        df.shape[1],
    )

    log1p_df = df.transform(np.log1p)

    logger.debug("LOG1P transformation computed")
    return log1p_df


def rate_diff_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Transform antibody counts as deviation from an expected baseline distribution.

    In this function we refer to baseline distribution as fixed ratio of different
    antibody types in each node. For example, if in total 10% of antibodies are
    HLA-ABC, in a node with 120 antibodies we expect to see 12 HLA-ABC counts.
    If we actually see 8 counts in this node, the rate_diff_transformation for
    HLA-ABC in this node will be -4.

    :param df: the dataframe of raw antibody counts (antibodies as columns)
    :returns: a dataframe with the counts difference from expected values
    :rtype: pd.DataFrame
    """
    antibody_counts_per_node = df.sum(axis=1)
    antibody_rates = df.sum(axis=0)
    antibody_rates = antibody_rates / antibody_rates.sum()

    expected_counts = antibody_counts_per_node.to_frame() @ antibody_rates.to_frame().T
    return df - expected_counts


def rel_normalization(df: pd.DataFrame, axis: Literal[0, 1] = 0) -> pd.DataFrame:
    """Normalize antibody counts to the relative amount per marker or component.

    A helper function that takes as input a dataframe of antibody
    counts and normalizes it using relative counts (the count of each
    marker or component is divided by its the total sum), element-wise.
    Use `axis=0` to apply the normalization by column (antibody) and `axis=1`
    to apply the normalization by row (component).

    :param df: the dataframe of antibody counts (antibodies as columns)
    :param axis: on which axis to apply the normalization
    :raises AssertionError: when the input axis is not valid
    :returns: a dataframe with the counts normalized
    :rtype: pd.DataFrame
    """
    if axis not in [0, 1]:
        raise AssertionError("Axis is required to be 0 or 1")

    logger.debug(
        "Computing REL normalization for antibody counts with %i nodes and %i markers",
        df.shape[0],
        df.shape[1],
    )

    norm_df = df.div(df.sum(axis=axis), axis=1 if axis == 0 else 0)

    logger.debug("REL normalization computed")
    return norm_df


def wilcoxon_test(
    df: pd.DataFrame,
    reference: str,
    target: str,
    contrast_column: str,
    value_column: str,
) -> pd.Series:
    """Perform a Wilcoxon rank-sum test between two groups.

    :param df: the dataframe containing the data.
    :param reference: name of the reference group in the contrast column.
    :param target: name of the target group in the contrast column.
    :param contrast_column: name the column containing the group information.
    :param value_column: name of the column containing the values to compare.
    :return: a series containing the test statistic, p-value and median difference.
    """
    reference_df = df.loc[df[contrast_column] == reference, :]
    target_df = df.loc[df[contrast_column] == target, :]

    if reference_df.empty or target_df.empty:
        return pd.Series({"stat": 0, "p_value": 1, "median_difference": 0})

    estimate = np.median(
        target_df[value_column].to_numpy()[:, None]
        - reference_df[value_column].to_numpy()
    )

    stat, p_value = mannwhitneyu(
        x=reference_df[value_column],
        y=target_df[value_column],
        alternative="two-sided",
    )

    return pd.Series({"stat": stat, "p_value": p_value, "median_difference": estimate})
