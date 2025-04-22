"""Functions for statistics/math.

This module contains functions related to various useful statistics/math

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def clr_transformation(
    df: pd.DataFrame,
    axis: Literal[0, 1] = 0,
    non_negative: bool = True,
) -> pd.DataFrame:
    """Transform antibody counts data with CLR (centered log ratio).

    This function performs a CLR (centered log ratio) transformation on the
    provided dataframe containing antibody counts. The CLR transformation
    divides the counts by the geometric mean and then applies a log
    transformation. Alternatively, it can log-transform the counts first and
    then subtract the geometric mean (log), centering the transformed counts
    around zero (which may include negative values).

    Args:
        df (pd.DataFrame): The dataframe of antibody counts.
        axis (Literal[0, 1], optional): The axis on which to apply the
            transformation. `axis=0` applies the transformation by columns
            (antibody), and `axis=1` applies it by rows (component). Defaults
            to 0.
        non_negative (bool, optional): If `True`, the non-negative CLR
            transformation is used. If `False`, the zero-centered CLR
            transformation is used. Defaults to True.

    Raises:
        AssertionError: If the input axis is not 0 or 1.

    Returns:
        pd.DataFrame: A dataframe with the antibody counts transformed.

    References:
        https://en.wikipedia.org/wiki/Compositional_data#Center_logratio_transform

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

    An outline of the method can be found here:
    https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure

    Args:
        pvalues (np.ndarray): An array of p-values to adjust.

    Returns:
        np.ndarray: The array of adjusted p-values in the same order as the input array.

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
    """Transform antibody counts using the log1p function.

    This function applies the natural logarithm of (1 + x) to the count of each
    marker or component, element-wise.

    Args:
        df (pd.DataFrame): The dataframe of antibody counts (antibodies as columns).

    Returns:
        pd.DataFrame: A dataframe with the counts normalized.

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

    The baseline distribution refers to a fixed ratio of different antibody types
    in each node. For example, if 10% of antibodies are HLA-ABC, in a node with
    120 antibodies, the expected count is 12. If the actual count is 8, the
    transformation for HLA-ABC in this node will be -4.

    Args:
        df (pd.DataFrame): The dataframe of raw antibody counts (antibodies as columns).

    Returns:
        pd.DataFrame: A dataframe with the counts difference from expected values.

    """
    antibody_counts_per_node = df.sum(axis=1)
    antibody_rates = df.sum(axis=0)
    antibody_rates = antibody_rates / antibody_rates.sum()

    expected_counts = antibody_counts_per_node.to_frame() @ antibody_rates.to_frame().T
    return df - expected_counts


def rel_normalization(df: pd.DataFrame, axis: Literal[0, 1] = 0) -> pd.DataFrame:
    """Normalize antibody counts to the relative amount per marker or component.

    This function normalizes antibody counts using relative counts, where the
    count of each marker or component is divided by its total sum. Use `axis=0`
    to apply the normalization by column (antibody) and `axis=1` to apply it by
    row (component).

    Args:
        df (pd.DataFrame): The dataframe of antibody counts (antibodies as columns).
        axis (Literal[0, 1]): The axis on which to apply the normalization.
            `axis=0` applies normalization by columns, and `axis=1` applies it by rows.

    Raises:
        AssertionError: If the input axis is not 0 or 1.

    Returns:
        pd.DataFrame: A dataframe with the counts normalized.

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

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        reference (str): Name of the reference group in the contrast column.
        target (str): Name of the target group in the contrast column.
        contrast_column (str): Name of the column containing the group information.
        value_column (str): Name of the column containing the values to compare.

    Returns:
        pd.Series: A series containing the test statistic, p-value, and median difference.

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


def _regress_out_confounder(pheno, exprs, rcond=1e-8):
    """Linear regression to remove confounding factors from abundance data."""
    design_matrix = np.column_stack((np.ones((len(pheno), 1)), pheno))
    coefficients, res, rank, s = np.linalg.lstsq(design_matrix, exprs, rcond=rcond)
    beta = coefficients[1:]  # remove intercept term
    return exprs - design_matrix[:, 1:].dot(beta)


def _get_background_abundance(dataframe: pd.DataFrame, axis=0):
    """Estimate the background abundance of a marker or component.

    Fit a double gaussian distribution to the abundance data and return the
    mean of the first gaussian as an estimation of the background level.
    """
    background = pd.Series(index=dataframe.index if axis == 0 else dataframe.columns)
    scores = pd.Series(index=dataframe.index if axis == 0 else dataframe.columns)
    gmm = GaussianMixture(n_components=2, max_iter=1000, random_state=0)
    if axis not in {0, 1}:
        raise ValueError(f"Axis was {axis}. Must be 0 or 1")
    ax_iter = dataframe.index if axis == 0 else dataframe.columns
    for i in ax_iter:
        current_axis = dataframe.loc[i, :] if axis == 0 else dataframe.loc[:, i]
        gmm = gmm.fit(current_axis.to_frame())
        background[i] = np.min(gmm.means_)
        scores[i] = np.abs(gmm.means_[1] - gmm.means_[0]) / np.sum(gmm.covariances_)
    return background, scores


def dsb_normalize(
    raw_abundance: pd.DataFrame, isotype_controls: Union[List, None] = None
):
    """Normalize abundance data using the empty-droplet-free method.

    This method is implemented as described in Mulè et al.'s dsb package.
    The normalization steps are:
    1. Log1p transformation.
    2. Remove background abundance per marker.
    3. Regularize abundance per component.

    Args:
        raw_abundance (pd.DataFrame): The raw abundance count data.
        isotype_controls (Union[List, None]): List of isotype controls.

    Raises:
        ValueError: If no isotype controls are provided.

    Returns:
        pd.DataFrame: Normalized abundance data.

    References:
        Integrating population and single-cell variations in vaccine responses
        identifies a naturally adjuvanted human immune setpoint,
        Matthew P. Mulè et al., Immunity, 2024,
        https://doi.org/10.1016/j.immuni.2024.04.009

    """
    log_abundance = np.log1p(raw_abundance)
    marker_background, _ = _get_background_abundance(log_abundance, axis=1)
    log_abundance = log_abundance - marker_background
    component_background, _ = _get_background_abundance(log_abundance, axis=0)

    if isotype_controls is not None:
        control_signals = log_abundance.loc[:, isotype_controls]
        control_signals["component_background"] = component_background
        control_signals = StandardScaler().fit_transform(control_signals)
        pheno = PCA(n_components=1).fit_transform(control_signals)
    else:
        raise ValueError("At least one isotype control must be provided.")

    normalized_abundance = _regress_out_confounder(pheno, log_abundance)

    return normalized_abundance
