"""
This module contains functions for doing size-based cell calling

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from pixelator.annotate.constants import (
    CELL_MIN_SIZE_SMOOTHING_FACTOR,
    DISTANCE_DEVIATION_FACTOR,
    CELL_MAX_SIZE_SMOOTHING_FACTOR,
    MINIMUM_NBR_OF_CELLS_FOR_SIZE_LIMIT,
    PRE_FILTER_LIMIT,
)

logger = logging.getLogger(__name__)


def find_component_size_limits(
    component_sizes: np.ndarray,
    direction: Literal["lower", "upper"],
) -> Optional[int]:
    """
    This function will attempt to find a cutoff for a distribution of component sizes.
    The direction of the cut-off is determined by the `direction` parameter (lower for
    min size and upper for max size).

    The underlying assumption for the lower bound is that there is one distribution
    that consists of small components, that are mostly noise and another distribution
    of components that are larger, and that make up the true cell components.

    The method employed here tries to find a drop in the size of components by
    looking at the size vs rank, on a log scale, essentially trying to find
    the point where there is a sharp drop in the size.

    This is done by finding the minimum of the second derivative of the
    spline-smoothed log(size_definition) ~ log(rank). This is similar to the method
    described by Lun et al. [1]_

    The underlying assumption and method for the upper bound is the same but we
    assign each component a point in a coordinate system based on derivate2 ~ derivate1,
    and compute the distance d from origo for each component.

    We select the top 50% largest components, and find the maximum rank R
    of a component where:

      d > DOUBLET_DISTANCE_DEVIATION_FACTOR * stdev(d)

    Essentially finding components that are outliers both in the first and the
    second derivate. We then use R, to find the component with rank R, as the size
    cutoff.

    .. [1] Lun, A., Riesenfeld, S., Andrews, T. et al. EmptyDrops: distinguishing
        cells from empty droplets in droplet-based single-cell RNA sequencing

    :param component_sizes: a numpy array of component sizes
    :return: the lower or upper bound cutoff
    :raises AssertionError: if the direction is not lower or upper
    :raises AssertionError: if component_sizes contain NaNs or zeros
    """

    def log_size_and_rank(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank component sizes by size, sort by their rank, and compute the
        log10 of both the sizes and ranks. The input dataframe must contain
        a `size` column with the component sizes. A new dataframe is
        returned with two new columns `log10_size` and `log10_rank`.
        """
        df["rank"] = df["size"].rank(ascending=False, method="first")
        df = df.sort_values(["rank"])
        df["log10_size"] = np.log10(df["size"])
        df["log10_rank"] = np.log10(df["rank"])
        return df

    def smooth(df: pd.DataFrame, x_var: str, y_var: str) -> pd.DataFrame:
        """
        Calculate a smoothing spline of df[x_var] ~ df[y_var]
        to make it possible to calculate a less unstable derivate.
        The input dataframe must contain the `x_var` and `y_var`
        columns, a new column `smooth` is added to the returned
        dataframe.
        """
        spline_func = UnivariateSpline(
            x=df[x_var],
            y=df[y_var],
            s=spline_smoothing_factor,
            check_finite=True,
        )
        df["smooth"] = spline_func(df[x_var])
        return df

    def derivatives(df: pd.DataFrame, x_var: str) -> pd.DataFrame:
        """
        Calculate the first and second derivatives of the smoothed
        `x_var` variable. The input dataframe must contain the
        `x_var` and `smooth` columns. The returned dataframe
        contains two new columns `der1` and `der2` with the
        computed derivatives.
        """
        df["der1"] = df["smooth"].diff() / df[x_var].diff()
        df["der2"] = df["der1"].diff() / df[x_var].diff()
        return df

    def find_der1_vs_der2_outliers(df: pd.DataFrame) -> pd.Series:
        """
        Find the distance from origo to each component in
        the space df[der1] ~ df[der2], then try to find
        outliers in the upper part of component ranks, by
        looking at the standard deviation of the distances.
        Finding the components that maximizes the
        rank and is an outlier. The input dataframe must
        contain the `der1` and `der2` columns and the returned
        dataframe will contain the `distance` and `rank` columns.
        The functions return a boolean Series where every outlier
        evaluates to True.
        """
        df["distance"] = np.linalg.norm(df[["der1", "der2"]].to_numpy(), axis=1)
        stddev_distance = np.std(df["distance"])
        largest_half = df["rank"] < (np.max(df["rank"]) / 2)
        extrem_distance = df["distance"] > DISTANCE_DEVIATION_FACTOR * stddev_distance
        rank = np.max(df[extrem_distance & largest_half]["rank"])
        return df["rank"] == rank - 1

    def minimum_der2(df: pd.DataFrame) -> pd.Series:
        """
        Find argmin element. The function returns
        a boolean Series where the global minimum
        of `der2` evaluates to True. The input dataframe
        must contain the `der2` column.
        """
        return df["der2"] == np.nanmin(df["der2"])

    if direction not in ["lower", "upper"]:
        raise AssertionError(f"direction must be lower or upper, got: {direction}")

    if np.isnan(component_sizes).any():
        raise AssertionError("the component sizes contains NaN values")

    if (component_sizes <= 0).any():
        raise AssertionError("the component sizes contains zeros or negative values")

    n_unique = len(np.unique(component_sizes))
    logger.debug(
        "Attempting to find size %s limit on %i component sizes where %i are unique",
        direction,
        len(component_sizes),
        n_unique,
    )

    # add a pre-filter step to make the method more robust
    pre_filter = component_sizes > PRE_FILTER_LIMIT
    logger.debug(
        (
            "Pre-filtering %s (%s%%) component sizes with size smaller than %s "
            "before trying to find a %s size limit"
        ),
        np.sum(pre_filter),
        round((np.sum(pre_filter) / len(pre_filter)) * 100, 0),
        PRE_FILTER_LIMIT,
        direction,
    )
    component_sizes = component_sizes[pre_filter]

    # check for the sizes that are unique
    if n_unique < MINIMUM_NBR_OF_CELLS_FOR_SIZE_LIMIT:
        logger.warning(
            "Too few unique component sizes (%i) to find a %s size limit",
            len(component_sizes),
            direction,
        )
        return None

    if direction == "lower":
        x_var = "log10_rank"
        y_var = "log10_size"
        determining_func = minimum_der2
        spline_smoothing_factor = CELL_MIN_SIZE_SMOOTHING_FACTOR
    else:
        x_var = "rank"
        y_var = "log10_size"
        spline_smoothing_factor = CELL_MAX_SIZE_SMOOTHING_FACTOR
        determining_func = find_der1_vs_der2_outliers

    # compute the smoothed derivatives of the log-rank sizes
    df = pd.DataFrame({"size": component_sizes})
    df = log_size_and_rank(df)
    try:
        df = smooth(df, x_var, y_var)
    except ValueError as err:
        logger.warning(
            "Spline smoothing failed with error %s, no size limit will be returned",
            str(err),
        )
        return None
    df = derivatives(df, x_var)

    # derive the cutoff
    df["cutoff"] = determining_func(df)
    potential_bounds = df[df["cutoff"]]["size"].tolist()
    bound = potential_bounds[0] if potential_bounds else None

    logger.debug("Size limit of %i found", bound)
    return bound
