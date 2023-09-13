"""
This module contains functions for finding aggregates

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from scipy.stats import iqr

from pixelator.annotate.constants import (
    TAU_HARD_THRESHOLD,
    TAU_IQR_LOWER_THRESHOLD,
    TAU_IQR_UPPER_THRESHOLD,
)

logger = logging.getLogger(__name__)


def specificity_tau(matrix: np.ndarray) -> np.ndarray:
    """
    Tau specificity score computed as described in [1]_.

    Essentially it gives us a score between 0 and 1, where a component that
    expresses a single marker would have a tau score of 1, and one where all
    markers are equally expressed would have a tau score of 0.

    .. [1] Yanai I, Benjamin H, Shmoish M, Chalifa-Caspi V, Shklar M, Ophir R,
        Bar-Even A, Horn-Saban S, Safran M, Domany E, Lancet D, Shmueli O.
        Genome-wide midrange transcription profiles reveal expression level
        relationships in human tissue specification. Bioinformatics.
        2005 Mar 1;21(5):650-9.
        doi: 10.1093/bioinformatics/bti042. Epub 2004 Sep 23. PMID: 15388519.

    :param matrix: a numpy matrix of marker counts
    :return: a vector of the computed tau values
    """
    max_count = np.max(matrix, axis=1)
    _, nbr_markers = matrix.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        x_prim = np.divide(matrix, max_count[:, None])
    tau_results = np.nansum(1 - x_prim, axis=1) / (nbr_markers - 1)

    return tau_results


def call_aggregates(adata: AnnData, inplace: bool = True) -> Optional[AnnData]:
    """
    We defined aggregates as components where either:
     - A single or a handful of markers account for almost all of the count data.
       These can likely be attributed to single antibodies forming aggregates
     - Low tau scores, meaning a an even number of counts for multiple markers.
       These likely come from multiple antibodies forming aggregates.

    For downstream analysis both of these types should be removed for most types
    of analysis.

    We find aggregates by computing a tau specificity score (see `tau_specificity`
    for details).

    We mark components as "high" if they that have a tau score above
    `annotation.constants.TAU_HARD_THRESHOLD`, or have a tau score above
    `annotation.contants.TAU_IQR_UPPER_THRESHOLD` * inter-quartile range
    from the median.

    We mark components as "low" if they that have a tau score below
    `annotation.contants.TAU_IQR_LOWER_THRESHOLD` * inter-quartile range
    from the median.

    The following data is added to the AnnData:
        - `obs["tau"]` = The tau specificity score of the component
        - `obs["tau_type"]` = "normal"/"high"/"low" for components with
            the respective levels of tau scores.
        - `uns["tau_thresholds"]["tau_upper_hard_limit"]` the upper hard
           limit used to set `tau_type` as high
        - `uns["tau_thresholds"]["tau_upper_iqr_limit]` the upper limit
           based on IQR used to set `tau_type` as high
        - `uns["tau_thresholds"]["tau_lower_iqr_limit]` the lower limit
           based on IQR used to set `tau_type` as low

    :param adata: an AnnData object to call aggregates on
    :param inplace: If `True` performs the operation inplace
    :return: the updated AnnData object or None if inplace is True.
    """
    logging.debug("Calling aggregates based on tau specificity scores")

    if not inplace:
        adata = adata.copy()

    tau_values = specificity_tau(adata.X)
    hard_limit = tau_values >= TAU_HARD_THRESHOLD
    iqr_upper_threshold = (
        np.median(tau_values) + iqr(tau_values) * TAU_IQR_UPPER_THRESHOLD
    )
    iqr_limit_upper = tau_values >= iqr_upper_threshold
    iqr_lower_threshold = (
        np.median(tau_values) - iqr(tau_values) * TAU_IQR_LOWER_THRESHOLD
    )
    iqr_limit_lower = tau_values <= iqr_lower_threshold

    adata.obs["tau_type"] = "normal"
    adata.obs.loc[(hard_limit | iqr_limit_upper), "tau_type"] = "high"
    adata.obs.loc[iqr_limit_lower, "tau_type"] = "low"
    adata.obs["tau"] = tau_values

    adata.uns["tau_thresholds"] = {}
    adata.uns["tau_thresholds"]["tau_upper_hard_limit"] = TAU_HARD_THRESHOLD
    adata.uns["tau_thresholds"]["tau_upper_iqr_limit"] = iqr_upper_threshold
    adata.uns["tau_thresholds"]["tau_lower_iqr_limit"] = iqr_lower_threshold

    total_nbr_aggregeates = np.sum(
        (adata.obs["tau_type"] == "high") | (adata.obs["tau_type"] == "low")
    )
    logger.info(
        ("Found %s aggregates. This is %1.0f%% of the components"),
        total_nbr_aggregeates,
        (total_nbr_aggregeates / len(adata.obs)) * 100,
    )

    return None if inplace else adata
