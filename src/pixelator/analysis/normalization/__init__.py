"""Functions for the normalization operations in pixelator.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _regress_out_confounder(pheno, exprs, rcond=1e-8):
    """Linear regression to remove confounding factors from abundance data."""
    design_matrix = np.column_stack((np.ones((len(pheno), 1)), pheno))
    coefficients, res, rank, s = np.linalg.lstsq(design_matrix, exprs, rcond=rcond)
    beta = coefficients[1:]  # remove intercept term
    return exprs - design_matrix[:, 1:].dot(beta)


def _get_background_abundance(dataframe: pd.DataFrame, axis=0):
    """Fit a double gaussian distribution to the abundance data and return the mean of the first gaussian as an estimation of the background level."""
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
    """empty-droplet-free method as implemented in Mulè et. al. dsb package.

    The normalization steps are: 1- log1p transformation, 2- remove background
    abundance per marker, 3- regularize abundance per component.

    :param raw_abundance: the raw abundance count data.
    :param isotype_controls: list of isotype controls.
    :return: normalized abundance data.
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
        raise ValueError(f"At least one isotype control must be provided.")

    normalized_abundance = _regress_out_confounder(pheno, log_abundance)

    return normalized_abundance
