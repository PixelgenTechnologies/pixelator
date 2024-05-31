"""Functions for the colocalization analysis in pixelator.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _limma(pheno, exprs, rcond=1e-8):
    """Linear regression to remove confounding factors from expression data."""
    design_matrix = np.column_stack((np.ones((len(pheno), 1)), pheno))
    coefficients, res, rank, s = np.linalg.lstsq(design_matrix, exprs, rcond=rcond)
    beta = coefficients[1:]  # remove intercept term
    return exprs - design_matrix[:, 1:].dot(beta)


def _get_baseline_expression(dataframe: pd.DataFrame, axis=0):
    """Fit a double gaussian distribution to the expression data and return mean of the first gaussian as baseline."""
    baseline = pd.Series(index=dataframe.index if axis == 0 else dataframe.columns)
    scores = pd.Series(index=dataframe.index if axis == 0 else dataframe.columns)
    if axis == 0:
        for i in dataframe.index:
            gmm = GaussianMixture(n_components=2, max_iter=1000, random_state=0)
            marker_data = dataframe.loc[i, :].to_frame()
            gmm = gmm.fit(marker_data)
            baseline[i] = np.min(gmm.means_)
            scores[i] = np.abs(gmm.means_[1] - gmm.means_[0]) / np.sum(gmm.covariances_)
    elif axis == 1:
        for i in dataframe.columns:
            gmm = GaussianMixture(n_components=2, max_iter=1000, random_state=0)
            marker_data = dataframe.loc[:, i].to_frame()
            gmm = gmm.fit(marker_data)
            baseline[i] = np.min(gmm.means_)
            scores[i] = np.abs(gmm.means_[1] - gmm.means_[0]) / np.sum(gmm.covariances_)
    return baseline, scores


def dsb_normalize(
    raw_expression: pd.DataFrame, isotype_controls: Union[List, None] = None
):
    """empty-droplet-free method as implemented in the dsb package.

    The normalization steps are: 1- log1p transformation, 2- remove baseline
    expression per marker, 3- regularize expression per component.

    :param raw_expression: the raw expression data.
    :param isotype_controls: list of isotype controls.
    :return: normalized expression data.
    """
    log_expression = np.log1p(raw_expression)
    marker_baseline, _ = _get_baseline_expression(log_expression, axis=1)
    log_expression = log_expression - marker_baseline
    component_baseline, _ = _get_baseline_expression(log_expression, axis=0)

    if isotype_controls is not None:
        control_signals = log_expression.loc[:, isotype_controls]
        control_signals["component_baseline"] = component_baseline
        pheno = PCA(n_components=1).fit_transform(control_signals)
    else:
        pheno = component_baseline.values.reshape(-1, 1)

    normalized_expression = _limma(pheno, log_expression)

    return normalized_expression
