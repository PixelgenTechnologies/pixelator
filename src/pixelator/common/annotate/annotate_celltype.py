"""Functions for annotating cell types.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from pixelator.common.utils import logger

logging.basicConfig(level=logging.INFO)


def annotate_cells(
    query: AnnData | pd.DataFrame,
    reference: AnnData | None = None,
    summarize_by_column: str | None = None,
    reference_groups: List[str] = ["celltype_l1", "celltype_l2"],
    method: str = "nmf",
    min_prediction_score: float = 0.3,
    min_cells_per_celltype: int | None = None,
    n_cells_per_group: int | None = None,
    skip_normalization: bool = False,
) -> AnnData:
    """Annotate cell types transferring labels from a reference to a query.

    Args:
    query (AnnData): The unannotated target dataset.
    reference (AnnData): The annotated reference dataset.
    summarize_by_column (str | None): If provided, summarizes annotations to the most common label per cluster/factor in this column.
    reference_groups (List[str]): The columns in `reference.obs` containing the ground truth labels.
    method (str): Either 'scanpy' (PCA + KNN, mimics Seurat) or 'nmf' (Seeded Non-Negative Matrix Factorization).
    min_prediction_score (float): Cells with prediction probabilities below this are labeled "Unknown".
    min_cells_per_celltype (int | None): Minimum number of cells required per cell type for annotation.
    n_cells_per_group (int | None): Number of cells to sample per group for annotation.
    skip_normalization (bool): If True, skips the library size normalization and log1p steps.

    Returns:
    AnnData: The query object with added annotations in `.obs`.

    """
    if reference is None:
        reference = sc.read_h5ad(
            "src/pixelator/common/resources/pbmc_reference_celltype_annotations.h5ad"
        )

    if not isinstance(reference, AnnData):
        raise TypeError("reference must be an AnnData object.")

    if isinstance(query, pd.DataFrame):
        query = AnnData(
            X=query.values,
            obs=pd.DataFrame(index=query.index),
            var=pd.DataFrame(index=query.columns),
        )

    if not isinstance(query, AnnData):
        raise TypeError("query must be an AnnData object or a Pandas DataFrame.")

    if not 0 <= min_prediction_score <= 1:
        raise ValueError("min_prediction_score must be between 0 and 1.")

    for ref_group in reference_groups:
        if ref_group not in reference.obs.columns:
            raise KeyError(f"Column '{ref_group}' not found in reference.obs")

    shared_features = list(set(query.var_names).intersection(reference.var_names))

    logger.debug(f"Found {len(shared_features)} overlapping features.")

    query_tmp = query[:, shared_features].copy()
    ref_tmp = reference[:, shared_features].copy()

    if not skip_normalization:
        logger.debug("Applying LogNormalize.")
        sc.pp.normalize_total(query_tmp, target_sum=1e4)
        sc.pp.log1p(query_tmp)
        sc.pp.normalize_total(ref_tmp, target_sum=1e4)
        sc.pp.log1p(ref_tmp)

    match method:
        case "scanpy":
            if min_cells_per_celltype is not None:
                logger.warning(
                    "min_cells_per_celltype is not applicable for 'scanpy' method and will be ignored."
                )
            if n_cells_per_group is not None:
                logger.warning(
                    "n_cells_per_group is not applicable for 'scanpy' method and will be ignored."
                )
            query = _scanpy_annotation(
                query=query_tmp,
                reference=ref_tmp,
                groups=reference_groups,
                skip_normalization=skip_normalization,
                min_prediction_score=min_prediction_score,
            )
        case "nmf":
            if n_cells_per_group is None:
                n_cells_per_group = 50
            if min_cells_per_celltype is None:
                min_cells_per_celltype = 10
            query = _seeded_nmf_annotation(
                query=query_tmp,
                reference=ref_tmp,
                groups=reference_groups,
                skip_normalization=skip_normalization,
                min_prediction_score=min_prediction_score,
                n_cells_per_group=n_cells_per_group,
                min_cells_per_celltype=min_cells_per_celltype,
            )
        case _:
            raise ValueError(
                f"Unsupported method: {method}. Supported methods are 'scanpy' and 'nmf'."
            )

    if summarize_by_column is not None:
        if summarize_by_column not in query.obs.columns:
            raise KeyError(f"Column '{summarize_by_column}' not found in query.obs")

        for ref_group in reference_groups:
            summary_df = (
                query.obs.groupby([summarize_by_column, ref_group])
                .size()
                .reset_index(name="n")
            )
            summary_df = summary_df.sort_values("n", ascending=False).drop_duplicates(
                subset=[summarize_by_column]
            )

            mapping_dict = dict(
                zip(summary_df[summarize_by_column], summary_df[ref_group])
            )
            summary_col_name = f"{ref_group}_summary"
            query.obs[summary_col_name] = query.obs[summarize_by_column].map(
                mapping_dict
            )

    return query


def _scanpy_annotation(
    query, reference, groups, skip_normalization, min_prediction_score
):
    """Mimic Seurat's integration and label transfer (PCA + KNN)."""
    logger.debug("Running PCA and mapping query to reference space.")
    sc.tl.pca(reference, svd_solver="arpack")

    X_ref_pca = reference.obsm["X_pca"]

    X_query_pca = query.X.dot(reference.varm["PCs"])

    for group in groups:
        logger.debug(f"Transferring labels for '{group}'.")

        knn = KNeighborsClassifier(n_neighbors=30, weights="distance", metric="l1")
        knn.fit(X_ref_pca, reference.obs[group].astype(str))

        predicted_labels = knn.predict(X_query_pca)

        if min_prediction_score > 0:
            probs = knn.predict_proba(X_query_pca)
            max_probs = probs.max(axis=1)
            predicted_labels[max_probs < min_prediction_score] = "Unknown"

        query.obs[group] = predicted_labels

    return query


def _seeded_nmf_annotation(
    query,
    reference,
    groups,
    skip_normalization,
    min_prediction_score,
    n_cells_per_group=100,
    min_cells_per_celltype=10,
):
    """Seeded NMF using enrichment matrices and Non-Negative Least Squares."""
    ref_data = (
        reference.X.toarray() if pd.api.types.is_sparse(reference.X) else reference.X
    )
    target_data = query.X.toarray() if pd.api.types.is_sparse(query.X) else query.X

    for group in groups:
        logger.debug(f"Running Seeded NMF for '{group}'.")

        labels = reference.obs[group].astype(str).values

        W, unique_groups = _get_w_matrix(
            ref_data=ref_data.T,
            groups=labels,
            n_cells_per_group=n_cells_per_group,
            min_cells_per_celltype=min_cells_per_celltype,
            seed=123,
        )

        nnls_solver = LinearRegression(fit_intercept=False, positive=True)
        nnls_solver.fit(W, target_data.T)
        H = nnls_solver.coef_  # Shape: (Cells x Groups)

        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        proportions = H / row_sums

        max_indices = proportions.argmax(axis=1)
        predicted_labels = np.array(unique_groups)[max_indices]

        if min_prediction_score > 0:
            max_probs = proportions.max(axis=1)
            predicted_labels[max_probs < min_prediction_score] = "Unknown"

        query.obs[group] = predicted_labels

    return query


def _get_w_matrix(
    ref_data,
    groups,
    n_cells_per_group=20,
    min_cells_per_celltype=20,
    seed=123,
):
    """Compute an enrichment matrix to be used as a seed for NMF.

    Args:
    ref_data (np.ndarray): The reference data matrix (Features x Cells).
    groups (np.ndarray):Array of group labels for each cell.
    n_cells_per_group (int): Number of cells to sample per group. Default is 20.
    min_cells_per_celltype (int): Minimum number of cells required per group. Default is 20.
    seed (int): Random seed for reproducibility. Default is 123.

    Returns:
    np.ndarray: The enrichment matrix (Features x Groups).
    np.ndarray: Array of unique group labels.

    """
    np.random.seed(seed)
    df_groups = pd.DataFrame({"group": groups})

    counts = df_groups["group"].value_counts()
    valid_groups = counts[counts >= min_cells_per_celltype].index
    for excluded_group in set(counts.index) - set(valid_groups):
        logger.debug(
            f"Excluding group '{excluded_group}' with only {counts[excluded_group]} cells. Minimum required number of cells is {min_cells_per_celltype}."
        )

    sampled_indices = []
    for g in valid_groups:
        idx = df_groups[df_groups["group"] == g].index
        sampled = np.random.choice(idx, min(len(idx), n_cells_per_group), replace=False)
        sampled_indices.extend(sampled)

    sampled_data = ref_data[:, sampled_indices]
    sampled_groups = df_groups.iloc[sampled_indices]["group"].values

    unique_groups = np.unique(sampled_groups)
    W = np.zeros((sampled_data.shape[0], len(unique_groups)))

    group_means = np.column_stack(
        [sampled_data[:, sampled_groups == g].mean(axis=1) for g in unique_groups]
    )

    for i, g in enumerate(unique_groups):
        x1 = group_means[:, i]
        mask_other = np.ones(len(unique_groups), dtype=bool)
        mask_other[i] = False
        x2 = group_means[:, mask_other].mean(axis=1) + 1.0
        W[:, i] = x1 / x2

    max_vals = W.max(axis=0)
    max_vals[max_vals == 0] = 1  # Avoid division by zero
    W = W / max_vals

    return W, unique_groups
