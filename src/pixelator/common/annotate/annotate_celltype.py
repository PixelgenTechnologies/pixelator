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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def annotate_cells(
    query: AnnData,
    reference: AnnData | None = None,
    summarize_by_column: str | None = None,
    reference_groups: List[str] = ["celltype_l1", "celltype_l2"],
    method: str = "nmf",
    min_prediction_score: float = 0.3,
    skip_normalization: bool = False,
    verbose: bool = False,
) -> AnnData:
    """Annotate cell types transferring labels from a reference to a query.

    Args:
    query (AnnData): The unannotated target dataset.
    reference (AnnData): The annotated reference dataset.
    summarize_by_column (str | None): If provided, summarizes annotations to the most common label per cluster/factor in this column.
    reference_groups (List[str]): The columns in `reference.obs` containing the ground truth labels.
    method (str): Either 'scanpy' (PCA + KNN, mimics Seurat) or 'nmf' (Seeded Non-Negative Matrix Factorization).
    min_prediction_score (float): Cells with prediction probabilities below this are labeled "Unknown".
    skip_normalization (bool): If True, skips the library size normalization and log1p steps.
    verbose (bool): If True, logs progress.

    Returns:
    AnnData: The query object with added annotations in `.obs`.

    """
    if reference is None:
        reference = read_adata_from_csv(
            counts_path="src/pixelator/common/resources/PBMC_cell_counts.csv",
            meta_path="src/pixelator/common/resources/PBMC_cell_annotation.csv",
        )

    if not isinstance(query, AnnData) or not isinstance(reference, AnnData):
        raise TypeError("query and reference must be AnnData objects.")
    if method not in ["scanpy", "nmf"]:
        raise ValueError("method must be either 'scanpy' or 'nmf'.")
    if not 0 <= min_prediction_score <= 1:
        raise ValueError("min_prediction_score must be between 0 and 1.")

    for ref_group in reference_groups:
        if ref_group not in reference.obs.columns:
            raise KeyError(f"Column '{ref_group}' not found in reference.obs")

    shared_features = list(set(query.var_names).intersection(reference.var_names))

    if verbose:
        logger.info(f"Found {len(shared_features)} overlapping features.")

    query_tmp = query[:, shared_features].copy()
    ref_tmp = reference[:, shared_features].copy()

    if not skip_normalization:
        if verbose:
            logger.info("Applying LogNormalize.")
        sc.pp.normalize_total(query_tmp, target_sum=1e4)
        sc.pp.log1p(query_tmp)
        sc.pp.normalize_total(ref_tmp, target_sum=1e4)
        sc.pp.log1p(ref_tmp)

    if method == "scanpy":
        query = _scanpy_annotation(
            query=query_tmp,
            reference=ref_tmp,
            groups=reference_groups,
            skip_normalization=skip_normalization,
            min_prediction_score=min_prediction_score,
            verbose=verbose,
        )
    elif method == "nmf":
        query = _seeded_nmf_annotation(
            query=query_tmp,
            reference=ref_tmp,
            groups=reference_groups,
            skip_normalization=skip_normalization,
            min_prediction_score=min_prediction_score,
            verbose=verbose,
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
    query, reference, groups, skip_normalization, min_prediction_score, verbose
):
    """Mimic Seurat's integration and label transfer (PCA + KNN)."""
    if verbose:
        logger.info("Running PCA and mapping query to reference space.")
    sc.tl.pca(reference, svd_solver="arpack")

    X_ref_pca = reference.obsm["X_pca"]

    X_query_pca = query.X.dot(reference.varm["PCs"])

    for group in groups:
        if verbose:
            logger.info(f"Transferring labels for '{group}'.")

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
    query, reference, groups, skip_normalization, min_prediction_score, verbose
):
    """Seeded NMF using enrichment matrices and Non-Negative Least Squares."""
    ref_data = (
        reference.X.toarray() if pd.api.types.is_sparse(reference.X) else reference.X
    )
    target_data = query.X.toarray() if pd.api.types.is_sparse(query.X) else query.X

    for group in groups:
        if verbose:
            logger.info(f"Running Seeded NMF for '{group}'.")

        labels = reference.obs[group].astype(str).values

        W, unique_groups = _get_w_matrix(
            ref_data=ref_data.T,
            groups=labels,
            n_cells_per_group=100,
            min_cells_per_celltype=10,
            seed=123,
            verbose=verbose,
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
    n_cells_per_group=50,
    min_cells_per_celltype=10,
    seed=123,
    verbose=True,
):
    """Compute an enrichment matrix to be used as a seed for NMF.

    Args:
    ref_data (np.ndarray): The reference data matrix (Features x Cells).
    groups (np.ndarray):Array of group labels for each cell.
    n_cells_per_group (int): Number of cells to sample per group. Default is 50.
    min_cells_per_celltype (int): Minimum number of cells required per group. Default is 10.
    seed (int): Random seed for reproducibility. Default is 123.
    verbose (bool): Whether to print progress messages. Default is True.

    Returns:
    np.ndarray: The enrichment matrix (Features x Groups).
    np.ndarray: Array of unique group labels.

    """
    np.random.seed(seed)
    df_groups = pd.DataFrame({"group": groups})

    counts = df_groups["group"].value_counts()
    valid_groups = counts[counts >= min_cells_per_celltype].index
    if verbose and len(valid_groups) < len(counts):
        logger.info(
            f"Excluded {len(counts) - len(valid_groups)} groups with < {min_cells_per_celltype} cells."
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


def read_adata_from_csv(counts_path: str, meta_path: str) -> AnnData:
    """Read single cell counts data and obs (meta-data) to an AnnData object.

    Args:
    counts_path (str): Path to the counts csv file.
    meta_path (str): Path to the obs csv file.

    Returns:
    AnnData: The AnnData object.

    """
    counts = pd.read_csv(counts_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)

    common_indices = meta.index.intersection(counts.index)
    counts = counts.loc[common_indices]
    meta = meta.loc[common_indices]

    adata = AnnData(X=counts.values, obs=meta)
    adata.var_names = counts.columns
    adata.obs_names = counts.index

    return adata
