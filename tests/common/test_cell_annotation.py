"""Tests for cell type annotation.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from pixelator.common.annotate.annotate_celltype import (
    _load_default_reference,
    annotate_cells,
)

DATA_ROOT = Path(__file__).parent / "data"


def test_annotate_cells():
    adata = sc.read_h5ad(DATA_ROOT / "pbmc_test_celltype_annotations.h5ad")
    annotated_adata = annotate_cells(adata)

    assert (
        annotated_adata.obs["reference_celltype_1"]
        == annotated_adata.obs["celltype_l1"]
    ).mean() > 0.9


def test_annotate_cells_scanpy_centers_query_by_reference_mean():
    ref = _load_default_reference()
    annotated = annotate_cells(
        ref.copy(),
        reference=ref.copy(),
        method="scanpy",
        reference_groups=["celltype_l1"],
        skip_normalization=True,
    )
    accuracy = (
        annotated.obs["celltype_l1"].astype(str).to_numpy()
        == ref.obs["celltype_l1"].astype(str).to_numpy()
    ).mean()
    assert accuracy > 0.99


def test_annotate_cells_scanpy_sparse_matches_dense():
    rng = np.random.default_rng(42)
    n_per_group = 40
    n_vars = 20
    offset = 100.0
    group_a = rng.normal(loc=offset + 0.0, scale=0.5, size=(n_per_group, n_vars))
    group_b = rng.normal(loc=offset + 3.0, scale=0.5, size=(n_per_group, n_vars))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    obs = pd.DataFrame(
        {"celltype": ["A"] * n_per_group + ["B"] * n_per_group},
        index=[f"c{i}" for i in range(2 * n_per_group)],
    )
    var = pd.DataFrame(index=[f"m{j}" for j in range(n_vars)])

    ref_dense = AnnData(X=X.copy(), obs=obs.copy(), var=var.copy())
    query_dense = AnnData(X=X.copy(), obs=pd.DataFrame(index=obs.index), var=var.copy())
    res_dense = annotate_cells(
        query_dense,
        reference=ref_dense,
        method="scanpy",
        reference_groups=["celltype"],
        skip_normalization=True,
    )

    ref_sparse = AnnData(X=sparse.csr_matrix(X.copy()), obs=obs.copy(), var=var.copy())
    query_sparse = AnnData(
        X=sparse.csr_matrix(X.copy()),
        obs=pd.DataFrame(index=obs.index),
        var=var.copy(),
    )
    res_sparse = annotate_cells(
        query_sparse,
        reference=ref_sparse,
        method="scanpy",
        reference_groups=["celltype"],
        skip_normalization=True,
    )

    expected = obs["celltype"].to_numpy()
    assert (res_dense.obs["celltype"].to_numpy() == expected).all()
    assert (res_sparse.obs["celltype"].to_numpy() == expected).all()
