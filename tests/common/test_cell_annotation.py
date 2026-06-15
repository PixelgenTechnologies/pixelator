"""Tests for cell type annotation.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import scanpy as sc

from pixelator.common.annotate.annotate_celltype import annotate_cells

DATA_ROOT = Path(__file__).parent / "data"


def test_annotate_cells():
    adata = sc.read_h5ad(DATA_ROOT / "pbmc_test_celltype_annotations.h5ad")
    annotated_adata = annotate_cells(adata)

    assert (
        annotated_adata.obs["reference_celltype_1"]
        == annotated_adata.obs["celltype_l1"]
    ).mean() > 0.9
