"""Smoke tests for the synthetic analysis pxl fixture.

Copyright © 2026 Pixelgen Technologies AB.
"""


def test_synthetic_analysis_pxl_dataset_has_expected_structure(
    synthetic_analysis_pxl_dataset,
):
    """The synthetic dataset has four components and a populated edgelist.

    Args:
        synthetic_analysis_pxl_dataset: Synthetic analysis pxl dataset.
    """
    adata = synthetic_analysis_pxl_dataset.adata()
    edgelist = synthetic_analysis_pxl_dataset.edgelist().to_polars()

    assert adata.n_obs == 4
    assert adata.n_vars == 40
    assert set(edgelist["component"].unique()) == set(adata.obs.index)
    assert (adata.X > 0).sum(axis=1).min() > 20
    assert edgelist.height == 4 * 3000
