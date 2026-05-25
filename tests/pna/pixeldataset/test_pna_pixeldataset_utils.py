"""Tests for pixelator.pna.pixeldataset.utils module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from pixelator.pna.pixeldataset.utils import update_metrics_anndata


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object."""
    X = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 0]])
    # observations (rows) are components/cells
    obs = pd.DataFrame(index=["obs1", "obs2", "obs3"])
    # features (cols) are antibodies
    var = pd.DataFrame(index=["ab1", "ab2", "ab3"])

    return anndata.AnnData(X=X, obs=obs, var=var)


def test_update_metrics_anndata_inplace(mock_adata):
    """Test updating metrics inplace.

    Args:
        mock_adata: Mock adata.

    """
    result = update_metrics_anndata(mock_adata, inplace=True)
    assert result is None

    # Check if metrics are updated in the original object
    assert "antibody_count" in mock_adata.var.columns
    assert "components" in mock_adata.var.columns
    assert "antibody_pct" in mock_adata.var.columns
    assert "n_antibodies" in mock_adata.obs.columns

    # Verify var metrics
    # antibody_count sum over columns:
    # ab1: 1+0+5 = 6
    # ab2: 2+3+0 = 5
    # ab3: 0+4+0 = 4
    pd.testing.assert_series_equal(
        mock_adata.var["antibody_count"],
        pd.Series([6, 5, 4], index=["ab1", "ab2", "ab3"], name="antibody_count").astype(
            int
        ),
    )

    # components (count non-zero entries per antibody):
    # ab1: obs1(1), obs3(5) -> 2
    # ab2: obs1(2), obs2(3) -> 2
    # ab3: obs2(4) -> 1
    pd.testing.assert_series_equal(
        mock_adata.var["components"],
        pd.Series([2, 2, 1], index=["ab1", "ab2", "ab3"], name="components").astype(
            int
        ),
    )

    # antibody_pct:
    # total counts = 6+5+4 = 15
    # ab1: 6/15 = 0.4
    # ab2: 5/15 = 0.333...
    # ab3: 4/15 = 0.266...
    expected_pct = pd.Series(
        [6 / 15, 5 / 15, 4 / 15], index=["ab1", "ab2", "ab3"], name="antibody_pct"
    ).astype(np.float32)

    pd.testing.assert_series_equal(mock_adata.var["antibody_pct"], expected_pct)

    # Verify obs metrics
    # n_antibodies (sum X > 0 axis=1)
    # obs1: 1, 2 -> 2
    # obs2: 3, 4 -> 2
    # obs3: 5 -> 1
    expected_n_antibodies = pd.Series(
        [2, 2, 1], index=["obs1", "obs2", "obs3"], name="n_antibodies"
    ).astype(np.uint32)

    pd.testing.assert_series_equal(
        mock_adata.obs["n_antibodies"], expected_n_antibodies
    )


def test_update_metrics_anndata_not_inplace(mock_adata):
    """Test updating metrics not inplace.

    Args:
        mock_adata: Mock adata.

    """
    result = update_metrics_anndata(mock_adata, inplace=False)

    # Original object should be unchanged (at least for the new columns)
    assert "antibody_count" not in mock_adata.var.columns
    assert "components" not in mock_adata.var.columns

    # Result object should be updated
    assert isinstance(result, anndata.AnnData)
    assert "antibody_count" in result.var.columns
    assert "components" in result.var.columns

    # Check one value to be sure
    assert result.var.loc["ab1", "antibody_count"] == 6


def test_update_metrics_anndata_empty(mock_adata):
    """Test updating metrics with empty data.

    Args:
        mock_adata: Mock adata.

    """
    # Create empty AnnData
    empty_adata = anndata.AnnData(
        X=np.zeros((0, 0)),
        obs=pd.DataFrame(index=pd.Index([], dtype=str)),
        var=pd.DataFrame(index=pd.Index([], dtype=str)),
    )
    update_metrics_anndata(empty_adata)
    assert empty_adata.n_obs == 0
    assert empty_adata.n_vars == 0


def test_update_metrics_anndata_sparse(mock_adata):
    """Test updating metrics with sparse matrix.

    Args:
        mock_adata: Mock adata.

    """
    from scipy import sparse

    # Convert mock_adata.X to sparse format
    mock_adata.X = sparse.csr_matrix(mock_adata.X)

    # Run update in place
    update_metrics_anndata(mock_adata, inplace=True)

    # Verify results - should be same as dense test

    # antibody_count
    pd.testing.assert_series_equal(
        mock_adata.var["antibody_count"],
        pd.Series([6, 5, 4], index=["ab1", "ab2", "ab3"], name="antibody_count").astype(
            int
        ),
    )

    # components
    pd.testing.assert_series_equal(
        mock_adata.var["components"],
        pd.Series([2, 2, 1], index=["ab1", "ab2", "ab3"], name="components").astype(
            int
        ),
    )

    # antibody_pct
    expected_pct = pd.Series(
        [6 / 15, 5 / 15, 4 / 15], index=["ab1", "ab2", "ab3"], name="antibody_pct"
    ).astype(np.float32)

    pd.testing.assert_series_equal(mock_adata.var["antibody_pct"], expected_pct)

    # n_antibodies
    expected_n_antibodies = pd.Series(
        [2, 2, 1], index=["obs1", "obs2", "obs3"], name="n_antibodies"
    ).astype(np.uint32)

    pd.testing.assert_series_equal(
        mock_adata.obs["n_antibodies"], expected_n_antibodies
    )
