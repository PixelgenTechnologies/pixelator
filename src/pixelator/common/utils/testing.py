"""Utility functions used for testing.

Copyright © 2026 Pixelgen Technologies AB.
"""

from anndata import AnnData
from pandas.testing import assert_frame_equal, assert_index_equal


def adata_assert_equal(actual: AnnData, expected: AnnData):
    """Assert that two AnnData objects are equal, ignoring the order of rows and columns."""
    assert_index_equal(actual.obs_names, expected.obs_names, check_order=False)
    assert_index_equal(actual.var_names, expected.var_names, check_order=False)

    # Check obs
    actual_obs = actual.obs.sort_index().sort_index(axis=1)
    expected_obs = expected.obs.sort_index().sort_index(axis=1)
    assert_frame_equal(actual_obs, expected_obs)

    # Check var
    actual_var = actual.var.sort_index()
    expected_var = expected.var.sort_index()
    assert_frame_equal(actual_var, expected_var)

    # Check uns
    assert actual.uns == expected.uns, f"Uns mismatch: {actual.uns} != {expected.uns}"

    # Check X
    assert_frame_equal(
        actual.to_df().sort_index().sort_index(axis=1),
        expected.to_df().sort_index().sort_index(axis=1),
    )
