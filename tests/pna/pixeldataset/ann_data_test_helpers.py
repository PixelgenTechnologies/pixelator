"""Shared AnnData stubs for pixeldataset unit tests.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


class StubAnnDataHelper:
    """Minimal stand-in for AnnDataHelper that returns a fixed AnnData."""

    def __init__(self, adata: AnnData):
        """Initialize the instance.

        Args:
            adata: Adata.

        """
        self._adata = adata
        self.read_adata_calls: int = 0

    def read_adata(
        self, *, add_log1p_transform: bool, add_clr_transform: bool
    ) -> AnnData:
        """Read adata.

        Returns:
                Result (AnnData).


        Args:
            add_log1p_transform: Add log1p transform.
            add_clr_transform: Add clr transform.

        """
        self.read_adata_calls += 1
        return self._adata


def make_test_adata(
    components: list[str], markers: list[str], x: np.ndarray
) -> AnnData:
    """Build a small AnnData with component index and marker_id columns.

    Args:
        components: Components.
        markers: Markers.
        x: X.

    """
    obs = pd.DataFrame(index=pd.Index(components, name="component"))
    var = pd.DataFrame(index=pd.Index(markers, name="marker_id"))
    return AnnData(X=x, obs=obs, var=var)
