"""Tests for the Edgelist wrapper class.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.pna.pixeldataset import Edgelist, PNAPixelDataset


class StubAnnDataHelper:
    def __init__(self, adata: AnnData):
        self._adata = adata
        self.read_adata_calls: int = 0

    def read_adata(
        self, *, add_log1p_transform: bool, add_clr_transform: bool
    ) -> AnnData:
        self.read_adata_calls += 1
        return self._adata


def _make_adata(components: list[str], markers: list[str], x: np.ndarray) -> AnnData:
    obs = pd.DataFrame(index=pd.Index(components, name="component"))
    var = pd.DataFrame(index=pd.Index(markers, name="marker_id"))
    return AnnData(X=x, obs=obs, var=var)


class TestEdgelistHelperInjection:
    def test_components_derived_from_injected_helper(self):
        components = ["c1", "c2"]
        adata = _make_adata(components, ["m1", "m2"], x=np.array([[1, 2], [3, 4]]))
        helper = StubAnnDataHelper(adata)

        edgelist = Edgelist(view=None, components=None, adata_helper=helper)

        assert edgelist.components == set(components)
        assert helper.read_adata_calls >= 1

    def test_explicit_components_bypass_helper(self):
        adata = _make_adata(["c1", "c2"], ["m1", "m2"], x=np.array([[1, 2], [3, 4]]))
        helper = StubAnnDataHelper(adata)

        edgelist = Edgelist(view=None, components={"c1"}, adata_helper=helper)

        assert edgelist.components == {"c1"}
        assert helper.read_adata_calls == 0


class TestEdgelistIntegration:
    def test_to_polars_returns_dataframe(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.edgelist().to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_to_df_returns_pandas(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.edgelist().to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_len_returns_edge_count(self, pxl_dataset: PNAPixelDataset):
        edgelist = pxl_dataset.edgelist()
        assert len(edgelist) > 0

    def test_is_empty_returns_false_for_populated(self, pxl_dataset: PNAPixelDataset):
        assert not pxl_dataset.edgelist().is_empty()

    def test_components_matches_dataset(self, pxl_dataset: PNAPixelDataset):
        edgelist = pxl_dataset.edgelist()
        assert edgelist.components == pxl_dataset.components()

    def test_str_representation(self, pxl_dataset: PNAPixelDataset):
        assert "EdgeList" in str(pxl_dataset.edgelist())
