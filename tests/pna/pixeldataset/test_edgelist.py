"""Tests for the Edgelist wrapper class.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from pixelator.pna.pixeldataset import Edgelist, PNAPixelDataset
from pixelator.pna.pixeldataset.types import Component
from tests.pna.pixeldataset.ann_data_test_helpers import (
    StubAnnDataHelper,
    make_test_adata,
)


class TestEdgelistHelperInjection:
    """Represent test edgelist helper injection."""

    def test_components_derived_from_injected_helper(self):
        """Verify components derived from injected helper."""
        components = ["c1", "c2"]
        adata = make_test_adata(components, ["m1", "m2"], x=np.array([[1, 2], [3, 4]]))
        helper = StubAnnDataHelper(adata)

        edgelist = Edgelist(view=None, components=None, adata_helper=helper)

        assert edgelist.components == set(components)
        assert helper.read_adata_calls >= 1

    def test_explicit_components_bypass_helper(self):
        """Verify explicit components bypass helper."""
        adata = make_test_adata(
            ["c1", "c2"], ["m1", "m2"], x=np.array([[1, 2], [3, 4]])
        )
        helper = StubAnnDataHelper(adata)

        edgelist = Edgelist(view=None, components={"c1"}, adata_helper=helper)

        assert edgelist.components == {"c1"}
        assert helper.read_adata_calls == 0


class TestEdgelistIntegration:
    """Represent test edgelist integration."""

    def test_to_polars_returns_dataframe(self, pxl_dataset: PNAPixelDataset):
        """Verify to polars returns dataframe.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.edgelist().to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_to_df_returns_pandas(self, pxl_dataset: PNAPixelDataset):
        """Verify to df returns pandas.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.edgelist().to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_len_returns_edge_count(self, pxl_dataset: PNAPixelDataset):
        """Verify len returns edge count.

        Args:
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        assert len(edgelist) > 0

    def test_is_empty_returns_false_for_populated(self, pxl_dataset: PNAPixelDataset):
        """Verify is empty returns false for populated.

        Args:
        pxl_dataset: Pxl dataset.

        """
        assert not pxl_dataset.edgelist().is_empty()

    def test_components_matches_dataset(self, pxl_dataset: PNAPixelDataset):
        """Verify components matches dataset.

        Args:
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        assert edgelist.components == pxl_dataset.components()

    def test_to_record_batches_yields_arrow_batches(self, pxl_dataset: PNAPixelDataset):
        """Verify to record batches yields arrow batches.

        Args:
        pxl_dataset: Pxl dataset.

        """
        batches = list(pxl_dataset.edgelist().to_record_batches())
        assert len(batches) > 0
        assert all(isinstance(b, pa.RecordBatch) for b in batches)

    def test_iterator_yields_components(self, pxl_dataset: PNAPixelDataset):
        """Verify iterator yields components.

        Args:
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        items = list(edgelist.iterator())
        assert len(items) > 0
        for item in items:
            assert isinstance(item, Component)
            assert isinstance(item.component_id, str)
            assert isinstance(item.frame, pl.LazyFrame)

    def test_repr(self, pxl_dataset: PNAPixelDataset):
        """Verify repr.

        Args:
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        assert repr(edgelist) == str(edgelist)

    def test_ipython_display(self, pxl_dataset: PNAPixelDataset, capsys):
        """Verify ipython display.

        Args:
        capsys: capsys.
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        edgelist._ipython_display_()
        captured = capsys.readouterr()
        assert "EdgeList" in captured.out

    def test_str_representation(self, pxl_dataset: PNAPixelDataset):
        """Verify str representation.

        Args:
        pxl_dataset: Pxl dataset.

        """
        assert "EdgeList" in str(pxl_dataset.edgelist())

    def test_str_many_components_omits_set(self, pxl_dataset: PNAPixelDataset):
        """When >5 components, str should show count instead of the full set.

        Args:
        pxl_dataset: Pxl dataset.

        """
        edgelist = pxl_dataset.edgelist()
        n_components = len(edgelist.components)
        if n_components <= 5:
            assert "component set:" in str(edgelist)
        else:
            assert f"{n_components} components" in str(edgelist)
