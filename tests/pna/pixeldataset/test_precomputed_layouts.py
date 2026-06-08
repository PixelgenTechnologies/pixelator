"""Tests for the PreComputedLayouts wrapper class.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import polars as pl

from pixelator.pna.pixeldataset import PNAPixelDataset, PreComputedLayouts
from tests.pna.pixeldataset.ann_data_test_helpers import (
    StubAnnDataHelper,
    make_test_adata,
)


class TestPreComputedLayoutsHelperInjection:
    """Represent test pre computed layouts helper injection."""

    def test_components_derived_from_injected_helper(self):
        """Verify components derived from injected helper."""
        components = ["c1", "c2"]
        adata = make_test_adata(components, ["m1", "m2"], x=np.array([[1, 2], [3, 4]]))
        helper = StubAnnDataHelper(adata)

        layouts = PreComputedLayouts(
            view=None,
            components=None,
            adata_helper=helper,
            add_marker_counts=False,
        )

        assert layouts.components == set(components)
        assert helper.read_adata_calls >= 1

    def test_explicit_components_bypass_helper(self):
        """Verify explicit components bypass helper."""
        adata = make_test_adata(
            ["c1", "c2"], ["m1", "m2"], x=np.array([[1, 2], [3, 4]])
        )
        helper = StubAnnDataHelper(adata)

        layouts = PreComputedLayouts(
            view=None,
            components={"c1"},
            adata_helper=helper,
            add_marker_counts=False,
        )

        assert layouts.components == {"c1"}
        assert helper.read_adata_calls == 0


class TestPreComputedLayoutsIntegration:
    """Represent test pre computed layouts integration."""

    def test_to_polars_returns_dataframe(self, pxl_dataset: PNAPixelDataset):
        """Verify to polars returns dataframe.

        Args:
            pxl_dataset: Pxl dataset.
        """
        df = pxl_dataset.precomputed_layouts().to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_to_df_returns_pandas(self, pxl_dataset: PNAPixelDataset):
        """Verify to df returns pandas.

        Args:
            pxl_dataset: Pxl dataset.
        """
        df = pxl_dataset.precomputed_layouts().to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_len_returns_node_count(self, pxl_dataset: PNAPixelDataset):
        """Verify len returns node count.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        assert len(layouts) > 0

    def test_is_empty_returns_false_for_populated(self, pxl_dataset: PNAPixelDataset):
        """Verify is empty returns false for populated.

        Args:
            pxl_dataset: Pxl dataset.
        """
        assert not pxl_dataset.precomputed_layouts().is_empty()

    def test_components_matches_dataset(self, pxl_dataset: PNAPixelDataset):
        """Verify components matches dataset.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        assert layouts.components == pxl_dataset.components()

    def test_spherical_norm_columns(self, pxl_dataset: PNAPixelDataset):
        """Verify spherical norm columns.

        Args:
            pxl_dataset: Pxl dataset.
        """
        df = pxl_dataset.precomputed_layouts(add_spherical_norm=True).to_polars()
        for col in ["x_norm", "y_norm", "z_norm"]:
            assert col in df.columns

    def test_marker_count_columns_present(self, pxl_dataset: PNAPixelDataset):
        """Verify marker count columns present.

        Args:
            pxl_dataset: Pxl dataset.
        """
        df = pxl_dataset.precomputed_layouts(add_marker_counts=True).to_polars()
        marker_cols = {"MarkerA", "MarkerB", "MarkerC"}
        assert marker_cols.issubset(set(df.columns))

    def test_no_marker_count_columns_when_disabled(self, pxl_dataset: PNAPixelDataset):
        """Verify no marker count columns when disabled.

        Args:
            pxl_dataset: Pxl dataset.
        """
        df = pxl_dataset.precomputed_layouts(add_marker_counts=False).to_polars()
        marker_cols = {"MarkerA", "MarkerB", "MarkerC"}
        assert not marker_cols.intersection(set(df.columns))

    def test_iterator_yields_component_dataframes(self, pxl_dataset: PNAPixelDataset):
        """Verify iterator yields component dataframes.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        for comp_id, component_df in layouts.iterator():
            assert isinstance(comp_id, str)
            assert isinstance(component_df, pd.DataFrame)

    def test_iterator_returns_polars_when_requested(self, pxl_dataset: PNAPixelDataset):
        """Verify iterator returns polars when requested.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        for comp_id, component_df in layouts.iterator(return_polars_df=True):
            assert isinstance(comp_id, str)
            assert isinstance(component_df, pl.DataFrame)

    def test_str_representation(self, pxl_dataset: PNAPixelDataset):
        """Verify str representation.

        Args:
            pxl_dataset: Pxl dataset.
        """
        assert "PreComputedLayouts" in str(pxl_dataset.precomputed_layouts())

    def test_describe(self, pxl_dataset: PNAPixelDataset):
        """Verify describe.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        desc = layouts.describe()
        assert "PreComputedLayouts" in desc
        assert "components" in desc
        assert "datapoints" in desc

    def test_repr(self, pxl_dataset: PNAPixelDataset):
        """Verify repr.

        Args:
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        assert repr(layouts) == str(layouts)

    def test_ipython_display(self, pxl_dataset: PNAPixelDataset, capsys):
        """Verify ipython display.

        Args:
            capsys: capsys.
            pxl_dataset: Pxl dataset.
        """
        layouts = pxl_dataset.precomputed_layouts()
        layouts._ipython_display_()
        captured = capsys.readouterr()
        assert "PreComputedLayouts" in captured.out
