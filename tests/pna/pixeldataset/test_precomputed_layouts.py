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
    def test_components_derived_from_injected_helper(self):
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
    def test_to_polars_returns_dataframe(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.precomputed_layouts().to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_to_df_returns_pandas(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.precomputed_layouts().to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_len_returns_node_count(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert len(layouts) > 0

    def test_is_empty_returns_false_for_populated(self, pxl_dataset: PNAPixelDataset):
        assert not pxl_dataset.precomputed_layouts().is_empty()

    def test_components_matches_dataset(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert layouts.components == pxl_dataset.components()

    def test_spherical_norm_columns(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.precomputed_layouts(add_spherical_norm=True).to_polars()
        for col in ["x_norm", "y_norm", "z_norm"]:
            assert col in df.columns

    def test_marker_count_columns_present(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.precomputed_layouts(add_marker_counts=True).to_polars()
        marker_cols = {"MarkerA", "MarkerB", "MarkerC"}
        assert marker_cols.issubset(set(df.columns))

    def test_no_marker_count_columns_when_disabled(self, pxl_dataset: PNAPixelDataset):
        df = pxl_dataset.precomputed_layouts(add_marker_counts=False).to_polars()
        marker_cols = {"MarkerA", "MarkerB", "MarkerC"}
        assert not marker_cols.intersection(set(df.columns))

    def test_iterator_yields_component_dataframes(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        for comp_id, component_df in layouts.iterator():
            assert isinstance(comp_id, str)
            assert isinstance(component_df, pd.DataFrame)

    def test_iterator_returns_polars_when_requested(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        for comp_id, component_df in layouts.iterator(return_polars_df=True):
            assert isinstance(comp_id, str)
            assert isinstance(component_df, pl.DataFrame)

    def test_str_representation(self, pxl_dataset: PNAPixelDataset):
        assert "PreComputedLayouts" in str(pxl_dataset.precomputed_layouts())

    def test_describe(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        desc = layouts.describe()
        assert "PreComputedLayouts" in desc
        assert "components" in desc
        assert "datapoints" in desc

    def test_repr(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert repr(layouts) == str(layouts)

    def test_ipython_display(self, pxl_dataset: PNAPixelDataset, capsys):
        layouts = pxl_dataset.precomputed_layouts()
        layouts._ipython_display_()
        captured = capsys.readouterr()
        assert "PreComputedLayouts" in captured.out
