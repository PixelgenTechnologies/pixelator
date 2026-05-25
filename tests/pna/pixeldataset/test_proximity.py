"""Tests for the Proximity wrapper class.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.proximity import Proximity
from tests.pna.pixeldataset.ann_data_test_helpers import (
    StubAnnDataHelper,
    make_test_adata,
)


class TestProximityHelperInjection:
    """Represent test proximity helper injection."""

    def test_components_derived_from_injected_helper(self):
        """Verify components derived from injected helper."""
        components = ["c1", "c2"]
        adata = make_test_adata(components, ["m1", "m2"], x=np.array([[1, 2], [3, 4]]))
        helper = StubAnnDataHelper(adata)

        prox = Proximity(
            view=None,
            components=None,
            markers=None,
            adata_helper=helper,
            add_marker_counts=False,
            add_log2_ratio=False,
        )

        assert prox.components == set(components)
        assert helper.read_adata_calls >= 1

    def test_markers_derived_from_injected_helper(self):
        """Verify markers derived from injected helper."""
        markers = ["m1", "m2"]
        adata = make_test_adata(["c1"], markers, x=np.array([[1, 2]]))
        helper = StubAnnDataHelper(adata)

        prox = Proximity(
            view=None,
            components=None,
            markers=None,
            adata_helper=helper,
            add_marker_counts=False,
            add_log2_ratio=False,
        )

        assert prox.markers == set(markers)

    def test_explicit_components_markers_bypass_helper(self):
        """Verify explicit components markers bypass helper."""
        adata = make_test_adata(
            ["c1", "c2"], ["m1", "m2"], x=np.array([[1, 2], [3, 4]])
        )
        helper = StubAnnDataHelper(adata)

        prox = Proximity(
            view=None,
            components={"c1"},
            markers={"m2"},
            adata_helper=helper,
            add_marker_counts=False,
            add_log2_ratio=False,
        )

        assert prox.components == {"c1"}
        assert prox.markers == {"m2"}
        assert helper.read_adata_calls == 0

    def test_post_process_uses_helper_for_marker_counts(self):
        """Verify post process uses helper for marker counts."""
        adata = make_test_adata(
            ["c1"], ["m1", "m2"], x=np.array([[10, 5]], dtype=np.uint32)
        )
        helper = StubAnnDataHelper(adata)

        prox = Proximity(
            view=None,
            components=None,
            markers=None,
            adata_helper=helper,
            add_marker_counts=True,
            add_log2_ratio=False,
        )

        input_df = pl.DataFrame(
            {
                "component": ["c1"],
                "marker_1": ["m1"],
                "marker_2": ["m2"],
                "join_count": [1],
                "join_count_expected_mean": [1],
            }
        )

        out = prox._post_process(input_df)

        assert helper.read_adata_calls == 1
        for col in [
            "marker_1_count",
            "marker_1_freq",
            "marker_2_count",
            "marker_2_freq",
            "min_count",
        ]:
            assert col in out.columns

        assert out["marker_1_count"][0] == 10
        assert out["marker_2_count"][0] == 5
        assert out["min_count"][0] == 5
        assert out["marker_1_freq"][0] == pytest.approx(10 / 15)
        assert out["marker_2_freq"][0] == pytest.approx(5 / 15)

    def test_post_process_adds_log2_ratio(self):
        """Verify post process adds log2 ratio."""
        adata = make_test_adata(["c1"], ["m1", "m2"], x=np.array([[10, 5]]))
        helper = StubAnnDataHelper(adata)

        prox = Proximity(
            view=None,
            components=None,
            markers=None,
            adata_helper=helper,
            add_marker_counts=False,
            add_log2_ratio=True,
        )

        input_df = pl.DataFrame(
            {
                "component": ["c1"],
                "marker_1": ["m1"],
                "marker_2": ["m2"],
                "join_count": [4],
                "join_count_expected_mean": [2],
            }
        )

        out = prox._post_process(input_df)
        assert "log2_ratio" in out.columns
        assert out["log2_ratio"][0] == pytest.approx(np.log2(4 / 2))


class TestProximityIntegration:
    """Represent test proximity integration."""

    def test_to_polars_returns_dataframe(self, pxl_dataset: PNAPixelDataset):
        """Verify to polars returns dataframe.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.proximity().to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_to_df_returns_pandas(self, pxl_dataset: PNAPixelDataset):
        """Verify to df returns pandas.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.proximity().to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_len_returns_proximity_count(self, pxl_dataset: PNAPixelDataset):
        """Verify len returns proximity count.

        Args:
        pxl_dataset: Pxl dataset.

        """
        prox = pxl_dataset.proximity()
        assert len(prox) > 0

    def test_is_empty_returns_false_for_populated(self, pxl_dataset: PNAPixelDataset):
        """Verify is empty returns false for populated.

        Args:
        pxl_dataset: Pxl dataset.

        """
        assert not pxl_dataset.proximity().is_empty()

    def test_marker_counts_columns_present(self, pxl_dataset: PNAPixelDataset):
        """Verify marker counts columns present.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.proximity().to_polars()
        for col in [
            "marker_1_count",
            "marker_2_count",
            "marker_1_freq",
            "marker_2_freq",
            "min_count",
        ]:
            assert col in df.columns

    def test_log2_ratio_column_present(self, pxl_dataset: PNAPixelDataset):
        """Verify log2 ratio column present.

        Args:
        pxl_dataset: Pxl dataset.

        """
        df = pxl_dataset.proximity().to_polars()
        assert "log2_ratio" in df.columns

    def test_str_representation(self, pxl_dataset: PNAPixelDataset):
        """Verify str representation.

        Args:
        pxl_dataset: Pxl dataset.

        """
        assert "Proximity" in str(pxl_dataset.proximity())

    def test_repr(self, pxl_dataset: PNAPixelDataset):
        """Verify repr.

        Args:
        pxl_dataset: Pxl dataset.

        """
        prox = pxl_dataset.proximity()
        assert repr(prox) == str(prox)

    def test_ipython_display(self, pxl_dataset: PNAPixelDataset, capsys):
        """Verify ipython display.

        Args:
        capsys: capsys.
        pxl_dataset: Pxl dataset.

        """
        prox = pxl_dataset.proximity()
        prox._ipython_display_()
        captured = capsys.readouterr()
        assert "Proximity" in captured.out
