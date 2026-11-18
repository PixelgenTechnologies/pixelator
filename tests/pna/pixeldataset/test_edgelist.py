"""Tests for the Edgelist wrapper class.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from pixelator.pna.pixeldataset import Edgelist, PNAPixelDataset
from pixelator.pna.pixeldataset.types import Component
from tests.pna.conftest import create_pxl_file
from tests.pna.pixeldataset.ann_data_test_helpers import (
    StubAnnDataHelper,
    make_test_adata,
)

MARKER_SEQUENCE_COLUMNS = {"marker_sequence_1", "marker_sequence_2"}


@pytest.fixture(name="pxl_dataset_with_marker_sequences", scope="module")
def pxl_dataset_with_marker_sequences_fixture(
    tmp_path_factory, edgelist_dataframe, panel
):
    """A dataset whose edgelist has the marker sequence columns."""
    tmp_dir = tmp_path_factory.mktemp("marker_sequences")
    edgelist_path = tmp_dir / "edgelist.parquet"
    edgelist_dataframe.with_columns(
        marker_sequence_1=pl.lit("ACGTACGTAC"),
        marker_sequence_2=pl.lit("TTTTGGGGCC"),
    ).write_parquet(edgelist_path)
    pxl_file = create_pxl_file(
        target=tmp_dir / "file.pxl",
        sample_name="test_sample",
        edgelist_parquet_path=edgelist_path,
        proximity_parquet_path=None,
        layout_parquet_path=None,
        panel=panel,
    )
    return PNAPixelDataset.from_pxl_files(pxl_file)


class TestEdgelistMarkerSequences:
    """Represent test edgelist marker sequences."""

    def test_to_polars_keeps_marker_sequences(
        self, pxl_dataset_with_marker_sequences: PNAPixelDataset
    ):
        """Verify to polars keeps the marker sequence columns.

        Args:
            pxl_dataset_with_marker_sequences: Pxl dataset with marker sequences.
        """
        df = pxl_dataset_with_marker_sequences.edgelist().to_polars()
        assert MARKER_SEQUENCE_COLUMNS <= set(df.columns)

    def test_iterator_leaves_out_marker_sequences(
        self, pxl_dataset_with_marker_sequences: PNAPixelDataset
    ):
        """Verify iterator frames leave out the marker sequence columns.

        Args:
            pxl_dataset_with_marker_sequences: Pxl dataset with marker sequences.
        """
        items = list(pxl_dataset_with_marker_sequences.edgelist().iterator())
        assert len(items) > 0
        for item in items:
            columns = set(item.frame.collect_schema().names())
            assert not columns & MARKER_SEQUENCE_COLUMNS
            assert {"umi1", "umi2", "marker_1", "marker_2", "component"} <= columns

    @pytest.mark.parametrize("include", [True, False])
    def test_to_record_batches_marker_sequences(
        self, pxl_dataset_with_marker_sequences: PNAPixelDataset, include: bool
    ):
        """Verify to record batches includes marker sequences only when asked.

        Args:
            pxl_dataset_with_marker_sequences: Pxl dataset with marker sequences.
            include: Whether to include the marker sequence columns.
        """
        batches = list(
            pxl_dataset_with_marker_sequences.edgelist().to_record_batches(
                include_marker_sequences=include
            )
        )
        assert len(batches) > 0
        for batch in batches:
            has_sequences = MARKER_SEQUENCE_COLUMNS <= set(batch.schema.names)
            assert has_sequences == include

    def test_record_batches_without_marker_sequences_on_file_lacking_them(
        self, pxl_dataset: PNAPixelDataset
    ):
        """Verify leaving out marker sequences works on files that never had them.

        Args:
            pxl_dataset: Pxl dataset.
        """
        with_all = list(pxl_dataset.edgelist().to_record_batches())
        without = list(
            pxl_dataset.edgelist().to_record_batches(include_marker_sequences=False)
        )
        assert [b.schema.names for b in without] == [b.schema.names for b in with_all]


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
