"""Tests for AnnDataHelper wrapper behavior.

Copyright © 2025 Pixelgen Technologies AB.
"""

import shutil
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from tests.pna.conftest import create_pxl_file


class TestAnnDataHelper:
    """Represent test ann data helper."""

    def test_anndata_helper_matches_dataset_adata_no_transforms(
        self, pxl_dataset, adata_data
    ):
        """Verify anndata helper matches dataset adata no transforms.

        Args:
            pxl_dataset: pxl dataset.
            adata_data: adata data.
        """
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"

        helper = AnnDataHelper(pxl_dataset.view)
        res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        adata_assert_equal(res, adata_data)

    def test_anndata_helper_respects_component_and_marker_filters(self, pxl_dataset):
        """Verify anndata helper respects component and marker filters.

        Args:
            pxl_dataset: pxl dataset.
        """
        filtered = pxl_dataset.filter(
            components={"fc07dea9b679aca7"},
            markers={"MarkerA"},
        )

        helper = AnnDataHelper(
            pxl_dataset.view,
            components={"fc07dea9b679aca7"},
            markers={"MarkerA"},
        )
        res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)

        assert set(res.obs.index) == {"fc07dea9b679aca7"}
        assert set(res.var.index) == {"MarkerA"}

        adata_assert_equal(
            res,
            filtered.adata(add_clr_transform=False, add_log1p_transform=False),
        )

    def test_anndata_helper_does_not_mutate_original(self, pxl_dataset):
        """Verify anndata helper does not mutate original.

        Args:
            pxl_dataset: pxl dataset.
        """
        helper = AnnDataHelper(pxl_dataset.view)

        adata = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        adata.layers["new_layer"] = adata.X + 1

        assert "new_layer" in adata.layers.keys()
        # Each call should return an independent AnnData object; callers may
        # mutate layers without affecting subsequent reads.
        adata2 = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        assert adata is not adata2
        assert "new_layer" not in adata2.layers.keys()


@pytest.mark.parametrize(
    "components,markers",
    [
        (None, None),
        ({"fc07dea9b679aca7"}, None),
        (None, {"MarkerA"}),
        ({"fc07dea9b679aca7"}, {"MarkerA"}),
    ],
)
def test_anndata_helper_basic_smoke(pxl_dataset, components, markers):
    """Verify anndata helper basic smoke.

    Args:
        pxl_dataset: pxl dataset.
        components: components.
        markers: markers.
    """
    helper = AnnDataHelper(pxl_dataset.view, components=components, markers=markers)
    res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
    assert res.n_obs >= 0
    assert res.n_vars >= 0


def test_record_batches_rename_legacy_marker_columns(tmp_path: Path, pxl_file: Path):
    """A legacy edgelist stream uses the same column names as ``to_polars``."""
    legacy = tmp_path / "legacy.pxl"
    shutil.copy(pxl_file, legacy)
    with duckdb.connect(str(legacy)) as connection:
        connection.execute("ALTER TABLE edgelist RENAME COLUMN marker_1 TO marker1")
        connection.execute("ALTER TABLE edgelist RENAME COLUMN marker_2 TO marker2")
    dataset = PNAPixelDataset.from_files(legacy)
    streamed = pl.concat(
        [pl.from_arrow(batch) for batch in dataset.edgelist().to_record_batches()],
        how="vertical",
    )
    loaded = dataset.edgelist().to_polars()
    assert "marker_1" in streamed.columns
    assert "marker_2" in streamed.columns
    assert "marker1" not in streamed.columns
    assert "marker2" not in streamed.columns
    assert streamed.sort(streamed.columns).equals(loaded.sort(loaded.columns))
