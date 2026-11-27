"""Tests for AnnDataHelper wrapper behavior.

Copyright © 2025 Pixelgen Technologies AB.
"""

import shutil
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from pixelator.common.statistics import clr_transformation, log1p_transformation
from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from tests.pna.conftest import create_pxl_file


class TestAnnDataHelper:
    """Represent test ann data helper."""

    def test_anndata_helper_matches_dataset_adata_no_transforms(
        self, pxl_dataset, adata_data, panel
    ):
        """Verify anndata helper matches dataset adata no transforms.

        Args:
            pxl_dataset: pxl dataset.
            adata_data: adata data.
            panel: panel.
        """
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"
        adata_data.var = adata_data.var.join(
            panel.marker_table().reindex(adata_data.var_names), how="left"
        )

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

        # The dataset drops the panel from uns, the helper keeps it if stored there
        res.uns.pop("panel_metadata", None)
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


@pytest.mark.parametrize(
    "add_clr_transform, add_log1p_transform",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_anndata_helper_adds_exactly_the_requested_transforms(
    pxl_dataset, add_clr_transform, add_log1p_transform
):
    """Each requested transform is computed from X, and only those are added.

    Args:
        pxl_dataset: pxl dataset.
        add_clr_transform: whether clr is requested.
        add_log1p_transform: whether log1p is requested.
    """
    res = AnnDataHelper(pxl_dataset.view).read_adata(
        add_clr_transform=add_clr_transform, add_log1p_transform=add_log1p_transform
    )

    requested = {"clr": add_clr_transform, "log1p": add_log1p_transform}
    assert set(res.obsm) == {key for key, wanted in requested.items() if wanted}
    counts = res.to_df()
    if add_clr_transform:
        np.testing.assert_allclose(
            res.obsm["clr"].to_numpy(),
            clr_transformation(df=counts, axis=1).to_numpy(),
        )
    if add_log1p_transform:
        np.testing.assert_allclose(
            res.obsm["log1p"].to_numpy(), log1p_transformation(df=counts).to_numpy()
        )


def test_anndata_helper_ignores_stored_transforms(tmp_path: Path, pxl_file: Path):
    """Transforms stored by older versions are never returned; other obsm keys are.

    Args:
        tmp_path: tmp path.
        pxl_file: pxl file.
    """
    legacy = tmp_path / "legacy.pxl"
    shutil.copy(pxl_file, legacy)
    with duckdb.connect(str(legacy)) as connection:
        x = connection.execute('SELECT * FROM "__adata__X"').pl()
        stale = x.with_columns(pl.exclude("index").cast(pl.Float64) * 0 - 1)
        for key in ("clr", "log1p", "custom"):
            connection.execute(
                f"CREATE TABLE __adata__obsm_{key} AS SELECT * FROM stale"
            )
    helper = AnnDataHelper(PNAPixelDataset.from_files(legacy).view)

    plain = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
    clr_only = helper.read_adata(add_clr_transform=True, add_log1p_transform=False)

    assert set(plain.obsm) == {"custom"}
    assert set(clr_only.obsm) == {"clr", "custom"}
    np.testing.assert_allclose(
        clr_only.obsm["clr"].to_numpy(),
        clr_transformation(df=clr_only.to_df(), axis=1).to_numpy(),
    )


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
