"""Tests for AnnDataHelper wrapper behavior.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from tests.pna.conftest import create_pxl_file


def _panel_with_version_product_and_uniprot(
    panel: PNAAntibodyPanel,
    *,
    version: str,
    product: str | None,
    marker_a_uniprot: str,
    added_column_name: str | None = None,
    added_column_value: str | None = None,
) -> PNAAntibodyPanel:
    """Clone a panel while tweaking version/product and marker metadata for tests."""
    panel_df = panel.df.copy()
    panel_df.loc["MarkerA", "uniprot_id"] = marker_a_uniprot
    if added_column_name is not None and added_column_value is not None:
        panel_df[added_column_name] = added_column_value
    metadata = panel.metadata.model_copy(
        update={"version": version, "product": product}
    )
    return PNAAntibodyPanel(df=panel_df, metadata=metadata)


def _write_component_suffix_parquet(source: Path, target: Path, suffix: str) -> None:
    """Write a parquet copy where `component` values are suffixed to avoid overlap."""
    (
        pl.scan_parquet(source)
        .with_columns((pl.col("component") + suffix).alias("component"))
        .sink_parquet(target)
    )


def _build_two_sample_dataset_with_panels(
    *,
    tmp_path: Path,
    edgelist_parquet_path: Path,
    panel_old: PNAAntibodyPanel,
    panel_new: PNAAntibodyPanel,
) -> PNAPixelDataset:
    """Create two on-disk PXL samples with distinct panels for bumping patch version tests."""
    sample_old = create_pxl_file(
        target=tmp_path / "sample_old.pxl",
        sample_name="sample_old",
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=None,
        layout_parquet_path=None,
        panel=panel_old,
    )

    sample_new_edgelist = tmp_path / "sample_new_edgelist.parquet"

    _write_component_suffix_parquet(
        source=edgelist_parquet_path,
        target=sample_new_edgelist,
        suffix="_sample_new",
    )

    sample_new = create_pxl_file(
        target=tmp_path / "sample_new.pxl",
        sample_name="sample_new",
        edgelist_parquet_path=sample_new_edgelist,
        proximity_parquet_path=None,
        layout_parquet_path=None,
        panel=panel_new,
    )
    return PNAPixelDataset.from_pxl_files([sample_old, sample_new])


class TestAnnDataHelper:
    def test_anndata_helper_matches_dataset_adata_no_transforms(
        self, pxl_dataset, adata_data
    ):
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"

        helper = AnnDataHelper(pxl_dataset.view)
        res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        adata_assert_equal(res, adata_data)

    def test_anndata_helper_respects_component_and_marker_filters(self, pxl_dataset):
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
    helper = AnnDataHelper(pxl_dataset.view, components=components, markers=markers)
    res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
    assert res.n_obs >= 0
    assert res.n_vars >= 0


class TestTryBumpAdataPanelVersion:
    """Coverage for automatic panel patch bump behavior in AnnDataHelper."""

    def test_bumps_to_latest_patch_when_prerequisites_are_met(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
    ):
        """Bump to latest patch when major/minor/product prerequisites are satisfied."""
        panel_old = _panel_with_version_product_and_uniprot(
            panel,
            version="0.1.0",
            product="test-product",
            marker_a_uniprot="P12345",
        )
        panel_new = _panel_with_version_product_and_uniprot(
            panel,
            version="0.1.1",
            product="test-product",
            marker_a_uniprot="Q9UPN0",
            added_column_name="target_class",
            added_column_value="new-value",
        )
        dataset = _build_two_sample_dataset_with_panels(
            tmp_path=tmp_path,
            edgelist_parquet_path=edgelist_parquet_path,
            panel_old=panel_old,
            panel_new=panel_new,
        )
        helper = AnnDataHelper(dataset.view)

        with dataset.view.open() as session:
            adata_old = helper._read_adata_from_sample(
                session=session, sample="sample_old"
            )
            adata_new = helper._read_adata_from_sample(
                session=session, sample="sample_new"
            )

        # Also test non panel columns are kept as is during the bump
        positive_cells_count = np.random.randint(0, 100, adata_old.var.shape[0])
        adata_old.var["positive_cells_count"] = positive_cells_count

        assert adata_old.var.loc["MarkerA", "uniprot_id"] == "P12345"
        assert adata_new.var.loc["MarkerA", "uniprot_id"] == "Q9UPN0"
        assert "target_class" not in adata_old.var.columns
        assert "target_class" in adata_new.var.columns

        bumped = helper._try_bump_adata_panel_version([adata_old, adata_new])

        assert "target_class" in bumped[0].var.columns
        assert bumped[0].var.loc["MarkerA", "target_class"] == "new-value"
        assert bumped[0].var.loc["MarkerA", "uniprot_id"] == "Q9UPN0"

        assert "target_class" in bumped[1].var.columns
        assert bumped[1].var.loc["MarkerA", "uniprot_id"] == "Q9UPN0"
        assert bumped[1].var.loc["MarkerA", "target_class"] == "new-value"

        assert "positive_cells_count" in bumped[0].var.columns
        assert "positive_cells_count" not in bumped[1].var.columns
        assert (
            adata_old.var["positive_cells_count"]
            == bumped[0].var["positive_cells_count"]
        ).all()

        assert (adata_old[:, "MarkerC"].X == bumped[0][:, "MarkerC"].X).all()
        assert (adata_new[:, "MarkerC"].X == bumped[1][:, "MarkerC"].X).all()

    @pytest.mark.parametrize(
        "new_version,new_product",
        [
            ("0.2.0", "test-product"),
            ("0.1.1", "different-product"),
            ("0.1.1", None),  # product is None
        ],
    )
    def test_skips_bump_when_prerequisites_are_not_met(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
        new_version: str,
        new_product: str | None,
    ):
        """Skip bump when version compatibility or product prerequisites are not met."""
        panel_old = _panel_with_version_product_and_uniprot(
            panel,
            version="0.1.0",
            product="test-product",
            marker_a_uniprot="P12345",
        )
        panel_new = _panel_with_version_product_and_uniprot(
            panel,
            version=new_version,
            product=new_product,
            marker_a_uniprot="Q9UPN0",
            added_column_name="target_class",
            added_column_value="new-version",
        )
        dataset = _build_two_sample_dataset_with_panels(
            tmp_path=tmp_path,
            edgelist_parquet_path=edgelist_parquet_path,
            panel_old=panel_old,
            panel_new=panel_new,
        )
        helper = AnnDataHelper(dataset.view)

        with dataset.view.open() as session:
            adata_old = helper._read_adata_from_sample(
                session=session, sample="sample_old"
            )
            adata_new = helper._read_adata_from_sample(
                session=session, sample="sample_new"
            )

        not_bumped = helper._try_bump_adata_panel_version([adata_old, adata_new])

        assert "target_class" not in not_bumped[0].var.columns

        assert (adata_old[:, "MarkerC"].X == not_bumped[0][:, "MarkerC"].X).all()
        assert (adata_new[:, "MarkerC"].X == not_bumped[1][:, "MarkerC"].X).all()
