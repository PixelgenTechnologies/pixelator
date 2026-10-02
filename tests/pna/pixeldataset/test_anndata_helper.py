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
from pixelator.pna.pixeldataset.io import Query, read_dataset_panel
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from tests.pna.conftest import create_pxl_file


def _panel_with_version_product_and_uniprot(
    panel: PNAAntibodyPanel,
    *,
    version: str,
    product: str | None,
    marker_a_uniprot: str,
    marker_a_new_name: str | None = None,
    added_column_name: str | None = None,
    added_column_value: str | None = None,
) -> PNAAntibodyPanel:
    """Clone a panel while tweaking version/product and marker metadata for tests.

    Args:
        panel: Panel.
        version: Version.
        product: Product.
        marker_a_uniprot: Marker a uniprot.
        marker_a_new_name: Marker a new name.
        added_column_name: Added column name.
        added_column_value: Added column value.
    """
    panel_df = panel.df.copy()
    panel_df.loc["MarkerA", "uniprot_id"] = marker_a_uniprot
    if marker_a_new_name is not None:
        panel_df = panel_df.rename(index={"MarkerA": marker_a_new_name})
    if added_column_name is not None and added_column_value is not None:
        panel_df[added_column_name] = added_column_value
    metadata = panel.metadata.model_copy(
        update={"version": version, "product": product}
    )
    return PNAAntibodyPanel(df=panel_df, metadata=metadata)


def _write_component_suffix_parquet(source: Path, target: Path, suffix: str) -> None:
    """Write a parquet copy where `component` values are suffixed to avoid overlap.

    Args:
        source: Source.
        target: Target.
        suffix: Suffix.
    """
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
    proximity_old: Path | None = None,
    proximity_new: Path | None = None,
) -> PNAPixelDataset:
    """Create two on-disk PXL samples with distinct panels for bumping patch version tests.

    Args:
        tmp_path: Tmp path.
        edgelist_parquet_path: Edgelist parquet path.
        panel_old: Panel old.
        panel_new: Panel new.
    """
    sample_old = create_pxl_file(
        target=tmp_path / "sample_old.pxl",
        sample_name="sample_old",
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_old,
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
        proximity_parquet_path=proximity_new,
        layout_parquet_path=None,
        panel=panel_new,
    )
    return PNAPixelDataset.from_pxl_files([sample_old, sample_new])


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
            panel.df.reindex(adata_data.var_names), how="left"
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


class TestTryBumpAdataPanelVersion:
    """Coverage for automatic panel patch bump behavior in AnnDataHelper."""

    def test_bumps_to_latest_patch_when_prerequisites_are_met(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
    ):
        """Bump to latest patch when major/minor/product prerequisites are satisfied.

        Args:
            tmp_path: Tmp path.
            edgelist_parquet_path: Edgelist parquet path.
            panel: Panel.
        """
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
            marker_a_new_name="MarkerANew",
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

        assert "MarkerA" in adata_old.var_names
        assert "MarkerANew" in adata_new.var_names
        assert "uniprot_id" not in adata_old.var.columns
        assert "target_class" not in adata_old.var.columns
        adata_old.var["uniprot_id"] = "STALE"
        adata_old.var["retired_field"] = "old-value"
        adata_old.uns["panel_metadata"] = {
            "name": "stale",
            "panel_columns": ["uniprot_id", "retired_field"],
        }

        bumped = dataset.view.apply_panel_patch_to_adatas([adata_old, adata_new])

        assert "MarkerANew" in bumped[0].var_names
        assert "MarkerA" not in bumped[0].var_names
        assert bumped[0].var.loc["MarkerANew", "uniprot_id"] == "Q9UPN0"
        assert bumped[0].var.loc["MarkerANew", "target_class"] == "new-value"
        assert "retired_field" not in bumped[0].var.columns
        assert bumped[1].var.loc["MarkerANew", "uniprot_id"] == "Q9UPN0"
        assert bumped[1].var.loc["MarkerANew", "target_class"] == "new-value"
        assert "panel_metadata" not in bumped[0].uns
        assert "panel_metadata" not in bumped[1].uns

        aligned = read_dataset_panel(dataset)
        assert aligned.df.loc["MarkerANew", "uniprot_id"] == "Q9UPN0"
        assert aligned.df.loc["MarkerANew", "target_class"] == "new-value"

        assert "positive_cells_count" in bumped[0].var.columns
        assert "positive_cells_count" not in bumped[1].var.columns
        assert (
            adata_old.var["positive_cells_count"]
            == bumped[0].var["positive_cells_count"]
        ).all()

        assert (adata_old[:, "MarkerC"].X == bumped[0][:, "MarkerC"].X).all()
        assert (adata_new[:, "MarkerC"].X == bumped[1][:, "MarkerC"].X).all()

    def test_edgelist_view_exposes_bumped_marker_ids(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
    ):
        """The session edgelist uses bumped ids before any later query."""
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
            marker_a_new_name="MarkerANew",
        )
        dataset = _build_two_sample_dataset_with_panels(
            tmp_path=tmp_path,
            edgelist_parquet_path=edgelist_parquet_path,
            panel_old=panel_old,
            panel_new=panel_new,
        )
        with dataset.view.open() as session:
            frame = session.execute_eager(
                Query("SELECT sample, marker_1, marker_2 FROM edgelist", {})
            )
        old = frame.filter(pl.col("sample") == "sample_old")
        old_ids = set(old["marker_1"].to_list() + old["marker_2"].to_list())
        assert "MarkerA" not in old_ids
        assert "MarkerANew" in old_ids

    def test_proximity_filter_uses_renamed_marker_ids(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
    ):
        """A filter on the bumped name still finds rows stored under the old id."""
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
            marker_a_new_name="MarkerANew",
        )
        old_proximity = tmp_path / "old_proximity.parquet"
        new_proximity = tmp_path / "new_proximity.parquet"
        pl.DataFrame(
            {
                "component": ["fc07dea9b679aca7", "fc07dea9b679aca7"],
                "marker_1": ["MarkerA", "MarkerB"],
                "marker_2": ["MarkerA", "MarkerC"],
            }
        ).write_parquet(old_proximity)
        pl.DataFrame(
            {
                "component": ["fc07dea9b679aca7_sample_new"],
                "marker_1": ["MarkerANew"],
                "marker_2": ["MarkerANew"],
            }
        ).write_parquet(new_proximity)
        dataset = _build_two_sample_dataset_with_panels(
            tmp_path=tmp_path,
            edgelist_parquet_path=edgelist_parquet_path,
            panel_old=panel_old,
            panel_new=panel_new,
            proximity_old=old_proximity,
            proximity_new=new_proximity,
        )

        proximity = dataset.filter(markers={"MarkerANew"}).proximity(
            add_marker_counts=False, add_logratio=False
        )
        frame = proximity.to_polars()
        pairs = set(
            zip(frame["marker_1"].to_list(), frame["marker_2"].to_list(), strict=True)
        )
        assert pairs == {("MarkerANew", "MarkerANew")}
        assert len(proximity) == 2

    def test_record_batches_use_renamed_marker_ids(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        panel: PNAAntibodyPanel,
    ):
        """The streamed edgelist uses the same marker ids as ``to_polars``."""
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
            marker_a_new_name="MarkerANew",
        )
        dataset = _build_two_sample_dataset_with_panels(
            tmp_path=tmp_path,
            edgelist_parquet_path=edgelist_parquet_path,
            panel_old=panel_old,
            panel_new=panel_new,
        )
        streamed = pl.concat(
            [pl.from_arrow(batch) for batch in dataset.edgelist().to_record_batches()],
            how="vertical",
        )
        loaded = dataset.edgelist().to_polars()
        assert streamed.sort(streamed.columns).equals(loaded.sort(loaded.columns))
        assert (
            "MarkerANew"
            in streamed["marker_1"].to_list() + streamed["marker_2"].to_list()
        )

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
        """Skip bump when version compatibility or product prerequisites are not met.

        Args:
            tmp_path: Tmp path.
            edgelist_parquet_path: Edgelist parquet path.
            panel: Panel.
            new_version: New version.
            new_product: New product.
        """
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

        not_bumped = dataset.view.apply_panel_patch_to_adatas([adata_old, adata_new])

        assert "MarkerA" in not_bumped[0].var_names
        assert not_bumped[0].var.loc["MarkerA", "uniprot_id"] == "P12345"
        assert "target_class" not in not_bumped[0].var.columns
        assert not_bumped[1].var.loc["MarkerA", "uniprot_id"] == "Q9UPN0"
        assert not_bumped[1].var.loc["MarkerA", "target_class"] == "new-version"
        assert "panel_metadata" not in not_bumped[0].uns

        assert (adata_old[:, "MarkerC"].X == not_bumped[0][:, "MarkerC"].X).all()
        assert (adata_new[:, "MarkerC"].X == not_bumped[1][:, "MarkerC"].X).all()
