"""Snapshot tests for PNAPixelDataset outputs.

Copyright © 2025 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import pytest

from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.pixeldataset import PixelDatasetConfig, PNAPixelDataset, read
from pixelator.pna.pixeldataset.io import PixelDataViewer, PxlFile
from tests.pna.conftest import create_pxl_file


class TestReadPixelDataset:
    def test_read_pxl_file_single_file(self, pxl_file: Path):
        dataset = read(pxl_file)
        assert isinstance(dataset, PNAPixelDataset)

    def test_read_pxl_file_multiple_files(self, pxl_file: Path):
        dataset = read([pxl_file])
        assert isinstance(dataset, PNAPixelDataset)

    def test_read_pixel_file_single_file_str(self, pxl_file: Path):
        dataset = read(str(pxl_file))
        assert isinstance(dataset, PNAPixelDataset)

    def test_read_pixel_file_multiple_files_str(self, pxl_file: Path):
        dataset = read([str(pxl_file)])
        assert isinstance(dataset, PNAPixelDataset)


class TestPNAPixelDataset:
    def test_from_files(self, pxl_file):
        dataset = PNAPixelDataset.from_pxl_files(pxl_file)
        assert isinstance(dataset, PNAPixelDataset)

    def test_from_files_with_config(self, pxl_file):
        config = PixelDatasetConfig(adata_join_method="outer")
        dataset = PNAPixelDataset.from_pxl_files(pxl_file, config)
        assert isinstance(dataset, PNAPixelDataset)

    def test_from_files_with_config_and_samples(self, pxl_file):
        dataset = PNAPixelDataset.from_pxl_files({"test_sample": pxl_file})
        assert isinstance(dataset, PNAPixelDataset)

    def test_sample_names(self, pxl_dataset):
        assert pxl_dataset.sample_names() == {"test_sample"}

    def test_components(self, pxl_dataset):
        assert pxl_dataset.components() == {
            "e7d82bca9694eea7",
            "3770519d30f36d18",
            "4920229146151c29",
            "fc07dea9b679aca7",
        }

    def test_markers(self, pxl_dataset):
        assert pxl_dataset.markers() == {"MarkerA", "MarkerB", "MarkerC"}

    def test_view(self, pxl_dataset):
        assert isinstance(pxl_dataset.view, PixelDataViewer)

    def test_adata(self, pxl_dataset: PNAPixelDataset, adata_data):
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"

        adata = pxl_dataset.adata(add_clr_transform=False, add_log1p_transform=False)
        adata_assert_equal(adata, adata_data)

    def test_adata_adds_transformation_by_default(
        self,
        pxl_dataset: PNAPixelDataset,
        adata_data,  # noqa: ARG002
    ):
        adata = pxl_dataset.adata()
        assert "log1p" in adata.obsm
        assert "clr" in adata.obsm

    def test_adata_should_not_mutate_original(
        self,
        pxl_dataset: PNAPixelDataset,
        adata_data,  # noqa: ARG002
    ):
        # Making changes to the adata object should not affect the original
        # i.e. we want to avoid unexpected mutations steming from referencing
        # the same underlying adata
        adata = pxl_dataset.adata(add_clr_transform=False, add_log1p_transform=False)
        adata.layers["new_layer"] = adata.X + 1

        assert "new_layer" in adata.layers.keys()
        assert "new_layer" not in pxl_dataset.adata().layers.keys()

    def test_metadata_returns_expected_mapping(self, pxl_dataset: PNAPixelDataset):
        metadata = pxl_dataset.metadata()
        assert metadata.keys() == {"test_sample"}
        assert metadata["test_sample"]["sample_name"] == "test_sample"
        assert metadata["test_sample"]["version"] == "0.1.0"
        assert metadata["test_sample"]["panel_name"] == "custom_panel"

    def test_metadata_returns_empty_dict_when_metadata_table_is_empty(
        self,
        tmp_path: Path,
        edgelist_parquet_path: Path,
        proximity_parquet_path: Path,
        layout_parquet_path: Path,
        panel,
    ):
        target = tmp_path / "empty_metadata.pxl"
        create_pxl_file(
            target=target,
            sample_name="test_sample",
            edgelist_parquet_path=edgelist_parquet_path,
            proximity_parquet_path=proximity_parquet_path,
            layout_parquet_path=layout_parquet_path,
            panel=panel,
        )

        # Replace the metadata table with an empty one so `PNAPixelDataset.metadata()`
        # can return `{}` (without requiring any actual metadata rows).
        with duckdb.connect(target, read_only=False) as con:
            con.sql("DROP TABLE IF EXISTS metadata;")
            con.sql("CREATE TABLE metadata (value JSON);")

        pxl_file = PxlFile(target, sample_name="test_sample")
        dataset = PNAPixelDataset.from_pxl_files(pxl_file)
        assert dataset.metadata() == {}


class TestPNAPixelDatasetFilter:
    def test_filter_pixel_dataset_by_component(self, pxl_dataset):
        filtered = pxl_dataset.filter(components={"fc07dea9b679aca7"})
        assert filtered.components() == {"fc07dea9b679aca7"}

        adata = filtered.adata()
        assert len(adata.obs) == 1
        assert adata.obs.index[0] == "fc07dea9b679aca7"

        assert (
            filtered.edgelist()
            .to_polars()
            .select("component")
            .unique()
            .get_column("component")[0]
            == "fc07dea9b679aca7"
        )

        assert (
            filtered.proximity()
            .to_polars()
            .select("component")
            .unique()
            .get_column("component")[0]
            == "fc07dea9b679aca7"
        )

        assert (
            filtered.precomputed_layouts()
            .to_polars()
            .select("component")
            .unique()
            .get_column("component")[0]
            == "fc07dea9b679aca7"
        )

    def test_filter_pixel_dataset_by_component_should_raise_for_non_existent(
        self, pxl_dataset: PNAPixelDataset
    ):
        with pytest.raises(ValueError):
            pxl_dataset.filter(components={"nonexistentcomp"})

    def test_filter_pixel_dataset_by_marker_should_raise_for_non_existent(
        self, pxl_dataset: PNAPixelDataset
    ):
        with pytest.raises(ValueError):
            pxl_dataset.filter(markers={"not-a-marker"})

    def test_filter_pixel_dataset_by_sample_should_raise_for_non_existent(
        self, pxl_dataset: PNAPixelDataset
    ):
        with pytest.raises(ValueError):
            pxl_dataset.filter(samples={"not-a-sample"})

    def test_filter_pixel_dataset_by_markers(self, pxl_dataset):
        filtered = pxl_dataset.filter(markers={"MarkerA"})
        assert filtered.markers() == {"MarkerA"}

        adata = filtered.adata()
        # AnnData should be filtered by marker
        assert len(adata.var) == 1
        assert adata.var.index[0] == "MarkerA"

        edgelist = filtered.edgelist().to_polars()
        unique_markers = set(
            edgelist.select(pl.col("marker_1"))
            .unique()
            .get_column("marker_1")
            .to_list()
        ).union(
            set(
                edgelist.select(pl.col("marker_2"))
                .unique()
                .get_column("marker_2")
                .to_list()
            )
        )

        # edgelists should not be filtered by marker
        assert unique_markers == {"MarkerA", "MarkerB", "MarkerC"}

        proximity = filtered.proximity().to_polars()
        unique_markers = set(
            proximity.select(pl.col("marker_1"))
            .unique()
            .get_column("marker_1")
            .to_list()
        ).union(
            proximity.select(pl.col("marker_2"))
            .unique()
            .get_column("marker_2")
            .to_list()
        )
        # proximity data should be filtered by marker
        assert unique_markers == {"MarkerA"}

        # precomputed layouts should not be filtered by marker
        layouts = filtered.precomputed_layouts().to_polars()
        assert (
            len(set(layouts.columns).intersection({"MarkerA", "MarkerB", "MarkerC"}))
            == 3
        )

    def test_filter_pixel_dataset_with_pandas_series(self, pxl_dataset):
        assert len(pxl_dataset.components()) == 4
        adata = pxl_dataset.adata()
        filter_from_pandas = adata.obs["n_umi1"] > 3
        filtered_dataset = pxl_dataset.filter(components=filter_from_pandas)
        assert filtered_dataset.components() == {
            "3770519d30f36d18",
            "4920229146151c29",
            "fc07dea9b679aca7",
        }

    def test_filter_pixel_dataset_with_polars_series(self, pxl_dataset):
        assert len(pxl_dataset.components()) == 4
        df = pl.DataFrame(pxl_dataset.adata().obs.reset_index())
        filter_from_polars = (
            df.filter(pl.col("n_umi1") > 3).select("component").get_column("component")
        )
        filtered_dataset = pxl_dataset.filter(components=filter_from_polars)
        assert filtered_dataset.components() == {
            "3770519d30f36d18",
            "4920229146151c29",
            "fc07dea9b679aca7",
        }

    def test_filter_pixel_dataset_with_polars_dataframe(self, pxl_dataset):
        assert len(pxl_dataset.components()) == 4
        df = pl.DataFrame(pxl_dataset.adata().obs.reset_index())
        filter_from_polars = df.filter(pl.col("n_umi1") > 3).select("component")
        filtered_dataset = pxl_dataset.filter(components=filter_from_polars)
        assert filtered_dataset.components() == {
            "3770519d30f36d18",
            "4920229146151c29",
            "fc07dea9b679aca7",
        }

    def test_filter_pixel_dataset_with_polars_dataframe_raises_if_multiple_columns(
        self, pxl_dataset
    ):
        assert len(pxl_dataset.components()) == 4
        df = pl.DataFrame(pxl_dataset.adata().obs.reset_index())
        # Selecting two columns to raise an error
        filter_from_polars = df.filter(pl.col("n_umi1") > 3).select(
            "component", "n_umi1"
        )
        with pytest.raises(ValueError):
            pxl_dataset.filter(components=filter_from_polars)


@pytest.fixture(
    name="pxl_dataset_w_sample_names",
    scope="module",
    params=[
        "1-sample-starting-with-nbr",
        "sample-containing-dash",
        "sample_with_underscores",
        "✅-sample-with-emoji",
    ],
)
def pixel_dataset_with_different_sample_names_fixture(
    request,
    tmp_path_factory,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
    panel,
):
    sample_name = request.param
    target = tmp_path_factory.mktemp("data") / (sample_name + ".pxl")
    target = create_pxl_file(
        target=target,
        sample_name=sample_name,
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
        panel=panel,
    )
    return PNAPixelDataset.from_pxl_files([target]), sample_name


@pytest.fixture(
    name="pxl_file_w_sample_names",
    scope="module",
    params=[
        "1-sample-starting-with-nbr",
        "sample-containing-dash",
        "sample_with_underscores",
        "✅-sample-with-emoji",
    ],
)
def pxl_file_with_sample_names_fixture(
    request,
    tmp_path_factory,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
    panel,
):
    sample_name = request.param
    target = tmp_path_factory.mktemp("data") / (sample_name + ".pxl")
    target = create_pxl_file(
        target=target,
        sample_name=sample_name,
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
        panel=panel,
    )
    return target


class TestPixelDatasetNames:
    """Test that pixel dataset can handle sample names that contain things like dashes, that are also keywords in duckdb."""

    def test_sample_names(self, pxl_dataset_w_sample_names):
        pxl_dataset, sample_name = pxl_dataset_w_sample_names
        assert len(pxl_dataset.sample_names()) == 1
        assert pxl_dataset.sample_names() == {sample_name}

    def test_edgelist(self, pxl_dataset_w_sample_names):
        pxl_dataset, sample_name = pxl_dataset_w_sample_names
        df = pxl_dataset.edgelist().to_polars()
        actual_sample_name = df.select("sample").unique()
        assert actual_sample_name.shape[0] == 1
        assert actual_sample_name[0, 0] == sample_name

    def test_anndata(self, pxl_dataset_w_sample_names):
        pxl_dataset, sample_name = pxl_dataset_w_sample_names
        df = pxl_dataset.adata()
        actual_sample_name = df.obs["sample"].unique()
        assert actual_sample_name.shape[0] == 1
        assert actual_sample_name[0] == sample_name


@pytest.fixture(name="multi_sample_dataset", scope="module")
def dataset_fixture(
    tmp_path_factory,
    edgelist_parquet_path,
    proximity_parquet_path,
    layout_parquet_path,
    panel,
):
    tmp_path = tmp_path_factory.mktemp("multi_sample_dataset")

    # Setup: Create two samples with different components
    sample1_path = create_pxl_file(
        target=tmp_path / "sample1.pxl",
        sample_name="sample1",
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
        panel=panel,
    )

    # Rename the components in the second sample to have other component names
    sample2_edgelist_path = tmp_path / "edgelist_sample2.parquet"
    pl.scan_parquet(edgelist_parquet_path).with_columns(
        (pl.col("component").str.replace("(.)", "${1}_sample2")).alias("component")
    ).sink_parquet(sample2_edgelist_path)
    sample2_proximity_path = tmp_path / "proximity_sample2.parquet"
    pl.scan_parquet(proximity_parquet_path).with_columns(
        (pl.col("component").str.replace("(.)", "${1}_sample2")).alias("component")
    ).sink_parquet(sample2_proximity_path)
    sample2_layout_path = tmp_path / "layout_sample2.parquet"
    pl.scan_parquet(layout_parquet_path).with_columns(
        (pl.col("component").str.replace("(.)", "${1}_sample2")).alias("component")
    ).sink_parquet(sample2_layout_path)

    sample2_path = create_pxl_file(
        target=tmp_path / "sample2.pxl",
        sample_name="sample2",
        edgelist_parquet_path=sample2_edgelist_path,
        proximity_parquet_path=sample2_proximity_path,
        layout_parquet_path=sample2_layout_path,
        panel=panel,
    )
    return PNAPixelDataset.from_pxl_files([sample1_path, sample2_path])


class TestPNAPixelDatasetFilterBySample:
    def test_filter_by_sample(
        self,
        multi_sample_dataset: PNAPixelDataset,
    ):
        dataset = multi_sample_dataset

        # Verify we have both samples
        assert dataset.sample_names() == {"sample1", "sample2"}

        # 3. Filter by "sample1"
        filtered_dataset = dataset.filter(samples=["sample1"])
        assert filtered_dataset.sample_names() == {"sample1"}

        # Check adata
        adata = filtered_dataset.adata()
        unique_samples_adata = set(adata.obs["sample"].unique())
        assert "sample2" not in unique_samples_adata
        assert unique_samples_adata == {"sample1"}

        # Check proximity
        proximity = filtered_dataset.proximity()
        proximity_df = proximity.to_polars()
        unique_samples = proximity_df["sample"].unique().to_list()
        assert "sample2" not in unique_samples
        assert unique_samples == ["sample1"]

        # Check edgelist
        edgelist_df = filtered_dataset.edgelist().to_polars()
        unique_samples_edgelist = edgelist_df["sample"].unique().to_list()
        assert "sample2" not in unique_samples_edgelist
        assert unique_samples_edgelist == ["sample1"]

        # Check layouts
        layouts_df = filtered_dataset.precomputed_layouts(
            add_marker_counts=False
        ).to_polars()
        unique_samples_layouts = layouts_df["sample"].unique().to_list()
        assert "sample2" not in unique_samples_layouts
        assert unique_samples_layouts == ["sample1"]

    def test_filter_combinations(
        self,
        multi_sample_dataset: PNAPixelDataset,
    ):
        """Test filtering with combinations of sample, component, and marker."""
        dataset = multi_sample_dataset

        # Define targets
        target_sample = "sample1"
        target_component = "fc07dea9b679aca7"  # Known component in test data
        target_marker = "MarkerA"  # Known marker in test data

        # Apply combined filter
        filtered_dataset = dataset.filter(
            samples=[target_sample],
            components=[target_component],
            markers=[target_marker],
        )

        # 1. Verify Dataset metadata
        assert filtered_dataset.sample_names() == {target_sample}
        assert filtered_dataset.components() == {target_component}
        assert filtered_dataset.markers() == {target_marker}

        # 2. Verify Adata (filtered by all three)
        adata = filtered_dataset.adata()
        assert set(adata.obs["sample"].unique()) == {target_sample}
        assert set(adata.obs.index) == {target_component}
        assert set(adata.var.index) == {target_marker}
        # Check explicit data presence
        assert adata.shape[0] == 1  # 1 component
        assert adata.shape[1] == 1  # 1 marker

        # 3. Verify Proximity (filtered by all three)
        proximity_df = filtered_dataset.proximity().to_polars()
        assert set(proximity_df["sample"].unique()) == {target_sample}
        assert set(proximity_df["component"].unique()) == {target_component}
        # Markers should be filtered
        markers_in_prox = set(proximity_df["marker_1"]) | set(proximity_df["marker_2"])
        # It's possible proximity is empty if no edges exist for just MarkerA-MarkerA
        # But if edges exist, they must be MarkerA.
        if not proximity_df.is_empty():
            assert markers_in_prox == {target_marker}

        # 4. Verify Edgelist (filtered by sample & component, NOT marker)
        edgelist_df = filtered_dataset.edgelist().to_polars()
        assert set(edgelist_df["sample"].unique()) == {target_sample}
        assert set(edgelist_df["component"].unique()) == {target_component}
        # Should contain other markers too
        markers_in_edgelist = set(edgelist_df["marker_1"]) | set(
            edgelist_df["marker_2"]
        )
        # Since the test data has multiple markers for this component
        assert len(markers_in_edgelist) > 1
        assert target_marker in markers_in_edgelist

        # 5. Verify Precomputed Layouts (filtered by sample & component)
        layouts_df = filtered_dataset.precomputed_layouts(
            add_marker_counts=False
        ).to_polars()
        assert set(layouts_df["sample"].unique()) == {target_sample}
        assert set(layouts_df["component"].unique()) == {target_component}

    def test_filter_by_component_multisample(
        self,
        multi_sample_dataset: PNAPixelDataset,
    ):
        """Test filtering only by component in a multisample scenario."""

        dataset = multi_sample_dataset
        # Pick a component that exists in only one of the samples
        target_component = "fc07dea9b679aca7"

        # Filter only by component
        filtered_dataset = dataset.filter(components=[target_component])

        # 1. Dataset metadata
        # Should contain only the sample that has the target component
        assert filtered_dataset.sample_names() == {"sample1"}
        assert filtered_dataset.components() == {target_component}

        # 2. Adata
        adata = filtered_dataset.adata()
        assert set(adata.obs["sample"].unique()) == {"sample1"}
        assert set(adata.obs.index.unique()) == {target_component}
        assert len(adata) == 1  # One row for the sample with this component

        # 3. Proximity
        proximity_df = filtered_dataset.proximity().to_polars()
        assert set(proximity_df["sample"].unique()) == {"sample1"}
        assert set(proximity_df["component"].unique()) == {target_component}

        # 4. Edgelist
        edgelist_df = filtered_dataset.edgelist().to_polars()
        assert set(edgelist_df["sample"].unique()) == {"sample1"}
        assert set(edgelist_df["component"].unique()) == {target_component}

        # 5. Layouts
        layouts_df = filtered_dataset.precomputed_layouts(
            add_marker_counts=False
        ).to_polars()
        assert set(layouts_df["sample"].unique()) == {"sample1"}
        assert set(layouts_df["component"].unique()) == {target_component}


class TestPNAPixelDatasetSnapshots:
    def test_proximity(self, snapshot, pxl_dataset):
        proximity = pxl_dataset.proximity()
        assert len(proximity) == 3

        proximity_df = proximity.to_polars()
        assert isinstance(proximity_df, pl.DataFrame)
        proximity_df = proximity_df.sort("component")

        result = StringIO()
        proximity_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "proximity.csv")

    def test_precomputed_layouts(self, snapshot, pxl_dataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert len(layouts) == 34
        assert len(layouts.components) == 4

        layouts_df = layouts.to_polars()
        assert isinstance(layouts_df, pl.DataFrame)
        layouts_df = layouts_df.sort("component").select(sorted(layouts_df.columns))

        result = StringIO()
        layouts_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")


class TestPrecomputedLayoutsWrapper:
    def test_precomputed_layouts_with_filter(self, pxl_dataset: PNAPixelDataset):
        filtered = pxl_dataset.filter(components={"fc07dea9b679aca7"})
        layouts = filtered.precomputed_layouts()
        assert len(layouts) == 11
        assert len(layouts.components) == 1

        layouts_df = layouts.to_polars()
        assert len(layouts_df) == 11
        assert isinstance(layouts.to_df(), pd.DataFrame)
        assert len(layouts.to_df()) == 11

        # Check iterator contains dataframes as expected
        for comp_id, component in layouts.iterator():
            assert isinstance(comp_id, str)
            assert isinstance(component, pd.DataFrame)

    def test_precomputed_layouts_with_norm(self, pxl_dataset: PNAPixelDataset):
        result = pxl_dataset.precomputed_layouts(add_spherical_norm=True).to_polars()

        assert "x_norm" in result.columns
        assert "y_norm" in result.columns
        assert "z_norm" in result.columns
