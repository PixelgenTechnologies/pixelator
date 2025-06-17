"""Copyright © 2025 Pixelgen Technologies AB."""

from io import StringIO
from pathlib import Path

import anndata
import pandas as pd
import polars as pl
import pytest
from anndata.tests.helpers import assert_equal as adata_assert_equal
from polars.testing import assert_frame_equal

from pixelator.pna.pixeldataset import (
    PixelDatasetConfig,
    PNAPixelDataset,
    read,
)
from pixelator.pna.pixeldataset.io import PixelDataViewer, PixelFileWriter
from tests.pna.pixeldataset.conftest import create_pxl_file


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
        # Right now we actually drop all uns data
        # I'm not sure if that is the best possible behavior
        # here, but it is the default when anndata
        # concatenates multiple datasets
        adata_data.uns = {}

        adata = pxl_dataset.adata(add_clr_transform=False, add_log1p_transform=False)
        adata_assert_equal(adata, adata_data)

    def test_adata_adds_transformation_by_default(
        self, pxl_dataset: PNAPixelDataset, adata_data
    ):
        adata = pxl_dataset.adata()
        assert "log1p" in adata.obsm_keys()
        assert "clr" in adata.obsm_keys()

    def test_adata_should_not_mutate_original(
        self, pxl_dataset: PNAPixelDataset, adata_data
    ):
        # Making changes to the adata object should not affect the original
        # i.e. we want to avoid unexpected mutations steming from referencing
        # the same underlying adata
        adata = pxl_dataset.adata(add_clr_transform=False, add_log1p_transform=False)
        adata.layers["new_layer"] = adata.X + 1

        assert "new_layer" in adata.layers.keys()
        assert "new_layer" not in pxl_dataset.adata().layers.keys()

    def test_edgelist(self, pxl_dataset, edgelist_dataframe):
        edgelist = pxl_dataset.edgelist().to_polars()
        edgelist_dataframe = edgelist_dataframe.with_columns(
            sample=pl.lit("test_sample")
        )
        assert_frame_equal(
            edgelist.sort("component"), edgelist_dataframe.sort("component")
        )

    def test_proximity(self, snapshot, pxl_dataset):
        proximity = pxl_dataset.proximity()
        assert len(proximity) == 3

        proximity_df = proximity.to_polars()
        proximity_df = proximity_df.sort("component")

        result = StringIO()
        proximity_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "proximity.csv")

    def test_precomputed_layouts(self, snapshot, pxl_dataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert len(layouts) == 34
        assert len(layouts.components) == 4

        layouts_df = layouts.to_polars()
        layouts_df = layouts_df.sort("component").select(sorted(layouts_df.columns))

        result = StringIO()
        layouts_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")

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


class TestPrecomputedLayouts:
    def test_precomputed_layouts(self, snapshot, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.precomputed_layouts()
        assert len(layouts) == 34
        assert len(layouts.components) == 4

        layouts_df = layouts.to_polars()
        layouts_df = layouts_df.sort("component").select(sorted(layouts_df.columns))

        result = StringIO()
        layouts_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")

        assert isinstance(layouts.to_df(), pd.DataFrame)

        # Check iterator contains dataframes as expected
        for comp_id, component in layouts.iterator():
            assert isinstance(comp_id, str)
            assert isinstance(component, pd.DataFrame)

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
):
    sample_name = request.param
    target = tmp_path_factory.mktemp("data") / (sample_name + ".pxl")
    target = create_pxl_file(
        target=target,
        sample_name=sample_name,
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
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
):
    sample_name = request.param
    target = tmp_path_factory.mktemp("data") / (sample_name + ".pxl")
    target = create_pxl_file(
        target=target,
        sample_name=sample_name,
        edgelist_parquet_path=edgelist_parquet_path,
        proximity_parquet_path=proximity_parquet_path,
        layout_parquet_path=layout_parquet_path,
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


def test_rewriting_anndata(pxl_file_w_sample_names):
    pxl_dataset = pxl_file_w_sample_names
    pxl = PNAPixelDataset.from_pxl_files(pxl_dataset)
    adata = pxl.adata()

    with PixelFileWriter(pxl_dataset) as writer:
        writer.write_adata(adata)
