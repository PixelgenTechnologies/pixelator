"""Tests for pixeldataset.py module.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
# pylint: disable=redefined-outer-name

import logging
import random
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pixelator.config import AntibodyPanel
from pixelator.graph import Graph, write_recovered_components
from pixelator.pixeldataset import (
    FileBasedPixelDatasetBackend,
    ObjectBasedPixelDatasetBackend,
    PixelDataset,
    PixelFileCSVFormatSpec,
    PixelFileFormatSpec,
    PixelFileParquetFormatSpec,
    read,
)
from pixelator.pixeldataset.aggregation import simple_aggregate
from pixelator.pixeldataset.utils import (
    _enforce_edgelist_types,
    antibody_metrics,
    component_antibody_counts,
    edgelist_to_anndata,
    read_anndata,
    write_anndata,
)
from pixelator.statistics import (
    clr_transformation,
    denoise,
    log1p_transformation,
    rel_normalization,
)

random.seed(42)
np.random.seed(42)


def test_pixel_file_format_spec_parquet(pixel_dataset_file):
    """test_pixel_file_format_spec_parquet."""
    res = PixelFileFormatSpec.guess_file_format(pixel_dataset_file)
    assert isinstance(res, PixelFileParquetFormatSpec)


def test_pixel_file_format_spec_csv(setup_basic_pixel_dataset, tmp_path):
    """test_pixel_file_format_spec_csv."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target), file_format="csv")
    res = PixelFileFormatSpec.guess_file_format(file_target)
    assert isinstance(res, PixelFileCSVFormatSpec)


def test_pixel_file_parquet_format_spec_can_save(setup_basic_pixel_dataset, tmp_path):
    """test_pixel_file_parquet_format_spec_can_save."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    assert not file_target.is_file()
    PixelFileParquetFormatSpec().save(dataset, str(file_target))
    assert file_target.is_file()


def test_pixel_file_parquet_no_index_in_parquet_files(tmp_path):
    # Checking EXE-1184
    # We do not want an index added to the parquet file, regardless of if they
    # read with pandas or polars
    file_target = tmp_path / "df.parquet"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]}, index=["X", "Y", "Z"])
    PixelFileParquetFormatSpec().serialize_dataframe(df, path=file_target)
    assert file_target.is_file()
    res1 = pd.read_parquet(file_target)
    assert set(res1.columns) == {"A", "B"}
    res2 = pl.scan_parquet(file_target)
    assert set(res2.columns) == {"A", "B"}
    res3 = pl.read_parquet(file_target)
    assert set(res3.columns) == {"A", "B"}


def test_pixel_file_csv_format_spec_can_save(setup_basic_pixel_dataset, tmp_path):
    """test_pixel_file_csv_format_spec_can_save."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    assert not file_target.is_file()
    PixelFileCSVFormatSpec().save(dataset, str(file_target))
    assert file_target.is_file()


def test_pixeldataset(setup_basic_pixel_dataset):
    """test_pixeldataset."""
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset

    assert_frame_equal(
        edgelist,
        dataset.edgelist,
    )

    assert_frame_equal(
        edgelist,
        _enforce_edgelist_types(dataset.edgelist_lazy.collect().to_pandas()),
    )

    assert_frame_equal(
        adata.to_df(),
        dataset.adata.to_df(),
    )

    assert metadata == dataset.metadata

    assert_frame_equal(
        polarization_scores,
        dataset.polarization,
    )

    assert_frame_equal(
        colocalization_scores,
        dataset.colocalization,
    )


def test_pixeldataset_save(setup_basic_pixel_dataset, tmp_path):
    """test_pixeldataset_save."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))

    assert file_target.is_file


def test_pixeldataset_read(setup_basic_pixel_dataset, tmp_path):
    """test_pixeldataset_read."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))
    assert isinstance(read(str(file_target)), PixelDataset)


def test_pixeldataset_from_file_parquet(setup_basic_pixel_dataset, tmp_path):
    """test_pixeldataset_from_file_parquet."""
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))
    dataset_new = PixelDataset.from_file(str(file_target))

    assert_frame_equal(edgelist, dataset_new.edgelist)
    assert_frame_equal(
        edgelist,
        # Note that we need to enforce the types manually here for this to work,
        # this is expected since the lazy edgelist is an advanced feature
        # where the user will need to manage the required datatypes themselves
        # as needed.
        _enforce_edgelist_types(dataset_new.edgelist_lazy.collect().to_pandas()),
    )

    assert_frame_equal(
        adata.to_df(),
        dataset_new.adata.to_df(),
    )

    assert metadata == dataset_new.metadata

    assert_frame_equal(polarization_scores, dataset_new.polarization)

    assert_frame_equal(colocalization_scores, dataset_new.colocalization)


def test_pixeldataset_from_file_parquet_backward_comp_with_pyarrow_types(
    setup_basic_pixel_dataset, tmp_path
):
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset

    # When reading the old file-format we will fall back to the
    # polars file reader
    with mock.patch(
        "pixelator.pixeldataset.pd.read_parquet",
        side_effect=ValueError(),
    ):
        file_target = tmp_path / "dataset.pxl"
        dataset.save(str(file_target))
        dataset_new = PixelDataset.from_file(str(file_target))

        assert_frame_equal(edgelist, dataset_new.edgelist)
        assert_frame_equal(
            edgelist,
            # Note that we need to enforce the types manually here for this to work,
            # this is expected since the lazy edgelist is an advanced feature
            # where the user will need to manage the required datatypes themselves
            # as needed.
            _enforce_edgelist_types(dataset_new.edgelist_lazy.collect().to_pandas()),
        )

        assert_frame_equal(
            adata.to_df(),
            dataset_new.adata.to_df(),
        )

        assert metadata == dataset_new.metadata

        assert_frame_equal(polarization_scores, dataset_new.polarization)

        assert_frame_equal(colocalization_scores, dataset_new.colocalization)


def test_pixeldataset_can_save_and_load_with_empty_edgelist(
    setup_basic_pixel_dataset, tmp_path
):
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.edgelist = pd.DataFrame()
    dataset.save(str(file_target))
    dataset_new = PixelDataset.from_file(str(file_target))
    assert dataset_new.edgelist.shape == (0, 9)
    assert dataset_new.edgelist.columns.tolist() == [
        "count",
        "umi_unique_count",
        "upi_unique_count",
        "upia",
        "upib",
        "umi",
        "marker",
        "sequence",
        "component",
    ]


def test_pixeldataset_graph(setup_basic_pixel_dataset):
    dataset, *_ = setup_basic_pixel_dataset
    full_graph = dataset.graph()
    assert isinstance(full_graph, Graph)
    assert len(full_graph.connected_components()) == 5


def test_pixeldataset_graph_raises_when_component_not_found(setup_basic_pixel_dataset):
    dataset, *_ = setup_basic_pixel_dataset
    with pytest.raises(KeyError):
        dataset.graph("not-a-component")


def test_pixeldataset_graph_finds_component(setup_basic_pixel_dataset):
    dataset, *_ = setup_basic_pixel_dataset
    component_graph = dataset.graph("PXLCMP0000000")
    assert isinstance(component_graph, Graph)
    assert len(component_graph.connected_components()) == 1


def test_pixeldataset_from_file_csv(setup_basic_pixel_dataset, tmp_path):
    """test_pixeldataset_from_file_csv."""
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target), file_format="csv")
    dataset_new = PixelDataset.from_file(str(file_target))

    assert_frame_equal(edgelist, dataset_new.edgelist)
    with pytest.raises(NotImplementedError):
        dataset_new.edgelist_lazy

    assert_frame_equal(
        adata.to_df(),
        dataset_new.adata.to_df(),
    )

    assert metadata == dataset_new.metadata

    assert_frame_equal(polarization_scores, dataset_new.polarization)

    assert_frame_equal(colocalization_scores, dataset_new.colocalization)


def test_pixeldataset_repr(setup_basic_pixel_dataset):
    """test_pixeldataset_repr."""
    dataset, *_ = setup_basic_pixel_dataset

    assert repr(dataset).splitlines() == [
        "Pixel dataset contains:",
        "  AnnData with 5 obs and 22 vars",
        "  Edge list with 30000 edges",
        "  Polarization scores with 6 elements",
        "  Colocalization scores with 5 elements",
        "  Metadata:",
        "    A: 1",
        "    B: 2",
        "    file_format_version: 1",
    ]


def assert_backend_can_set_values(pixel_dataset_backend):
    """assert_backend_can_set_values."""
    assert pixel_dataset_backend.adata
    pixel_dataset_backend.adata = None
    assert not pixel_dataset_backend.adata

    assert not pixel_dataset_backend.edgelist.empty
    pixel_dataset_backend.edgelist = None
    assert not pixel_dataset_backend.edgelist

    assert not pixel_dataset_backend.polarization.empty
    pixel_dataset_backend.polarization = None
    assert not pixel_dataset_backend.polarization

    assert not pixel_dataset_backend.colocalization.empty
    pixel_dataset_backend.colocalization = None
    assert not pixel_dataset_backend.colocalization

    assert pixel_dataset_backend.metadata
    pixel_dataset_backend.metadata = None
    assert not pixel_dataset_backend.metadata


def test_file_based_pixel_dataset_backend_set_attrs(pixel_dataset_file):
    """test_file_based_pixel_dataset_backend_set_attrs."""
    pixel_dataset_backend = FileBasedPixelDatasetBackend(pixel_dataset_file)
    assert_backend_can_set_values(pixel_dataset_backend)


def test_object_based_pixel_dataset_backend_set_attrs(setup_basic_pixel_dataset):
    """test_object_based_pixel_dataset_backend_set_attrs."""
    (
        _,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset
    pixel_dataset_backend = ObjectBasedPixelDatasetBackend(
        adata=adata,
        edgelist=edgelist,
        polarization=polarization_scores,
        colocalization=colocalization_scores,
        metadata=metadata,
    )
    assert_backend_can_set_values(pixel_dataset_backend)


def test_write_recovered_components(tmp_path: Path):
    """test_write_recovered_components."""
    file_target = tmp_path / "components_recovered.csv"
    write_recovered_components(
        {"PXLCMP0000000": ["PXLCMP0000000", "PXLCMP0000001"]},
        filename=file_target,
    )

    result = pd.read_csv(file_target)
    assert list(result.columns) == ["cell_id", "recovered_from"]
    assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "cell_id": ["PXLCMP0000000", "PXLCMP0000001"],
                "recovered_from": ["PXLCMP0000000", "PXLCMP0000000"],
            }
        ),
    )


def test_antibody_metrics(full_graph_edgelist: pd.DataFrame):
    """test_antibody_metrics."""
    assert_frame_equal(
        antibody_metrics(edgelist=full_graph_edgelist),
        pd.DataFrame(
            data={
                "antibody_count": [1250, 1250],
                "components": [1, 1],
                "antibody_pct": [0.5, 0.5],
            },
            index=pd.CategoricalIndex(
                ["A", "B"],
                name="marker",
            ),
        ),
    )


def test_antibody_counts(full_graph_edgelist: pd.DataFrame):
    """test_antibody_counts."""
    counts = component_antibody_counts(edgelist=full_graph_edgelist)
    assert_array_equal(
        counts.to_numpy(),
        np.array([[1250, 1250]]),
    )
    assert sorted(counts.columns) == sorted(["A", "B"])


def test_adata_creation(edgelist: pd.DataFrame, panel: AntibodyPanel):
    """test_adata_creation."""
    adata = edgelist_to_anndata(edgelist=edgelist, panel=panel)
    assert adata.n_vars == panel.size
    assert adata.n_obs == edgelist["component"].nunique()
    assert sorted(adata.var) == sorted(
        [
            "antibody_count",
            "components",
            "antibody_pct",
            "control",
            "nuclear",
        ]
    )
    assert sorted(adata.obs) == sorted(
        [
            "vertices",
            "edges",
            "antibodies",
            "upia",
            "upib",
            "umi",
            "reads",
            "mean_reads_per_molecule",
            "median_reads_per_molecule",
            "mean_upia_degree",
            "median_upia_degree",
            "mean_umi_per_upia",
            "median_umi_per_upia",
            "upia_per_upib",
        ]
    )
    assert "normalized_rel" in adata.obsm
    assert "clr" in adata.obsm
    assert "denoised" in adata.obsm
    assert "log1p" in adata.obsm


def test_read_write_anndata(adata: AnnData):
    """test_read_write_anndata."""
    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "example.h5ad"
        write_anndata(adata, output_path)
        assert output_path.is_file()
        adata2 = read_anndata(str(output_path))
        assert_frame_equal(adata.to_df(), adata2.to_df())
        assert_frame_equal(adata.obs, adata2.obs)
        assert_frame_equal(adata.var, adata2.var)
        assert_array_equal(
            adata.obsm["normalized_rel"],
            adata2.obsm["normalized_rel"],
        )
        assert_array_equal(
            adata.obsm["clr"],
            adata2.obsm["clr"],
        )
        assert_array_equal(
            adata.obsm["denoised"],
            adata2.obsm["denoised"],
        )
        assert_array_equal(
            adata.obsm["log1p"],
            adata2.obsm["log1p"],
        )


def test_edgelist_to_anndata_missing_markers(
    panel: AntibodyPanel, edgelist: pd.DataFrame, caplog
):
    """test_edgelist_to_anndata_missing_markers."""
    with caplog.at_level(logging.WARN):
        edgelist_to_anndata(edgelist, panel)

    assert "The given 'panel' is missing markers" in caplog.text


def test_edgelist_to_anndata(
    adata: AnnData, panel: AntibodyPanel, edgelist: pd.DataFrame
):
    """test_edgelist_to_anndata."""
    antibodies = panel.markers
    counts_df = component_antibody_counts(edgelist=edgelist)
    counts_df = counts_df.reindex(columns=antibodies, fill_value=0)
    assert_array_equal(adata.X, counts_df.to_numpy())

    counts_df_norm = rel_normalization(counts_df, axis=1)
    assert_array_equal(
        adata.obsm["normalized_rel"],
        counts_df_norm.to_numpy(),
    )

    counts_df_clr = clr_transformation(counts_df, axis=1)
    assert_array_equal(
        adata.obsm["clr"],
        counts_df_clr.to_numpy(),
    )

    counts_df_denoised = denoise(
        counts_df_clr, quantile=1.0, antibody_control=panel.markers_control, axis=1
    )
    assert_array_equal(
        adata.obsm["denoised"],
        counts_df_denoised.to_numpy(),
    )

    counts_df_log1p = log1p_transformation(counts_df)
    assert_array_equal(
        adata.obsm["log1p"],
        counts_df_log1p.to_numpy(),
    )

    assert set(adata.obs_names) == set(edgelist["component"].unique())


def test_simple_aggregate(setup_basic_pixel_dataset):
    """test_simple_aggregate."""
    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2 = dataset_1.copy()

    result = simple_aggregate(
        sample_names=["sample1", "sample2"], datasets=[dataset_1, dataset_2]
    )

    assert not (
        set(result.adata.obs.index.unique()).difference(
            result.edgelist["component"].unique()
        )
        and set(result.adata.obs.index.unique()).difference(
            set(result.colocalization["component"].unique())
        )
        and set(result.adata.obs.index.unique()).difference(
            set(result.polarization["component"].unique())
        )
    )

    assert len(result.edgelist) == 2 * len(dataset_1.edgelist)
    assert "sample" in result.edgelist.columns
    row = result.edgelist.iloc[0]
    assert re.match(r"PXLCMP(\d+)_sample\d", row["component"])
    assert result.edgelist["sample"].dtype == pd.CategoricalDtype(
        categories=["sample1", "sample2"], ordered=False
    )
    assert isinstance(result.edgelist["component"].dtype, pd.CategoricalDtype)

    assert len(result.adata) == 2 * len(dataset_1.adata)
    assert len(result.adata.var) == len(dataset_1.adata.var)
    assert result.adata.var_keys() == dataset_1.adata.var_keys()
    expected_var = pd.DataFrame.from_dict(
        {
            "antibody_count": {
                "CD45": 18150,
                "CD3": 9218,
                "CD19": 0,
                "IgG1ctrl": 0,
                "CD20": 4816,
                "CD69": 0,
                "HLA-DR": 0,
                "CD8": 0,
                "CD14": 0,
                "IsoT_ctrl": 13940,
                "CD45RA": 4534,
                "CD45RO": 0,
                "CD62L": 0,
                "CD82": 0,
                "CD7": 0,
                "CD70": 0,
                "CD72": 4730,
                "CD162": 0,
                "CD26": 0,
                "CD63": 0,
                "CD4": 0,
                "hashtag": 4612,
            },
            "components": {
                "CD45": 10,
                "CD3": 10,
                "CD19": 0,
                "IgG1ctrl": 0,
                "CD20": 10,
                "CD69": 0,
                "HLA-DR": 0,
                "CD8": 0,
                "CD14": 0,
                "IsoT_ctrl": 10,
                "CD45RA": 10,
                "CD45RO": 0,
                "CD62L": 0,
                "CD82": 0,
                "CD7": 0,
                "CD70": 0,
                "CD72": 10,
                "CD162": 0,
                "CD26": 0,
                "CD63": 0,
                "CD4": 0,
                "hashtag": 10,
            },
            "antibody_pct": {
                "CD45": 0.3025,
                "CD3": 0.15363333333333334,
                "CD19": 0.0,
                "IgG1ctrl": 0.0,
                "CD20": 0.08026666666666667,
                "CD69": 0.0,
                "HLA-DR": 0.0,
                "CD8": 0.0,
                "CD14": 0.0,
                "IsoT_ctrl": 0.23233333333333334,
                "CD45RA": 0.07556666666666667,
                "CD45RO": 0.0,
                "CD62L": 0.0,
                "CD82": 0.0,
                "CD7": 0.0,
                "CD70": 0.0,
                "CD72": 0.07883333333333334,
                "CD162": 0.0,
                "CD26": 0.0,
                "CD63": 0.0,
                "CD4": 0.0,
                "hashtag": 0.07686666666666667,
            },
            "nuclear": {
                "CD45": False,
                "CD3": False,
                "CD19": False,
                "IgG1ctrl": False,
                "CD20": False,
                "CD69": False,
                "HLA-DR": False,
                "CD8": False,
                "CD14": False,
                "IsoT_ctrl": False,
                "CD45RA": False,
                "CD45RO": False,
                "CD62L": False,
                "CD82": False,
                "CD7": False,
                "CD70": False,
                "CD72": False,
                "CD162": False,
                "CD26": False,
                "CD63": False,
                "CD4": False,
                "hashtag": False,
            },
            "control": {
                "CD45": False,
                "CD3": False,
                "CD19": False,
                "IgG1ctrl": True,
                "CD20": False,
                "CD69": False,
                "HLA-DR": False,
                "CD8": False,
                "CD14": False,
                "IsoT_ctrl": True,
                "CD45RA": False,
                "CD45RO": False,
                "CD62L": False,
                "CD82": False,
                "CD7": False,
                "CD70": False,
                "CD72": False,
                "CD162": False,
                "CD26": False,
                "CD63": False,
                "CD4": False,
                "hashtag": False,
            },
        }
    )
    expected_var.index.name = "marker"
    assert_frame_equal(result.adata.var, expected_var)

    assert len(result.polarization) == 2 * len(dataset_1.polarization)
    assert len(result.colocalization) == 2 * len(dataset_1.colocalization)

    assert list(result.metadata.keys()) == ["samples"]
    assert result.metadata["samples"]["sample1"] == dataset_1.metadata


def test_simple_aggregate_ignore_edgelist(setup_basic_pixel_dataset):
    """test_simple_aggregate."""
    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2 = dataset_1.copy()

    result = simple_aggregate(
        sample_names=["sample1", "sample2"],
        datasets=[dataset_1, dataset_2],
        ignore_edgelists=True,
    )

    # We want an empty edgelist, but wit all the correct columns
    assert result.edgelist.shape == (0, 9)


def test_filter_should_return_proper_typed_edgelist_data(setup_basic_pixel_dataset):
    # Test to check for bug EXE-1177
    # This bug was caused by filtering returning an incorrectly typed
    # edgelist, which in turn caused getting the graph to fail
    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2 = dataset_1.copy()

    aggregated_data = simple_aggregate(
        sample_names=["sample1", "sample2"], datasets=[dataset_1, dataset_2]
    )

    result = aggregated_data.filter(components=aggregated_data.adata.obs.index[:2])
    assert isinstance(result.edgelist["component"].dtype, pd.CategoricalDtype)
    # Running graph here to make sure it does not raise an exception
    result.graph(result.adata.obs.index[0])


def test_on_aggregation_not_passing_unique_sample_names_should_raise(
    tmp_path,
    setup_basic_pixel_dataset,
):
    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2 = dataset_1.copy()

    file_target_1 = tmp_path / "dataset_1.pxl"
    dataset_1.save(str(file_target_1))
    file_target_2 = tmp_path / "dataset_2.pxl"
    dataset_2.save(str(file_target_2))

    with pytest.raises(AssertionError):
        simple_aggregate(
            sample_names=["sample1", "sample1"],
            datasets=[
                read(file_target_1),
                read(file_target_2),
            ],
        )


def test_aggregation_all_samples_show_up(
    tmp_path,
    setup_basic_pixel_dataset,
):
    # There used to be a bug (EXE-1186) where only the first two samples
    # were actually aggregated. This test is here to catch that potential
    # problem in the future.

    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2 = dataset_1.copy()
    dataset_3 = dataset_1.copy()
    dataset_4 = dataset_1.copy()

    file_target_1 = tmp_path / "dataset_1.pxl"
    dataset_1.save(str(file_target_1))
    file_target_2 = tmp_path / "dataset_2.pxl"
    dataset_2.save(str(file_target_2))
    file_target_3 = tmp_path / "dataset_3.pxl"
    dataset_3.save(str(file_target_3))
    file_target_4 = tmp_path / "dataset_4.pxl"
    dataset_4.save(str(file_target_4))

    result = simple_aggregate(
        sample_names=["sample1", "sample2", "sample3", "sample4"],
        datasets=[
            read(file_target_1),
            read(file_target_2),
            read(file_target_3),
            read(file_target_4),
        ],
    )
    assert set(result.edgelist["sample"].unique()) == {
        "sample1",
        "sample2",
        "sample3",
        "sample4",
    }
    assert set(result.polarization["sample"].unique()) == {
        "sample1",
        "sample2",
        "sample3",
        "sample4",
    }
    assert set(result.colocalization["sample"].unique()) == {
        "sample1",
        "sample2",
        "sample3",
        "sample4",
    }
    assert set(result.adata.obs["sample"].unique()) == {
        "sample1",
        "sample2",
        "sample3",
        "sample4",
    }


def test_lazy_edgelist_should_warn_and_rm_on_index_column(setup_basic_pixel_dataset):
    # Checking EXE-1184
    # Pixelator 0.13.0-0.15.2 stored an index column in the parquet files
    # which showed up as a column when reading the lazy frames. This
    # caused graph building to fail when working on concatenated
    # pixeldatasets, since then this column would be propagated to the
    # edgelist - this broke the Graph construction since
    # it assumes that the first two columns should be the vertices
    dataset, *_ = setup_basic_pixel_dataset
    dataset.edgelist["index"] = pd.Series(range(len(dataset.edgelist)))

    with pytest.warns(UserWarning):
        result = dataset.edgelist_lazy
        assert set(result.columns) == {
            "sequence",
            "upib",
            "upia",
            "upi_unique_count",
            "umi",
            "umi_unique_count",
            "component",
            "count",
            "marker",
        }


def test_copy(setup_basic_pixel_dataset):
    """test_copy."""
    dataset_1, *_ = setup_basic_pixel_dataset
    dataset_2_no_copy = dataset_1
    assert dataset_1 == dataset_2_no_copy
    dataset_2_copy = dataset_1.copy()
    assert not dataset_1 == dataset_2_copy


def _assert_has_components(dataset, comp_set):
    assert set(dataset.adata.obs.index) == comp_set
    assert set(dataset.edgelist["component"]) == comp_set
    assert set(dataset.polarization["component"]) == comp_set
    assert set(dataset.colocalization["component"]) == comp_set


def test_filter_by_component(setup_basic_pixel_dataset):
    """test_filter by component."""
    dataset_1, *_ = setup_basic_pixel_dataset

    # Assert before filter contains all data
    _assert_has_components(
        dataset_1,
        {
            "PXLCMP0000003",
            "PXLCMP0000004",
            "PXLCMP0000000",
            "PXLCMP0000002",
            "PXLCMP0000001",
        },
    )

    # Try filtering
    result = dataset_1.filter(
        components=[
            "PXLCMP0000000",
        ]
    )

    # Original should not have changed
    _assert_has_components(
        dataset_1,
        {
            "PXLCMP0000003",
            "PXLCMP0000004",
            "PXLCMP0000000",
            "PXLCMP0000002",
            "PXLCMP0000001",
        },
    )

    _assert_has_components(
        result,
        {
            "PXLCMP0000000",
        },
    )


def test_filter_by_marker(setup_basic_pixel_dataset):
    """test_filter by marker."""
    dataset_1, *_ = setup_basic_pixel_dataset

    original_adata_markers = set(dataset_1.adata.var.index)
    original_edgelist_markers = set(dataset_1.edgelist["marker"])
    original_pol_markers = set(dataset_1.polarization["marker"])
    original_coloc_markers = set(dataset_1.colocalization["marker_1"]).union(
        set(dataset_1.colocalization["marker_2"])
    )

    # Try filtering
    result = dataset_1.filter(markers=["CD3", "CD45"])

    # Original should not have changed
    assert set(dataset_1.adata.var.index) == original_adata_markers
    assert set(dataset_1.edgelist["marker"]) == original_edgelist_markers
    assert set(dataset_1.polarization["marker"]) == original_pol_markers
    assert (
        set(dataset_1.colocalization["marker_1"]).union(
            set(dataset_1.colocalization["marker_2"])
        )
        == original_coloc_markers
    )

    # The results should only contain CD3 and CD45
    assert set(result.adata.var.index) == {"CD3", "CD45"}
    assert set(result.polarization["marker"]) == {"CD3", "CD45"}
    # In the colocalization frame we keep all it's interaction partners
    assert set(result.colocalization["marker_1"]).union(
        set(result.colocalization["marker_2"])
    ) == {"CD3", "CD45"}
    assert_array_equal(
        result.adata.obs["antibodies"], np.repeat(2, len(result.adata.obs))
    )
    # The edgelist should contain all the original markers since it should
    # not be filtered
    assert set(result.edgelist["marker"]) == original_edgelist_markers


def test_filter_by_component_and_marker(setup_basic_pixel_dataset):
    """test_filter by component and marker."""
    dataset_1, *_ = setup_basic_pixel_dataset

    original_adata_markers = set(dataset_1.adata.var.index)
    original_edgelist_markers = set(dataset_1.edgelist["marker"])
    original_pol_markers = set(dataset_1.polarization["marker"])
    original_coloc_markers = set(dataset_1.colocalization["marker_1"]).union(
        set(dataset_1.colocalization["marker_2"])
    )

    # Try filtering
    result = dataset_1.filter(components=["PXLCMP0000000"], markers=["CD3", "CD45"])

    # Original should not have changed
    assert set(dataset_1.adata.var.index) == original_adata_markers
    assert set(dataset_1.edgelist["marker"]) == original_edgelist_markers
    assert set(dataset_1.polarization["marker"]) == original_pol_markers
    assert (
        set(dataset_1.colocalization["marker_1"]).union(
            set(dataset_1.colocalization["marker_2"])
        )
        == original_coloc_markers
    )

    _assert_has_components(
        result,
        {
            "PXLCMP0000000",
        },
    )

    # The results should only contain CD3 and CD45
    assert set(result.adata.var.index) == {"CD3", "CD45"}
    assert set(result.polarization["marker"]) == {"CD3", "CD45"}
    assert set(result.colocalization["marker_1"]).union(
        set(result.colocalization["marker_2"])
    ) == {"CD3", "CD45"}
    assert_array_equal(
        result.adata.obs["antibodies"], np.repeat(2, len(result.adata.obs))
    )
    # The edgelist should contain all the original markers since it should
    # not be filtered
    assert set(result.edgelist["marker"]) == original_edgelist_markers


def test__enforce_edgelist_types():
    # Typically we don't test private functions, but in this case
    # I wanted to make sure that this function works as expected
    # without exposing it as a public function (since the user shouldn't
    # have to worry about it). Importing it here in tests, will have to
    # be seen as a pragmatic solution to this problem.

    data = pd.DataFrame(
        {
            "count": [1, 3, 1],
            "umi_unique_count": [2, 4, 2],
            "upi_unique_count": [3, 6, 3],
            "upia": ["AAA", "AAA", "ATT"],
            "upib": ["GGG", "GGG", "GCC"],
            "umi": ["TAT", "ATA", "TTC"],
            "marker": ["CD20", "CD20", "CD3"],
            "sequence": ["AAA", "AAA", "TTT"],
            "component": ["PXL000001", "PXL000001", "PXL000001"],
        }
    )

    result = _enforce_edgelist_types(data)
    expected = {
        "count": "uint16",
        "umi_unique_count": "uint16",
        "upi_unique_count": "uint16",
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
        "component": "category",
    }
    assert result.dtypes.to_dict() == expected
