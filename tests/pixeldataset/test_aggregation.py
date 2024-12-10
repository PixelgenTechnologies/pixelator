"""Tests for pixeldataset.aggregation module.

Copyright © 2024 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name

import random
import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pixeldataset import (
    read,
)
from pixelator.pixeldataset.aggregation import simple_aggregate

random.seed(42)
np.random.seed(42)


def test_simple_aggregate(setup_basic_pixel_dataset):
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
    assert re.match(r"^.{16}_sample\d+$", row["component"])
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

    assert len(result.precomputed_layouts.to_df()) == 2 * len(
        dataset_1.precomputed_layouts.to_df()
    )


def test_simple_aggregate_do_not_have_problems_with_layouts_when_working_with_files(
    setup_basic_pixel_dataset, tmp_path
):
    dataset_1, *_ = setup_basic_pixel_dataset
    tmp_data_set_path_1 = tmp_path / "dataset_1.pxl"
    tmp_data_set_path_2 = tmp_path / "dataset_2.pxl"
    dataset_1.save(tmp_data_set_path_1)
    dataset_1.save(tmp_data_set_path_2)

    datasets = list([read(tmp_data_set_path_1), read(tmp_data_set_path_2)])
    result = simple_aggregate(
        sample_names=["sample1", "sample2"],
        datasets=datasets,
    )

    assert len(result.precomputed_layouts.to_df()) == 2 * len(
        dataset_1.precomputed_layouts.to_df()
    )


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
    assert result.edgelist.shape == (0, 7)


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
