"""Tests for pixeldataset module.

Copyright © 2022 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name

import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from pixelator.graph import Graph
from pixelator.pixeldataset import (
    PixelDataset,
    read,
)
from pixelator.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pixeldataset.utils import (
    _enforce_edgelist_types,
)

random.seed(42)
np.random.seed(42)


def test_pixeldataset(setup_basic_pixel_dataset):
    """test_pixeldataset."""
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
        precomputed_layouts,
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

    assert_array_equal(precomputed_layouts.to_df(), dataset.precomputed_layouts.to_df())


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
        precomputed_layouts,
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

    assert_frame_equal(
        precomputed_layouts.to_df()
        .sort_index(axis=1)
        .sort_values(by=precomputed_layouts.to_df().columns.to_list())
        .reset_index(drop=True),
        dataset_new.precomputed_layouts.to_df()
        .sort_index(axis=1)
        .sort_values(by=precomputed_layouts.to_df().columns.to_list())
        .reset_index(drop=True),
    )


def test_pixeldataset_can_save_and_load_with_empty_edgelist(
    setup_basic_pixel_dataset, tmp_path
):
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.edgelist = pd.DataFrame()
    dataset.save(str(file_target))
    dataset_new = PixelDataset.from_file(str(file_target))
    assert dataset_new.edgelist.shape == (0, 7)
    assert dataset_new.edgelist.columns.tolist() == [
        "count",
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
    component_graph = dataset.graph("2ac2ca983a4b82dd")
    assert isinstance(component_graph, Graph)
    assert len(component_graph.connected_components()) == 1


def test_pixeldataset_precomputed_layouts(setup_basic_pixel_dataset):
    dataset, *_ = setup_basic_pixel_dataset

    precomputed_layouts = dataset.precomputed_layouts
    assert isinstance(precomputed_layouts, PreComputedLayouts)


def test_pixeldataset_from_file_csv(setup_basic_pixel_dataset, tmp_path):
    """test_pixeldataset_from_file_csv."""
    (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
        precomputed_layouts,
    ) = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    # Writing pre-computed layouts is not supported for csv files
    dataset.precomputed_layouts = None
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

    # Layouts are not supported by csv backed files
    with pytest.raises(NotImplementedError):
        assert_frame_equal(
            precomputed_layouts.to_df(), dataset_new.precomputed_layouts.to_df()
        )


def test_pixeldataset_repr(setup_basic_pixel_dataset):
    """test_pixeldataset_repr."""
    dataset, *_ = setup_basic_pixel_dataset

    assert repr(dataset).splitlines() == [
        "Pixel dataset contains:",
        "  AnnData with 5 obs and 22 vars",
        "  Edge list with 30000 edges",
        "  Polarization scores with 6 elements",
        "  Colocalization scores with 5 elements",
        "  Contains precomputed layouts",
        "  Metadata:",
        "    A: 1",
        "    B: 2",
        "    file_format_version: 1",
    ]


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
            "umi",
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
    assert set(dataset.precomputed_layouts.to_df()["component"]) == comp_set


def test_filter_by_component(setup_basic_pixel_dataset):
    """test_filter by component."""
    dataset_1, *_ = setup_basic_pixel_dataset

    # Assert before filter contains all data
    _assert_has_components(
        dataset_1,
        {
            "701ec72d3bda62d5",
            "bec92437d668cfa1",
            "2ac2ca983a4b82dd",
            "6ed5d4e4cfe588bd",
            "ce2709afa8ebd1c9",
        },
    )

    # Try filtering
    result = dataset_1.filter(
        components=[
            "701ec72d3bda62d5",
        ]
    )

    # Original should not have changed
    _assert_has_components(
        dataset_1,
        {
            "701ec72d3bda62d5",
            "bec92437d668cfa1",
            "2ac2ca983a4b82dd",
            "6ed5d4e4cfe588bd",
            "ce2709afa8ebd1c9",
        },
    )

    _assert_has_components(
        result,
        {
            "701ec72d3bda62d5",
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
    original_precomputed_layouts_columns = set(
        dataset_1.precomputed_layouts.to_df().columns
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
    # The edgelist/precomputed layouts should contain all the original markers since it should
    # not be filtered
    assert set(result.edgelist["marker"]) == original_edgelist_markers
    assert set(result.precomputed_layouts.to_df().columns).issuperset(
        original_precomputed_layouts_columns
    )


def test_filter_by_component_and_marker(setup_basic_pixel_dataset):
    """test_filter by component and marker."""
    dataset_1, *_ = setup_basic_pixel_dataset

    original_adata_markers = set(dataset_1.adata.var.index)
    original_edgelist_markers = set(dataset_1.edgelist["marker"])
    original_pol_markers = set(dataset_1.polarization["marker"])
    original_coloc_markers = set(dataset_1.colocalization["marker_1"]).union(
        set(dataset_1.colocalization["marker_2"])
    )
    original_precomputed_layouts_columns = set(
        dataset_1.precomputed_layouts.to_df().columns
    )

    # Try filtering
    result = dataset_1.filter(components=["2ac2ca983a4b82dd"], markers=["CD3", "CD45"])

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
            "2ac2ca983a4b82dd",
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
    # The edgelist/precomputed layouts should contain all the original markers since it should
    # not be filtered
    assert set(result.edgelist["marker"]) == original_edgelist_markers
    assert set(result.precomputed_layouts.to_df().columns).issuperset(
        original_precomputed_layouts_columns
    )


def test__enforce_edgelist_types():
    # Typically we don't test private functions, but in this case
    # I wanted to make sure that this function works as expected
    # without exposing it as a public function (since the user shouldn't
    # have to worry about it). Importing it here in tests, will have to
    # be seen as a pragmatic solution to this problem.

    data = pd.DataFrame(
        {
            "count": [1, 3, 1],
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
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
        "component": "category",
    }
    assert result.dtypes.to_dict() == expected
