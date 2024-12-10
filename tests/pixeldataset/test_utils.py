"""Tests for pixeldataset.utils module.

Copyright © 2023 Pixelgen Technologies AB.
"""

# pylint: disable=redefined-outer-name

import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from pixelator.config import AntibodyPanel
from pixelator.graph import write_recovered_components
from pixelator.pixeldataset.utils import (
    antibody_metrics,
    component_antibody_counts,
    edgelist_to_anndata,
    read_anndata,
    write_anndata,
)
from pixelator.statistics import (
    clr_transformation,
    log1p_transformation,
)
from pixelator.utils import batched

random.seed(42)
np.random.seed(42)


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
            "a_pixel_b_pixel_ratio",
            "a_pixels",
            "antibodies",
            "b_pixels",
            "mean_a_pixels_per_b_pixel",
            "mean_b_pixels_per_a_pixel",
            "mean_molecules_per_a_pixel",
            "mean_reads_per_molecule",
            "median_a_pixels_per_b_pixel",
            "median_b_pixels_per_a_pixel",
            "median_molecules_per_a_pixel",
            "median_reads_per_molecule",
            "molecules",
            "pixels",
            "reads",
            "is_potential_doublet",
            "n_edges_to_split_doublet",
        ]
    )
    assert "clr" in adata.obsm
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
            adata.obsm["clr"],
            adata2.obsm["clr"],
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

    counts_df_clr = clr_transformation(counts_df, axis=1)
    assert_array_equal(
        adata.obsm["clr"],
        counts_df_clr.to_numpy(),
    )

    counts_df_log1p = log1p_transformation(counts_df)
    assert_array_equal(
        adata.obsm["log1p"],
        counts_df_log1p.to_numpy(),
    )

    assert set(adata.obs_names) == set(edgelist["component"].unique())


def test_batched_empty_iterable():
    """Test batched with an empty iterable."""
    iterable = []
    n = 3
    batches = list(batched(iterable, n))
    assert batches == []


def test_batched_single_batch():
    """Test batched with a single batch."""
    iterable = [1, 2, 3]
    n = 3
    batches = list(batched(iterable, n))
    assert batches == [(1, 2, 3)]


def test_batched_multiple_batches():
    """Test batched with multiple batches."""
    iterable = [1, 2, 3, 4, 5, 6]
    n = 2
    batches = list(batched(iterable, n))
    assert batches == [(1, 2), (3, 4), (5, 6)]


def test_batched_last_batch_shorter():
    """Test batched with the last batch being shorter."""
    iterable = [1, 2, 3, 4, 5]
    n = 3
    batches = list(batched(iterable, n))
    assert batches == [(1, 2, 3), (4, 5)]


def test_batched_n_less_than_one():
    """Test batched with n less than one."""
    iterable = [1, 2, 3]
    n = 0
    try:
        _ = list(batched(iterable, n))
    except ValueError as e:
        assert str(e) == "n must be at least one"
    else:
        assert False, "Expected ValueError"
