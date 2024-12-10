"""
Tests for the annotate module

Copyright © 2023 Pixelgen Technologies AB.
"""

from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData

from pixelator.annotate import (
    NoCellsFoundException,
    cluster_components,
    filter_components_sizes,
)
from pixelator.cli.annotate import annotate_components
from pixelator.config import AntibodyPanel
from pixelator.pixeldataset.utils import read_anndata


def test_filter_components_no_filters(adata: AnnData):
    sizes = adata.obs["pixels"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=None,
        max_size=None,
    )
    assert sum(result) == len(sizes)


def test_filter_components_min_size(adata: AnnData):
    # This is the basic shape of the anndata object used here
    #            vertices  edges  markerS  ...  mean_count  degree_mean_upia       umi
    # component                            ...
    # 0000000        1995   6000        7  ...         1.0          6.018054  6.018054
    # 0000001        1998   6000        7  ...         1.0          6.006006  6.006006
    # 0000002        1996   6000        7  ...         1.0          6.018054  6.018054
    # 0000003        1995   6000        7  ...         1.0          6.024096  6.024096
    # 0000004        1996   6000        7  ...         1.0          6.000000  6.000000
    sizes = adata.obs["pixels"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=1995,
        max_size=None,
    )
    assert sum(result) == 3
    assert sorted(sizes[result]) == [1996, 1996, 1998]


def test_filter_components_max_size(adata: AnnData):
    # This is the basic shape of the anndata object used here
    #            vertices  edges  markerS  ...  mean_count  degree_mean_upia       umi
    # component                            ...
    # 0000000        1995   6000        7  ...         1.0          6.018054  6.018054
    # 0000001        1998   6000        7  ...         1.0          6.006006  6.006006
    # 0000002        1996   6000        7  ...         1.0          6.018054  6.018054
    # 0000003        1995   6000        7  ...         1.0          6.024096  6.024096
    # 0000004        1996   6000        7  ...         1.0          6.000000  6.000000
    sizes = adata.obs["pixels"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=None,
        max_size=1996,
    )
    assert sum(result) == 2
    assert sorted(sizes[result]) == [1995, 1995]


def test_filter_components_all_active(adata: AnnData):
    # This is the basic shape of the anndata object used here
    # component pixels a_pixels b_pixels antibodies molecules reads mean_reads_per_molecule median_reads_per_molecule mean_b_pixels_per_a_pixel median_b_pixels_per_a_pixel mean_a_pixels_per_b_pixel median_a_pixels_per_b_pixel a_pixel_b_pixel_ratio mean_molecules_per_a_pixel median_molecules_per_a_pixel
    # PXLCMP0000000 1996    997 999 7   6000    6000    1.0 1.0 6.018054    6.0 6.018054    6.0 0.997998    6.018054    6.0
    # PXLCMP0000001 1995    996 999 7   6000    6000    1.0 1.0 6.024096    6.0 6.024096    6.0 0.996997    6.024096    6.0
    # PXLCMP0000002 1998    999 999 7   6000    6000    1.0 1.0 6.006006    6.0 6.006006    6.0 1.000000    6.006006    6.0
    # PXLCMP0000003 1996    1000    996 7   6000    6000    1.0 1.0 6.000000    6.0 6.000000    6.0 1.004016    6.000000    6.0
    # PXLCMP0000004 1995    997 998 7   6000    6000    1.0 1.0 6.018054    6.0 6.018054    6.0 0.998998    6.018054    6.0

    sizes = adata.obs["pixels"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=1995,
        max_size=1998,
    )
    assert sum(result) == 2
    assert sorted(sizes[result]) == [1996, 1996]


@pytest.mark.integration_test
def test_cluster_components(data_root):
    adata = read_anndata(
        str(data_root / "Sample01_human_pbmcs_unstimulated.adata.h5ad")
    )
    # Clear the existing leiden annotation
    del adata.obs["leiden"]

    cluster_components(adata, "clr", random_seed=1)

    assert not adata.obs["leiden"].empty

    expected_groups = [
        ["RCVCMP0000000", "RCVCMP0000006"],
        ["RCVCMP0000007", "RCVCMP0000008", "RCVCMP0000012", "RCVCMP0000002"],
        ["RCVCMP0000005", "RCVCMP0000013"],
        ["RCVCMP0000003"],
        ["RCVCMP0000010"],
    ]

    # We check the groups to be the same here, because the integer name
    # of the leiden groups are not the same across runs.
    for group in expected_groups:
        assert len(set(adata.obs["leiden"].loc[group])) == 1


@pytest.mark.integration_test
def test_annotate_adata(edgelist: pd.DataFrame, tmp_path: Path, panel: AntibodyPanel):
    output_prefix = "test_filtered"
    metrics_file = tmp_path / "metrics.json"
    assert not metrics_file.is_file()

    # TODO test dynamic_filter ON (need to change distribution of test data)

    tmp_edgelist_file = tmp_path / "tmp_edgelist.parquet"
    edgelist.to_parquet(tmp_edgelist_file, index=False)

    annotate_components(
        input=str(tmp_edgelist_file),
        panel=panel,
        output=str(tmp_path),
        output_prefix=output_prefix,
        metrics_file=str(metrics_file),
        min_size=None,
        max_size=None,
        dynamic_filter=None,
        verbose=True,
        aggregate_calling=True,
    )
    assert (tmp_path / f"{output_prefix}.raw_components_metrics.csv.gz").is_file()
    assert (tmp_path / f"{output_prefix}.annotate.dataset.pxl").is_file()
    assert metrics_file.is_file()


@pytest.mark.integration_test
def test_annotate_adata_should_raise_no_cells_count_exception(
    edgelist: pd.DataFrame, tmp_path: Path, panel: AntibodyPanel
):
    with pytest.raises(NoCellsFoundException) as expected_exception:
        output_prefix = "test_filtered"
        metrics_file = tmp_path / "metrics.json"
        assert not metrics_file.is_file()
        tmp_edgelist_file = tmp_path / "tmp_edgelist.parquet"
        edgelist.to_parquet(tmp_edgelist_file, index=False)

        annotate_components(
            input=str(tmp_edgelist_file),
            panel=panel,
            output=str(tmp_path),
            output_prefix=output_prefix,
            metrics_file=str(metrics_file),
            min_size=100_000,  # Nothing should pass this
            max_size=None,
            dynamic_filter=None,
            verbose=True,
            aggregate_calling=True,
        )
