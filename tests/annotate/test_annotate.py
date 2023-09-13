"""
Tests for the annotate module

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData

from pixelator.annotate import filter_components_sizes
from pixelator.cli.annotate import annotate_components
from pixelator.config import AntibodyPanel


def test_filter_components_no_filters(adata: AnnData):
    sizes = adata.obs["vertices"].to_numpy()
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
    sizes = adata.obs["vertices"].to_numpy()
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
    sizes = adata.obs["vertices"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=None,
        max_size=1996,
    )
    assert sum(result) == 2
    assert sorted(sizes[result]) == [1995, 1995]


def test_filter_components_all_active(adata: AnnData):
    # This is the basic shape of the anndata object used here
    #            vertices  edges  markerS  ...  mean_count  degree_mean_upia       umi
    # component                            ...
    # 0000000        1995   6000        7  ...         1.0          6.018054  6.018054
    # 0000001        1998   6000        7  ...         1.0          6.006006  6.006006
    # 0000002        1996   6000        7  ...         1.0          6.018054  6.018054
    # 0000003        1995   6000        7  ...         1.0          6.024096  6.024096
    # 0000004        1996   6000        7  ...         1.0          6.000000  6.000000
    sizes = adata.obs["vertices"].to_numpy()
    result = filter_components_sizes(
        component_sizes=sizes,
        min_size=1995,
        max_size=1998,
    )
    assert sum(result) == 2
    assert sorted(sizes[result]) == [1996, 1996]


@pytest.mark.integration_test
def test_annotate_adata(edgelist: pd.DataFrame, tmp_path: Path, panel: AntibodyPanel):
    output_prefix = "test_filtered"
    metrics_file = tmp_path / "metrics.json"
    assert not metrics_file.is_file()

    # TODO test dynamic_filter ON (need to change distribution of test data)

    tmp_edgelist_file = tmp_path / "tmp_edgelist.csv.gz"
    edgelist.to_csv(
        tmp_edgelist_file, header=True, index=False, sep=",", compression="gzip"
    )

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
    assert (tmp_path / f"{output_prefix}.dataset.pxl").is_file()
    assert metrics_file.is_file()
