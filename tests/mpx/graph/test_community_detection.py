"""Tests relating to community detection.

Copyright © 2022 Pixelgen Technologies AB.
"""

import networkx as nx
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from pixelator.mpx.graph.community_detection import (
    connect_components,
    recover_technical_multiplets,
)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connect_components(enable_backend, input_edgelist, output_dir, metrics_file):
    """Test connect components function.

    Args:
    enable_backend: Enable backend.
    input_edgelist: Input edgelist.
    output_dir: Output dir.
    metrics_file: Metrics file.

    """
    connect_components(
        input=input_edgelist,
        output=output_dir,
        sample_name="test",
        metrics_file=metrics_file,
        multiplet_recovery=True,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.parquet"))
    result = pd.read_parquet(result_pixel_data_file)
    assert len(result["component"].unique()) == 2


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connect_components_benchmark(
    benchmark, enable_backend, input_edgelist, output_dir, metrics_file
):
    """Verify connect components benchmark.

    Args:
    benchmark: benchmark.
    enable_backend: enable backend.
    input_edgelist: input edgelist.
    output_dir: output dir.
    metrics_file: metrics file.

    """
    benchmark(
        connect_components,
        input=input_edgelist,
        output=output_dir,
        sample_name="test",
        metrics_file=metrics_file,
        multiplet_recovery=True,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.parquet"))
    result = pd.read_parquet(result_pixel_data_file)
    assert len(result["component"].unique()) == 2


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connect_components_no_recovery(
    enable_backend, input_edgelist, output_dir, metrics_file
):
    """Test connect components with no recovery function.

    Args:
    enable_backend: Enable backend.
    input_edgelist: Input edgelist.
    output_dir: Output dir.
    metrics_file: Metrics file.

    """
    connect_components(
        input=input_edgelist,
        output=output_dir,
        sample_name="test",
        metrics_file=metrics_file,
        multiplet_recovery=False,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.parquet"))
    result = pd.read_parquet(result_pixel_data_file)
    assert len(result["component"].unique()) == 1


def test_recovery_technical_multiplets(
    edgelist_with_communities: pd.DataFrame,
):
    """Test recovery of technical multiplet components.

    Args:
    edgelist_with_communities: Edgelist with communities.

    """
    assert len(edgelist_with_communities["component"].unique()) == 1
    node_component_map = pd.Series(
        index=set(edgelist_with_communities["upia"]).union(
            set(edgelist_with_communities["upib"])
        )
    )
    node_component_map[:] = 0
    edges = (
        edgelist_with_communities.groupby(["upia", "upib"])
        .count()
        .reset_index()
        .rename(columns={"count": "len"})
    )
    result, depth_info = recover_technical_multiplets(
        edgelist=edges,
        node_component_map=node_component_map,
    )
    assert result.nunique() == 2
    assert set(depth_info.unique()) == {1}


def test_recovery_technical_multiplets_benchmark(
    benchmark,
    edgelist_with_communities: pd.DataFrame,
):
    """Verify recovery technical multiplets benchmark.

    Args:
    benchmark: benchmark. edgelist_with_communities (pd.DataFrame): edgelist with communities.
    edgelist_with_communities: Edgelist with communities.

    """
    assert len(edgelist_with_communities["component"].unique()) == 1

    edges = (
        edgelist_with_communities.groupby(["upia", "upib"])
        .count()
        .reset_index()
        .rename(columns={"count": "len"})
    )
    node_component_map = pd.Series(
        index=set(edgelist_with_communities["upia"]).union(
            set(edgelist_with_communities["upib"])
        )
    )
    node_component_map[:] = 0
    result, depth_info = benchmark(
        recover_technical_multiplets,
        edgelist=edges,
        node_component_map=node_component_map,
    )
    assert result.nunique() == 2
    assert set(depth_info.unique()) == {1}
