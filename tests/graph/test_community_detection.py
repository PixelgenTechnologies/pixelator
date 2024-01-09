"""Tests relating to community detection.

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest
from pixelator.graph.community_detection import (
    community_detection_crossing_edges,
    connect_components,
    detect_edges_to_remove,
    recover_technical_multiplets,
)

import polars as pl


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_connect_components(enable_backend, input_edgelist, output_dir, metrics_file):
    """Test connect components function."""
    connect_components(
        input=input_edgelist,
        output=output_dir,
        output_prefix="test",
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
    benchmark(
        connect_components,
        input=input_edgelist,
        output=output_dir,
        output_prefix="test",
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
    """Test connect components with no recovery function."""
    connect_components(
        input=input_edgelist,
        output=output_dir,
        output_prefix="test",
        metrics_file=metrics_file,
        multiplet_recovery=False,
        min_count=1,
    )

    result_pixel_data_file = next(output_dir.glob("*.edgelist.parquet"))
    result = pd.read_parquet(result_pixel_data_file)
    assert len(result["component"].unique()) == 1


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_recovery_technical_multiplets(
    enable_backend, edgelist_with_communities: pd.DataFrame, graph_with_communities
):
    """Test recovery of technical multiplet components."""
    assert len(edgelist_with_communities["component"].unique()) == 1

    result, info = recover_technical_multiplets(
        edgelist=pl.DataFrame(edgelist_with_communities).lazy(),
        graph=graph_with_communities,
    )
    assert len(result.collect().to_pandas()["component"].unique()) == 2
    assert info.keys() == {"PXLCMP0000000"}
    assert sorted(list(info.values())[0]) == ["RCVCMP0000000", "RCVCMP0000001"]


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_recovery_technical_multiplets_benchmark(
    benchmark,
    enable_backend,
    edgelist_with_communities: pd.DataFrame,
    graph_with_communities,
):
    assert len(edgelist_with_communities["component"].unique()) == 1

    result, info = benchmark(
        recover_technical_multiplets,
        edgelist=pl.LazyFrame(edgelist_with_communities).lazy(),
        graph=graph_with_communities,
    )
    assert len(result.collect().to_pandas()["component"].unique()) == 2
    assert info.keys() == {"PXLCMP0000000"}
    assert sorted(list(info.values())[0]) == ["RCVCMP0000000", "RCVCMP0000001"]


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_community_detection_crossing_edges(enable_backend, graph_with_communities):
    """Test discovery of crossing edges from graph with communities."""
    result = community_detection_crossing_edges(
        graph=graph_with_communities,
        leiden_iterations=2,
    )
    assert result == [{"CTCGTACCTGGGACTGATACT", "TGTAAGTCAGTTGCAGGTTGG"}]


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_detect_edges_to_remove(enable_backend, graph_with_communities):
    """Test discovery of edges to remove from edgelist."""
    result = detect_edges_to_remove(graph_with_communities, leiden_iterations=2)
    assert result == [{"CTCGTACCTGGGACTGATACT", "TGTAAGTCAGTTGCAGGTTGG"}]
