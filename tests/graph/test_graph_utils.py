"""Tests for the graph utils module.

Copyright © 2023 Pixelgen Technologies AB.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from pixelator.graph import Graph
from pixelator.graph.utils import (
    components_metrics,
    create_node_markers_counts,
    edgelist_metrics,
    update_edgelist_membership,
)
from pixelator.report.common.json_encoder import PixelatorJSONEncoder
from pixelator.report.models import SummaryStatistics


def test_components_metrics(full_graph_edgelist: pd.DataFrame):
    """Test generating component metrics."""
    # test component metrics
    metrics = components_metrics(edgelist=full_graph_edgelist)
    assert_frame_equal(
        metrics,
        pd.DataFrame.from_records(
            [
                {
                    "pixels": 100,
                    "a_pixels": 50,
                    "b_pixels": 50,
                    "antibodies": 2,
                    "molecules": 2500,
                    "reads": np.uint64(2500),
                    "mean_reads_per_molecule": 1.0,
                    "median_reads_per_molecule": 1.0,
                    "mean_b_pixels_per_a_pixel": 50.0,
                    "median_b_pixels_per_a_pixel": 50.0,
                    "mean_a_pixels_per_b_pixel": 50.0,
                    "median_a_pixels_per_b_pixel": 50.0,
                    "a_pixel_b_pixel_ratio": 1.0,
                    "mean_molecules_per_a_pixel": 50.0,
                    "median_molecules_per_a_pixel": 50.0,
                }
            ],
            index=pd.Index(["23885f346392ff2c"], name="component"),
        ),
    )


def _create_df_with_expected_types(df):
    """Make sure that the dataframe gets the correct types and names."""
    df.columns.name = "markers"
    df.columns = df.columns.astype("string[pyarrow]")
    df.index.name = "node"
    return df


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_k_eq_0(enable_backend, pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 0."""
    result = create_node_markers_counts(graph=pentagram_graph, k=0)

    expected = pd.DataFrame(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected = _create_df_with_expected_types(expected)
    # The sort order is not guaranteed to be deterministic here,
    # hence the sorting.
    assert_frame_equal(result.sort_index(), expected.sort_index())


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_k_eq_1(enable_backend, pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 1."""
    result = create_node_markers_counts(graph=pentagram_graph, k=1)

    expected = pd.DataFrame(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected = _create_df_with_expected_types(expected)
    assert_frame_equal(result.sort_index(), expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_k_eq_2(enable_backend, pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 2."""
    result = create_node_markers_counts(graph=pentagram_graph, k=2)

    expected = pd.DataFrame(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected = _create_df_with_expected_types(expected)
    assert_frame_equal(result.sort_index(), expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_k_eq_2_with_mean(enable_backend, pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 2, mean values."""
    result = create_node_markers_counts(
        graph=pentagram_graph, k=2, normalization="mean"
    )

    expected = pd.DataFrame(
        [
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected = _create_df_with_expected_types(expected)
    assert_frame_equal(result.sort_index(), expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts(
    enable_backend, random_graph_edgelist: pd.DataFrame
):
    """Test build a node marker matrix with a neigbourhood of 0."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=0)
    assert counts.shape == (graph.vcount(), 2)
    # Since every edge transfers it's counts to both the A and the B node
    # we get a doubling of the number of counts compared to the number of counts
    # for each marker in the edgelist
    assert counts["A"].sum() == random_graph_edgelist["marker"].value_counts()["A"] * 2
    assert counts["B"].sum() == random_graph_edgelist["marker"].value_counts()["B"] * 2


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_with_neighbourhood_1_with_mean_normalization(
    enable_backend, random_graph_edgelist: pd.DataFrame
):
    """Test build a node marker matrix with a neigbourhood of 1, with the mean value."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=1, normalization="mean")
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == pytest.approx(715.79, abs=0.01)
    assert counts["B"].sum() == pytest.approx(706.57, abs=0.01)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_with_neighbourhood_2(
    enable_backend, random_graph_edgelist: pd.DataFrame
):
    """Test build a node marker matrix with a neigbourhood of 2."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=2)
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == 8027
    assert counts["B"].sum() == 7870


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_column_order(
    enable_backend, random_graph_edgelist: pd.DataFrame
):
    """Columns should always be returned in alphabetical sort order."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=0)
    assert counts.columns.to_list() == ["A", "B"]


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_k_eq_1_with_mean(enable_backend, pentagram_graph):
    """Test build a node marker matrix with a neigbourhood of 1, with the mean value."""
    result = create_node_markers_counts(
        graph=pentagram_graph, k=1, normalization="mean"
    )

    expected = pd.DataFrame(
        [
            [1 / 3, 0, 1 / 3, 1 / 3, 0],
            [0, 1 / 3, 0, 1 / 3, 1 / 3],
            [1 / 3, 0, 1 / 3, 0, 1 / 3],
            [1 / 3, 1 / 3, 0, 1 / 3, 0],
            [0, 1 / 3, 1 / 3, 0, 1 / 3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected = _create_df_with_expected_types(expected)
    assert_frame_equal(result.sort_index(), expected)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_create_node_markers_counts_with_neighbourhood_1(
    enable_backend, random_graph_edgelist: pd.DataFrame
):
    """Test build a node marker matrix with a neigbourhood of 1."""
    graph = Graph.from_edgelist(
        edgelist=random_graph_edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    counts = create_node_markers_counts(graph=graph, k=1)
    assert counts.shape == (graph.vcount(), 2)
    assert counts["A"].sum() == 2803
    assert counts["B"].sum() == 2759


def test_edgelist_metrics(full_graph_edgelist: pd.DataFrame, snapshot):
    """Test generating edgelist metrics."""
    metrics = edgelist_metrics(full_graph_edgelist)

    snapshot.assert_match(
        json.dumps(metrics, indent=4, cls=PixelatorJSONEncoder), "edgelist_metrics.json"
    )


def test_edgelist_metrics_on_lazy_dataframe(
    full_graph_edgelist: pd.DataFrame, snapshot
):
    full_graph_edgelist = pl.DataFrame(full_graph_edgelist).lazy()
    metrics = edgelist_metrics(full_graph_edgelist)

    snapshot.assert_match(
        json.dumps(metrics, indent=4, cls=PixelatorJSONEncoder),
        "edgelist_metrics_lazy.json",
    )


def test_update_edgelist_membership(data_root):
    """Test updating the edgelist membership."""
    edgelist = pd.read_csv(str(data_root / "test_edge_list.csv"))
    result = update_edgelist_membership(edgelist.copy())

    assert "component" not in edgelist.columns
    assert set(result["component"].unique()) == {
        "bec92437d668cfa1",
        "701ec72d3bda62d5",
        "ce2709afa8ebd1c9",
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
    }


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_update_edgelist_membership_benchmark(benchmark, enable_backend, data_root):
    """Test updating the edgelist membership."""
    edgelist = pd.read_csv(str(data_root / "test_edge_list.csv"))
    result = benchmark(update_edgelist_membership, edgelist.copy())

    assert "component" not in edgelist.columns
    assert set(result["component"].unique()) == {
        "bec92437d668cfa1",
        "701ec72d3bda62d5",
        "ce2709afa8ebd1c9",
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
    }


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_update_edgelist_membership_lazyframe(enable_backend, data_root):
    edgelist = pl.read_csv(str(data_root / "test_edge_list.csv")).lazy()
    assert "component" not in edgelist.columns

    result = update_edgelist_membership(edgelist).collect().to_pandas()

    assert set(result["component"].unique()) == {
        "bec92437d668cfa1",
        "701ec72d3bda62d5",
        "ce2709afa8ebd1c9",
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
    }
