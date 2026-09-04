"""Tests for `pixelator.pna.analysis.segmentation.partition_counts`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.analysis.segmentation import partition_counts
from pixelator.pna.graph import PNAGraph

CELL1 = "cell1"
CELL2 = "cell2"
INTERFACE = "interface"


def _tiny_graph() -> PNAGraph:
    """Two small communities plus two interface nodes, exclusive markers."""
    edges = pd.DataFrame(
        {
            "umi1": ["a1", "a2", "a3"],
            "umi2": ["b1", "b2", "b3"],
            "read_count": [1, 1, 1],
            "marker_1": ["CD3e", "CD20", "HLA-ABC"],
            "marker_2": ["CD4", "CD19", "HLA-ABC"],
        }
    )
    return PNAGraph.from_edgelist(edges)


@pytest.fixture
def graph() -> PNAGraph:
    return _tiny_graph()


@pytest.fixture
def node_order(graph: PNAGraph) -> list:
    return list(graph.raw.nodes())


@pytest.fixture
def labels(node_order: list) -> list[str]:
    by_node = {
        "a1": CELL1,
        "b1": CELL1,
        "a2": CELL2,
        "b2": CELL2,
        "a3": INTERFACE,
        "b3": INTERFACE,
    }
    return [by_node[node] for node in node_order]


@pytest.fixture
def expected_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CD19": [0, 1, 0],
            "CD20": [0, 1, 0],
            "CD3e": [1, 0, 0],
            "CD4": [1, 0, 0],
            "HLA-ABC": [0, 0, 2],
        },
        index=pd.Index([CELL1, CELL2, INTERFACE], name="partition"),
    )


def _align_expected(result: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    return expected.reindex(index=result.index, columns=result.columns)


def test_partition_counts_requires_exactly_one_argument(graph):
    with pytest.raises(
        ValueError, match="Either `partition` or `partition_column` must be provided"
    ):
        partition_counts(graph)

    with pytest.raises(
        ValueError,
        match="One of `partition` or `partition_column` must be provided, not both",
    ):
        partition_counts(graph, partition=["a"], partition_column="compartment")


def test_partition_counts_aggregates_by_partition_vector(
    graph, labels, expected_counts
):
    result = partition_counts(graph, partition=labels)
    expected = _align_expected(result, expected_counts)
    assert_frame_equal(result, expected, check_dtype=False)
    assert list(result.index) == [CELL1, CELL2, INTERFACE]


def test_partition_counts_aggregates_by_partition_column(
    graph, labels, expected_counts
):
    nx.set_node_attributes(
        graph.raw, dict(zip(graph.raw.nodes(), labels)), "compartment"
    )
    result = partition_counts(graph, partition_column="compartment")
    expected = _align_expected(result, expected_counts)
    assert_frame_equal(result, expected, check_dtype=False)


def test_partition_counts_length_mismatch_raises(graph, labels):
    with pytest.raises(
        ValueError, match="Length of `partition` must match the number of nodes"
    ):
        partition_counts(graph, partition=labels[:2])


def test_partition_counts_missing_column_raises(graph):
    with pytest.raises(
        ValueError, match="Column 'compartment' not found in cell graph node attributes"
    ):
        partition_counts(graph, partition_column="compartment")


def test_partition_counts_wrong_graph_type_raises():
    with pytest.raises(TypeError, match="graph must be a PNAGraph"):
        partition_counts("not a graph", partition=["a"])


def test_partition_counts_exported_from_analysis():
    from pixelator.pna.analysis import partition_counts as exported

    assert exported is partition_counts
