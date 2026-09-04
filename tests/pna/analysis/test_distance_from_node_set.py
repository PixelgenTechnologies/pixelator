"""Tests for `pixelator.pna.analysis.segmentation.distance_from_node_set`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import polars as pl
import pytest

from pixelator.pna.analysis.segmentation import distance_from_node_set
from pixelator.pna.graph import PNAGraph

# Bipartite line a-b-c-d-e plus a disconnected edge f-g:
#   a -- b -- c -- d -- e    f -- g
_EDGES = {
    "umi1": ["a", "c", "c", "e", "f"],
    "umi2": ["b", "b", "d", "d", "g"],
    "marker_1": ["MA", "MC", "MC", "ME", "MF"],
    "marker_2": ["MB", "MB", "MD", "MD", "MG"],
    "read_count": [1, 1, 1, 1, 1],
}


def _synthetic_graph() -> PNAGraph:
    return PNAGraph.from_edgelist(pl.DataFrame(_EDGES).lazy())


def _distances(graph: PNAGraph) -> dict:
    return nx.get_node_attributes(graph.raw, "distance_from_seed")


@pytest.fixture
def graph() -> PNAGraph:
    return _synthetic_graph()


def test_distance_from_node_set_single_seed_exact_hops(graph):
    result = distance_from_node_set(graph, "a")

    assert result is graph
    distances = _distances(graph)
    assert distances == {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": None,
        "g": None,
    }
    assert all(isinstance(distances[n], int) for n in "abcde")


def test_distance_from_node_set_multiple_seeds(graph):
    distance_from_node_set(graph, ["a", "e"])
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["e"] == 0
    assert distances["b"] == 1
    assert distances["d"] == 1
    assert distances["c"] == 2
    assert distances["f"] is None
    assert distances["g"] is None
    assert sum(d == 0 for d in distances.values() if d is not None) == 2


def test_distance_from_node_set_respects_max_iter(graph):
    distance_from_node_set(graph, "a", max_iter=2)
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["b"] == 1
    assert distances["c"] == 2
    assert distances["d"] is None
    assert distances["e"] is None
    assert distances["f"] is None
    assert distances["g"] is None
    reached = [d for d in distances.values() if d is not None]
    assert max(reached) == 2


def test_distance_from_node_set_max_iter_zero(graph):
    distance_from_node_set(graph, ["a", "c"], max_iter=0)
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["c"] == 0
    assert all(distances[n] is None for n in "bdefg")


def test_distance_from_node_set_replaces_existing_attribute(graph):
    nx.set_node_attributes(graph.raw, 999, "distance_from_seed")
    distance_from_node_set(graph, "a")
    distances = _distances(graph)

    assert 999 not in distances.values()
    assert distances["a"] == 0
    assert distances["f"] is None


def test_distance_from_node_set_missing_seed_raises(graph):
    with pytest.raises(ValueError, match="seed nodes must be present"):
        distance_from_node_set(graph, "not_a_real_node")


def test_distance_from_node_set_partially_missing_seeds_raises(graph):
    with pytest.raises(ValueError, match="seed nodes must be present"):
        distance_from_node_set(graph, ["a", "missing"])


def test_distance_from_node_set_empty_seeds_raises(graph):
    with pytest.raises(ValueError, match="at least one node"):
        distance_from_node_set(graph, [])


def test_distance_from_node_set_invalid_graph_raises(graph):
    with pytest.raises(TypeError, match="PNAGraph"):
        distance_from_node_set("not a graph", "a")


def test_distance_from_node_set_invalid_max_iter_raises(graph):
    with pytest.raises(TypeError, match="max_iter"):
        distance_from_node_set(graph, "a", max_iter=1.5)
    with pytest.raises(ValueError, match="max_iter"):
        distance_from_node_set(graph, "a", max_iter=-1)


def test_distance_from_node_set_numpy_integer_max_iter(graph):
    distance_from_node_set(graph, "a", max_iter=np.int64(2))
    assert _distances(graph)["c"] == 2
    assert _distances(graph)["d"] is None


def test_distance_from_node_set_verbose_logs(graph, caplog):
    with caplog.at_level("INFO"):
        distance_from_node_set(graph, "a", max_iter=2, verbose=True)

    assert any("Iteration" in rec.message for rec in caplog.records)


def test_distance_from_node_set_exported_from_analysis():
    from pixelator.pna.analysis import distance_from_node_set as exported

    assert exported is distance_from_node_set
    assert (
        inspect.signature(distance_from_node_set).parameters["max_iter"].default == 40
    )
