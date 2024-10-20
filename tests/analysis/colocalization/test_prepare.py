"""
Tests for the colocalization modules

Copyright © 2023 Pixelgen Technologies AB.
"""

import functools

import pandas as pd
import pytest
from numpy.random import default_rng
from pandas.testing import assert_frame_equal

from pixelator.analysis.colocalization.prepare import (
    filter_by_marker_counts,
    filter_by_region_counts,
    filter_by_unique_values,
    prepare_from_graph,
)
from pixelator.graph.utils import Graph

random_number_generator = default_rng(seed=433)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_prepare_from_graph(enable_backend, edgelist):
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=False,
    )
    for component in graph.connected_components().subgraphs():
        result = prepare_from_graph(component, n_neighbours=0)
        assert len(result) == len(component.vs)
        unique_markers_in_component = functools.reduce(
            set.union,
            [
                set(node_markers.keys())
                for node_markers in component.vs.get_attribute("markers")
            ],
        )
        assert len(result.columns) == len(unique_markers_in_component)


def test_filter_by_region_counts():
    df = pd.DataFrame(
        [[1, 2, 3, 4], [2, 3, 5, 6], [10, 5, 3, 8]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    result = filter_by_region_counts(df, min_region_counts=15)
    assert result.shape == (2, 4)


def test_filter_by_marker_counts():
    df = pd.DataFrame(
        [[1, 2, 3, 4], [2, 3, 5, 6], [10, 5, 3, 8]],
        columns=["marker1", "marker2", "marker3", "marker4"],
    )
    result = filter_by_marker_counts(df, min_marker_counts=10)
    assert result.shape == (3, 3)


def test_filter_by_unique_values():
    df = pd.DataFrame([[1, 2], [2, 2], [3, 2]], columns=["marker1", "marker2"])
    result = filter_by_unique_values(df=df, at_least_n_unique=2)
    assert result.shape == (3, 1)
