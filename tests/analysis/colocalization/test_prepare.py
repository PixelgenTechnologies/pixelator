"""
Tests for the colocalization modules

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import functools

import pandas as pd
from numpy.random import default_rng
from pandas.testing import assert_frame_equal

from pixelator.analysis.colocalization.prepare import (
    filter_by_marker_counts,
    filter_by_region_counts,
    filter_by_unique_values,
    prepare_from_edgelist,
    prepare_from_graph,
)
from pixelator.graph.utils import Graph

random_number_generator = default_rng(seed=433)


def test_prepare_from_edgelist(edgelist):
    result = prepare_from_edgelist(edgelist=edgelist, group_by="upia")
    assert len(result) == edgelist["upia"].nunique()
    assert len(result.columns) == edgelist["marker"].nunique()


def test_prepare_from_graph(edgelist):
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=False,
    )
    for component in graph.components().subgraphs():
        result = prepare_from_graph(component, n_neighbours=0)
        assert len(result) == len(component.vs)
        unique_markers_in_component = functools.reduce(
            set.union,
            [set(node_markers.keys()) for node_markers in component.vs["markers"]],
        )
        assert len(result.columns) == len(unique_markers_in_component)


def test_prepare_from_graph_and_edgelist_eq_for_no_neigbours(edgelist):
    for _, component_df in edgelist.groupby("component"):
        graph = Graph.from_edgelist(
            edgelist=component_df,
            add_marker_counts=True,
            simplify=True,
            use_full_bipartite=False,
        )
        graph_result = prepare_from_graph(graph, n_neighbours=0)
        edgelist_result = prepare_from_edgelist(edgelist=component_df, group_by="upia")
        # We drop the indexes since whey will be named differently depending on the
        # method.
        assert_frame_equal(
            graph_result.sort_index().reset_index(drop=True),
            edgelist_result.sort_index().reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            check_column_type=False,
        )


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
