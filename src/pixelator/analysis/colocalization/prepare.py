"""Functions used to prepare a graph for colocalization computations.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from typing import Literal

import pandas as pd

from pixelator.analysis.colocalization.types import RegionByCountsDataFrame
from pixelator.graph.utils import Graph, create_node_markers_counts


def prepare_from_graph(graph: Graph, n_neighbours: int = 1) -> RegionByCountsDataFrame:
    """Prepare a RegionByCountsDataFrame from a graph.

    :param graph: the graph to prepare from
    :param n_neighbours: size of the neighbourhood, defaults to 1
    :return: a RegionByCountsDataFrame
    :rtype: RegionByCountsDataFrame
    """
    counts_df = create_node_markers_counts(graph=graph, k=n_neighbours)
    return counts_df


def prepare_from_edgelist(
    edgelist: pd.DataFrame, group_by: Literal["upia", "upib"] = "upia"
) -> RegionByCountsDataFrame:
    """Prepare a RegionByCountsDataFrame from an edgelist.

    Prepare a RegionByCountsDataFrame from an edgelist, using
    either upia or upib as regions

    :param edgelist: edgelist to prepare from
    :param group_by: value to create regions from, defaults to "upia"
    :return: a RegionByCountsDataFrame
    :rtype: RegionByCountsDataFrame
    """
    markers_per_pixel = (
        edgelist.astype({"upia": "string[pyarrow]", "upib": "string[pyarrow]"})
        .groupby(by=group_by)["marker"]
        .value_counts()
        .reset_index()
    ).pivot_table(index=group_by, columns="marker", values="count", fill_value=0)
    markers_per_pixel.columns.name = "markers"
    markers_per_pixel.index.name = "node"
    markers_per_pixel.index = markers_per_pixel.index.astype("string[pyarrow]")
    markers_per_pixel = markers_per_pixel.reindex(
        sorted(markers_per_pixel.columns), axis=1
    )
    markers_per_pixel.columns = markers_per_pixel.columns.astype("string[pyarrow]")
    return markers_per_pixel


def filter_by_region_counts(
    df: RegionByCountsDataFrame, min_region_counts: int = 5
) -> RegionByCountsDataFrame:
    """Filter by counts in the region.

    Filter regions from a RegionByCountsDataFrame based on
    how many counts are in the region

    :param df: dataframe to filter
    :param min_region_counts: minimum number of counts in region > min_region_counts,
                              defaults to 5
    :return: the filtered dataframe
    :rtype: RegionByCountsDataFrame
    """
    return df[df.sum(axis="columns") > min_region_counts]


def filter_by_marker_counts(
    df: RegionByCountsDataFrame, min_marker_counts: int = 10
) -> RegionByCountsDataFrame:
    """Filter markers by minimum count.

    Filter markers from a RegionByCountsDataFrame based on how many counts
    available for that marker

    :param df: dataframe to filter
    :param min_marker_counts: marker > min_marker_counts, defaults to 10
    :return: the filtered dataframe
    :rtype: RegionByCountsDataFrame
    """
    return df.loc[:, df.sum(axis="index") > min_marker_counts]


def filter_by_unique_values(
    df: RegionByCountsDataFrame, at_least_n_unique: int = 1
) -> RegionByCountsDataFrame:
    """Filter markers by minimum number of unique values.

    Filter markers from a RegionByCountsDataFrame based on the minimum
    number of unique values for that marker

    :param df: dataframe to filter
    :param at_least_n_unique: minimum number of unique values >= at_least_n_unique,
                              defaults to 1
    :return: the filtered dataframe
    :rtype: RegionByCountsDataFrame
    """
    return df.loc[:, df.nunique() >= at_least_n_unique]
