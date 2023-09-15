"""
This module contains functions providing to utility procedures
on pixelator graph operations.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import warnings
from functools import reduce
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import identity

from pixelator.graph.constants import (
    DEFAULT_COMPONENT_PREFIX,
    DIGITS,
)
from pixelator.graph.graph import Graph

logger = logging.getLogger(__name__)


def components_metrics(edgelist: pd.DataFrame) -> pd.DataFrame:
    """
    A helper function that computes a dataframe of metrics for
    each component in the data present in the edge list given
    as input (component column). The metrics include: vertices,
    edges, markers, upis, degree mean and max.
    :param edgelist: an edge list dataframe with a membership column
    :returns: a pd.DataFrame with the metrics per component
    :raises: AssertionError when the input edge list is not valid
    """
    if "component" not in edgelist.columns:
        raise AssertionError("Edge list is missing the membership column")

    logger.debug(
        "Computing components metrics for edge list with %i edges", edgelist.shape[0]
    )

    cmetrics = []
    index = []
    # iterate the components to obtain the metrics of each component
    for component_id, group_df in edgelist.groupby("component"):
        # compute metrics
        n_edges = group_df.shape[0]
        n_vertices = len(
            set(group_df["upia"].unique().tolist() + group_df["upib"].unique().tolist())
        )
        n_markers = group_df["marker"].nunique()
        upia_count = group_df["upia"].nunique()
        upib_count = group_df["upib"].nunique()
        tot_count = group_df["count"].sum()
        mean_count = group_df["count"].mean()
        median_count = group_df["count"].median()
        tot_umi = group_df["umi"].nunique()
        upia_degree = group_df.groupby("upia")["upib"].nunique()
        upia_mean_degree = upia_degree.mean()
        upia_median_degree = upia_degree.median()
        umi_degree = group_df.groupby("upia")["umi"].count()
        upi_umi_median = umi_degree.median()
        upi_umi_mean = umi_degree.mean()
        upia_per_upib = upia_count / upib_count
        cmetrics.append(
            (
                n_vertices,
                n_edges,
                n_markers,
                upia_count,
                upib_count,
                tot_umi,
                tot_count,
                mean_count,
                median_count,
                upia_mean_degree,
                upia_median_degree,
                upi_umi_mean,
                upi_umi_median,
                upia_per_upib,
            )
        )
        index.append(component_id)

    # create components metrics data frame
    components_metrics = pd.DataFrame(
        index=pd.Index(index, name="component"),
        columns=[
            "vertices",
            "edges",
            "antibodies",
            "upia",
            "upib",
            "umi",
            "reads",
            "mean_reads",
            "median_reads",
            "mean_upia_degree",
            "median_upia_degree",
            "mean_umi_per_upia",
            "median_umi_per_upia",
            "upia_per_upib",
        ],
        data=cmetrics,
    )

    logger.debug("Component metrics computed")
    return components_metrics


def create_node_markers_counts(
    graph: Graph,
    k: int = 0,
    normalization: Optional[Literal["mean"]] = None,
) -> pd.DataFrame:
    """
    A helper function that computes and returns a data frame of antibody counts per
    node (vertex) of the graph given as input (preferably a fully connected component).
    The parameter k allows to include neighbors (of each node) when computing the
    counts (using `agg_func` to aggregate the counts). K defines the number of levels
    when searching neighbors. The graph must contain a vertex attribute called 'markers'
    which is dictionary of marker counts per vertex.
    :param graph: a graph (preferably a connected component)
    :param k: number of neighbors to include per node (0 no neighbors,
              1 first level, ...)
    :param normalization: selects a normalization method to apply when
                          building neighborhoods
    :returns: a pd.DataFrame with the antibody counts per node
    """

    if k == 0 and normalization:
        warnings.warn(
            (
                f"Using `normalization={normalization}` when k=0 "
                "has no effect, since no neighborhood is created."
            )
        )

    if "markers" not in graph.vs.attributes():
        raise AssertionError("Could not find 'markers' in vertex attributes")
    markers = list(sorted(graph.vs[0]["markers"].keys()))
    node_marker_counts = pd.DataFrame.from_records(
        graph.vs["markers"], columns=markers, index=graph.vs["name"]
    )
    node_marker_counts = node_marker_counts.reindex(
        sorted(node_marker_counts.columns), axis=1
    )
    node_marker_counts.columns.name = "markers"
    if k == 0:
        return node_marker_counts

    # This method first finds all nodes that are connected by a path
    # with the a shortest path of k or shorter, encoded in a new
    # adjacency matrix An.
    # We then find the marker counts for those neighbourhoods
    # by matrix multiplication.

    def sparse_mat_power(x, n):
        if n == 0:
            return identity(x.shape[0])
        return reduce(lambda x, y: x @ y, (x for _ in range(0, n)))

    A = graph.get_adjacency_sparse()
    An = (
        reduce(lambda x, y: x + y, [sparse_mat_power(A, n) for n in range(0, k + 1)])
        > 0
    ).astype(int)
    neighbourhood_counts = An * node_marker_counts

    # TODO Optionally add more methods here
    if normalization == "mean":
        nbr_of_neighbors_per_node = An.sum(axis=1)
        neighbourhood_counts = neighbourhood_counts / nbr_of_neighbors_per_node

    df = pd.DataFrame(
        data=neighbourhood_counts, columns=node_marker_counts.columns.copy()
    )
    df.columns.name = "markers"
    return df


def edgelist_metrics(edgelist: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    A simple function that computes a dictionary of basic metrics
    from an edge list (pd.DataFrame).
    :param edgelist: the edge list (pd.DataFrame)
    :returns: a dictionary of metrics (metric -> float)
    """
    metrics: Dict[str, Union[int, float]] = {}
    metrics["total_upia"] = edgelist["upia"].nunique()
    metrics["total_upib"] = edgelist["upib"].nunique()
    metrics["total_umi"] = edgelist["umi"].nunique()
    metrics["total_upi"] = metrics["total_upia"] + metrics["total_upib"]
    metrics["frac_upib_upia"] = round(metrics["total_upib"] / metrics["total_upia"], 2)
    metrics["markers"] = edgelist["marker"].nunique()
    metrics["edges"] = edgelist.shape[0]
    upia_degree = edgelist.groupby("upia")["upib"].nunique()
    metrics["upia_degree_mean"] = round(upia_degree.mean(), 2)
    metrics["upia_degree_median"] = round(upia_degree.median(), 2)
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )
    components = graph.connected_components()
    metrics["vertices"] = graph.vcount()
    metrics["components"] = len(components)
    metrics["components_modularity"] = round(components.modularity, 2)
    biggest = components.giant()
    metrics["frac_largest_edges"] = round(biggest.ecount() / metrics["edges"], 2)
    metrics["frac_largest_vertices"] = round(biggest.vcount() / metrics["vertices"], 2)
    return metrics


def update_edgelist_membership(
    edgelist: pd.DataFrame,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    A helper function that computes a vector of component membership ids (str)
    for each edge (row) in the given `edgelist` (pd.DataFrame). Each component id
    corresponds to a connected component in the respective graph (iGraph). The
    prefix attribute `prefix` is prepended to each component id. The component ids
    will be added to the returned edge list in a column called component.
    :params edgelist: the edge list (pd.DataFrame)
    :params prefix: the prefix to prepend to the component ids, if None will
                    use `DEFAULT_COMPONENT_PREFIX`
    :returns: the update edge list (pd.DataFrame)
    """
    logger.debug("Updating membership in edge list with %i rows", edgelist.shape[0])

    if prefix is None:
        prefix = DEFAULT_COMPONENT_PREFIX

    # check if the edge list contains a component
    if "component" in edgelist.columns:
        logger.info("The input edge list already contain a component column")

    # create graph
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )

    # iterate the components to create the membership column
    membership = np.empty(edgelist.shape[0], dtype=object)
    connected_components = graph.connected_components()
    component_id_format = f"{prefix}{{:0{DIGITS}d}}"
    for i, component in enumerate(connected_components):
        component_id = component_id_format.format(i)
        edges = [e.index for e in graph.es.select(_within=component)]
        membership[edges] = component_id

    # update the edge list
    edgelist = edgelist.assign(component=membership)
    logger.debug("Membership in edge list updated")

    return edgelist
