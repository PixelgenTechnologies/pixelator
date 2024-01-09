"""Module contains various useful graph functions.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import warnings
from functools import reduce
from typing import Dict, List, Literal, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import identity

from pixelator.graph.backends.implementations import (
    NetworkXGraphBackend,
)
from pixelator.graph.constants import (
    DEFAULT_COMPONENT_PREFIX,
    DIGITS,
)
from pixelator.graph.graph import Graph

logger = logging.getLogger(__name__)


def union(graphs: List[Graph]) -> Graph:
    """Create union of graphs.

    Create a union of the provided graphs, merging any vertices
    which share the same name.

    :param graphs: the graphs to create the union from
    :return: a new graph that is the union of the input `graphs`
    :rtype: Graph
    :raises: AssertionError if not all underlying graphs have the same backend type.
    """
    backends = [type(g._backend) for g in graphs]
    if not all(map(lambda b: backends[0] == b, backends)):
        raise AssertionError("All graph objects must share the same backend")

    if backends[0] == NetworkXGraphBackend:
        return Graph(
            backend=NetworkXGraphBackend.from_raw(
                nx.union_all(
                    [graph._backend.raw for graph in graphs],
                    rename=[f"g{idx}-" for idx, _ in enumerate(graphs)],
                )
            )
        )

    raise NotImplementedError()


def components_metrics(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics per component.

    A helper function that computes a dataframe of metrics for
    each component in the data present in the edge list given
    as input (component column). The metrics include: vertices,
    edges, markers, upis, degree mean and max.
    :param edgelist: an edge list dataframe with a membership column
    :returns: a pd.DataFrame with the metrics per component
    :rtype: pd.DataFrame
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
    for component_id, group_df in edgelist.groupby("component", observed=True):
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
        # Please note that we need to use observed=True
        # here upia is a categorical column, and since not
        # all values are present in all components, this is
        # required to get a correct value.
        upia_degree = group_df.groupby("upia", observed=True)["upib"].nunique()
        upia_mean_degree = upia_degree.mean()
        upia_median_degree = upia_degree.median()
        # Same reasoning as above
        umi_degree = group_df.groupby("upia", observed=True)["umi"].count()
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
            "mean_reads_per_molecule",
            "median_reads_per_molecule",
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
    """Create a matrix of marker counts for each in the graph.

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
    :rtype: pd.DataFrame
    :raises AssertionError: if no 'markers' attribute is found on the vertices
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
    markers = list(sorted(next(iter(graph.vs))["markers"].keys()))
    node_marker_counts = pd.DataFrame.from_records(
        list(graph.vs.get_attribute("markers")),
        columns=markers,
        index=list(graph.vs.get_attribute("name")),
    )
    node_marker_counts = node_marker_counts.reindex(
        sorted(node_marker_counts.columns), axis=1
    )
    node_marker_counts.columns.name = "markers"
    node_marker_counts.columns = node_marker_counts.columns.astype("string[pyarrow]")
    node_marker_counts.index = pd.Index(
        list(graph.vs.get_attribute("name")), dtype="string[pyarrow]", name="node"
    )
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
        data=neighbourhood_counts,
        columns=node_marker_counts.columns.copy(),
        index=pd.Index(
            list(graph.vs.get_attribute("name")), dtype="string[pyarrow]", name="node"
        ),
    )
    df.columns.name = "markers"
    return df


MetricsDict = Dict[str, Union[int, float]]


def _calculate_graph_metrics(
    metrics: MetricsDict,
    graph: Optional[Graph],
    edgelist: Union[pd.DataFrame, pl.LazyFrame],
) -> MetricsDict:
    if not graph:
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


def _edgelist_metrics_pandas_data_frame(
    edgelist: pd.DataFrame, graph: Optional[Graph] = None
) -> MetricsDict:
    metrics: Dict[str, Union[int, float]] = {}
    metrics["total_upia"] = edgelist["upia"].nunique()
    metrics["total_upib"] = edgelist["upib"].nunique()
    metrics["total_umi"] = edgelist["umi"].nunique()
    metrics["total_upi"] = metrics["total_upia"] + metrics["total_upib"]
    metrics["frac_upib_upia"] = round(metrics["total_upib"] / metrics["total_upia"], 2)
    metrics["markers"] = edgelist["marker"].nunique()
    metrics["edges"] = edgelist.shape[0]
    metrics["mean_count"] = round(edgelist["count"].mean(), 2)

    # Please note that we need to use observed=True
    # here upia is a categorical column, and since not
    # all values are present in all components, this is
    # required to get a correct value.
    upia_degree = edgelist.groupby("upia", observed=True)["upib"].nunique()
    metrics["upia_degree_mean"] = round(upia_degree.mean(), 2)
    metrics["upia_degree_median"] = round(upia_degree.median(), 2)

    metrics = _calculate_graph_metrics(metrics=metrics, graph=graph, edgelist=edgelist)
    return metrics


def _edgelist_metrics_lazy_frame(
    edgelist: pl.LazyFrame, graph: Optional[Graph] = None
) -> MetricsDict:
    metrics: Dict[str, Union[int, float]] = {}
    metrics["total_upia"] = edgelist.select(pl.col("upia")).collect().n_unique()
    metrics["total_upib"] = edgelist.select(pl.col("upib")).collect().n_unique()
    metrics["total_umi"] = edgelist.select(pl.col("umi")).collect().n_unique()
    metrics["total_upi"] = metrics["total_upia"] + metrics["total_upib"]
    metrics["frac_upib_upia"] = round(metrics["total_upib"] / metrics["total_upia"], 2)
    metrics["markers"] = edgelist.select(pl.col("marker")).collect().n_unique()
    # Note that we get upi here and count that, because otherwise just calling count
    # here confuses polars since there is a column with that name.
    metrics["edges"] = (
        edgelist.select(pl.col("upia")).select(pl.count()).collect()[0, 0]
    )
    metrics["mean_count"] = round(
        edgelist.select(pl.col("count")).mean().collect()[0, 0], 2
    )

    upia_degree = edgelist.group_by(pl.col("upia")).agg(pl.n_unique("upib"))
    metrics["upia_degree_mean"] = round(upia_degree.mean().collect()[0, 1], 2)
    metrics["upia_degree_median"] = round(upia_degree.median().collect()[0, 1], 2)

    metrics = _calculate_graph_metrics(metrics=metrics, graph=graph, edgelist=edgelist)

    return metrics


def edgelist_metrics(
    edgelist: Union[pd.DataFrame, pl.LazyFrame], graph: Optional[Graph] = None
) -> Dict[str, Union[int, float]]:
    """Compute edgelist metrics.

    A simple function that computes a dictionary of basic metrics
    from an edge list (pd.DataFrame).

    :param edgelist: the edge list (pd.DataFrame)
    :param graph: optionally add the graph instance that corresponds to the
                  edgelist (to not have to re-compute it)
    :returns: a dictionary of metrics
    :rtype: Dict[str, Union[int, float]]
    :raises TypeError: if edgelist is not either pd.DataFrame or pl.LazyFrame
    """
    if isinstance(edgelist, pd.DataFrame):
        logger.debug("Compputing edgelist metrics where edgelist type is pd.DataFrame")
        return _edgelist_metrics_pandas_data_frame(edgelist=edgelist, graph=graph)

    if isinstance(edgelist, pl.LazyFrame):
        logger.debug("Computing edgelist metrics where edgelist type is pl.LazyFrame")
        return _edgelist_metrics_lazy_frame(edgelist=edgelist, graph=graph)

    raise TypeError("edgelist was not of type `pd.DataFrame` or `pl.LazyFrame")


def _update_edgelist_membership_data_frame(
    edgelist: pd.DataFrame,
    graph: Optional[Graph] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    logger.debug("Updating membership in edge list with %i rows", edgelist.shape[0])

    if prefix is None:
        prefix = DEFAULT_COMPONENT_PREFIX

    if "component" in edgelist.columns:
        logger.info("The input edge list already contain a component column")

    if not graph:
        graph = Graph.from_edgelist(
            edgelist=edgelist,
            add_marker_counts=False,
            simplify=False,
            use_full_bipartite=True,
        )

    logger.debug("Fetching connected components")
    connected_components = graph.connected_components()
    logger.debug(
        "Got the connected components. "
        "Will begin the iteration to updated edge memberships"
    )

    membership = np.empty(edgelist.shape[0], dtype=object)
    component_id_format = f"{prefix}{{:0{DIGITS}d}}"
    for i, component in enumerate(connected_components):
        component_id = component_id_format.format(i)
        edges = [
            e.index
            for e in graph.es.select_within({v.index for v in component.vertices()})
        ]
        membership[edges] = component_id
    edgelist = edgelist.assign(component=membership)

    logger.debug("Membership in edge list updated")
    return edgelist


def _update_edgelist_membership_lazy_frame(
    edgelist: pl.LazyFrame,
    graph: Optional[Graph] = None,
    prefix: Optional[str] = None,
) -> pl.LazyFrame:
    if prefix is None:
        prefix = DEFAULT_COMPONENT_PREFIX

    if "component" in edgelist.columns:
        logger.info("The input edge list already contains a component column")

    if not graph:
        graph = Graph.from_edgelist(
            edgelist=edgelist,
            add_marker_counts=False,
            simplify=False,
            use_full_bipartite=True,
        )

    logger.debug("Searching for connected components")
    connected_components = graph.connected_components()

    logger.debug("Building edge to component mappings")
    edge_index_to_component_mapping = {
        e.index: component_idx
        for component_idx, component in enumerate(connected_components)
        for e in graph.es.select_within({v.index for v in component.vertices()})
    }

    def _map_edge_index_to_component_id():
        return pl.col("edge_index").map_dict(edge_index_to_component_mapping)

    def _build_component_name_str():
        return pl.format(
            "{}{}",
            pl.lit(prefix),
            pl.col("component_index")
            .cast(pl.Utf8)
            .str.pad_start(length=DIGITS, fill_char="0"),
        )

    logger.debug("Mapping components on the edge list")
    edgelist_with_component_info = (
        edgelist.with_row_count(name="edge_index")
        .with_columns(_map_edge_index_to_component_id().alias("component_index"))
        .with_columns(_build_component_name_str().alias("component"))
    )
    edgelist_with_component_info = edgelist_with_component_info.drop(
        columns=["edge_index", "component_index"]
    )
    return edgelist_with_component_info


def update_edgelist_membership(
    edgelist: Union[pd.DataFrame, pl.LazyFrame],
    graph: Optional[Graph] = None,
    prefix: Optional[str] = None,
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """Update the edgelist with component names.

    Compute the connected components of the graph represented
    by the `edgelist` (or directly on the `graph` if provided).
    These connected components are assigned a numeric id. These
    id's are added as a `component` column in the `edgelist`.
    If a component column already exists, this will be updated.

    The name of each component will be determined in the following way:
    `prefix`+[up to 7 0's of padding]+[component number].

    :param edgelist: the edge list
    :param graph: optionally, the graph the corresponding to the edgelist
                  if you have already computed this. This is convenient to
                  avoid graph recomputation when it is not necessary.
                  It is important that you know that the edgelist and the graph
                  are in-sync if you use this feature.
    :param prefix: the prefix to prepend to the component ids, if None will
                    use `DEFAULT_COMPONENT_PREFIX`
    :returns: the updated edge list
    :rtype: Union[pd.DataFrame, pl.LazyFrame]
    :raises TypeError: if edgelist is not either pd.DataFrame or pl.LazyFrame
    """
    if isinstance(edgelist, pd.DataFrame):
        logger.debug("Updating edgelist where type is pd.DataFrame")
        return _update_edgelist_membership_data_frame(
            edgelist=edgelist, graph=graph, prefix=prefix
        )

    if isinstance(edgelist, pl.LazyFrame):
        logger.debug("Updating edgelist where type is pl.LazyFrame")
        return _update_edgelist_membership_lazy_frame(
            edgelist=edgelist, graph=graph, prefix=prefix
        )

    raise TypeError("edgelist was not of type pd.DataFrame or pl.LazyFrame")
