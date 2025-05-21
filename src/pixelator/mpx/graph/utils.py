"""Module contains various useful graph functions.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
import typing
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import pandas as pd
import polars as pl
import xxhash
from scipy.sparse.linalg import matrix_power

from pixelator.common.graph import Graph
from pixelator.common.graph.backends.implementations import (
    NetworkXGraphBackend,
)
from pixelator.common.report.models import SummaryStatistics

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
        a_pixels = group_df["upia"].nunique()
        b_pixels = group_df["upib"].nunique()
        pixels = a_pixels + b_pixels
        antibodies = group_df["marker"].nunique()
        molecules = group_df.shape[0]

        reads = group_df["count"].sum()
        mean_reads_per_molecule = group_df["count"].mean()
        median_reads_per_molecule = group_df["count"].median()

        # Please note that we need to use observed=True
        # here upia is a categorical column, and since not
        # all values are present in all components, this is
        # required to get a correct value.
        b_pixels_per_a_pixel_series = group_df.groupby("upia", observed=True)[
            "upib"
        ].nunique()
        mean_b_pixels_per_a_pixel = b_pixels_per_a_pixel_series.mean()
        median_b_pixels_per_a_pixel = b_pixels_per_a_pixel_series.median()

        a_pixels_per_b_pixel = group_df.groupby("upib", observed=True)["upia"].nunique()
        mean_a_pixels_per_b_pixel = a_pixels_per_b_pixel.mean()
        median_a_pixels_per_b_pixel = a_pixels_per_b_pixel.median()

        # Same reasoning as above
        molecule_count_per_a_pixel_series = group_df.groupby("upia", observed=True)[
            "umi"
        ].count()
        mean_molecules_per_a_pixel = molecule_count_per_a_pixel_series.mean()
        median_molecules_per_a_pixel = molecule_count_per_a_pixel_series.median()

        a_pixel_b_pixel_ratio = a_pixels / b_pixels

        cmetrics.append(
            (
                pixels,
                a_pixels,
                b_pixels,
                antibodies,
                molecules,
                reads,
                mean_reads_per_molecule,
                median_reads_per_molecule,
                mean_b_pixels_per_a_pixel,
                median_b_pixels_per_a_pixel,
                mean_a_pixels_per_b_pixel,
                median_a_pixels_per_b_pixel,
                a_pixel_b_pixel_ratio,
                mean_molecules_per_a_pixel,
                median_molecules_per_a_pixel,
            )
        )
        index.append(component_id)

    # create components metrics data frame
    components_metrics = pd.DataFrame(
        index=pd.Index(index, name="component"),
        columns=[
            "pixels",
            "a_pixels",
            "b_pixels",
            "antibodies",
            "molecules",
            "reads",
            "mean_reads_per_molecule",
            "median_reads_per_molecule",
            "mean_b_pixels_per_a_pixel",
            "median_b_pixels_per_a_pixel",
            "mean_a_pixels_per_b_pixel",
            "median_a_pixels_per_b_pixel",
            "a_pixel_b_pixel_ratio",
            "mean_molecules_per_a_pixel",
            "median_molecules_per_a_pixel",
        ],
        data=cmetrics,
    )

    logger.debug("Component metrics computed")
    return components_metrics


def _get_extended_adjacency(graph: Graph, k: int = 0):
    A = graph.get_adjacency_sparse()
    A.setdiag(1, k=0)
    An = (matrix_power(A, k) > 0).astype(int)
    return An


def _get_neighborhood_counts(
    node_marker_counts,
    graph,
    k: int = 0,
    normalization: Optional[Literal["mean"]] = None,
):
    An = _get_extended_adjacency(graph, k=k)
    neighbourhood_counts = An @ node_marker_counts

    # TODO Optionally add more methods here
    if normalization == "mean":
        nbr_of_neighbors_per_node = An.sum(axis=1)
        # Reshape to ensure broadcasting compatibility
        nbr_of_neighbors_per_node = nbr_of_neighbors_per_node.reshape((-1, 1))
        neighbourhood_counts = neighbourhood_counts / nbr_of_neighbors_per_node

    df = pd.DataFrame(
        data=neighbourhood_counts,
        columns=node_marker_counts.columns.copy(),
        index=node_marker_counts.index.copy(),
    )
    df.columns.name = "markers"
    df.index.name = "node"

    return df


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
    """
    if k == 0 and normalization:
        warnings.warn(
            (
                f"Using `normalization={normalization}` when k=0 "
                "has no effect, since no neighborhood is created."
            )
        )

    node_marker_counts = graph.node_marker_counts

    if k == 0:
        return node_marker_counts

    neighborhood_counts = _get_neighborhood_counts(
        node_marker_counts=node_marker_counts,
        graph=graph,
        k=k,
        normalization=normalization,
    )
    return neighborhood_counts


class EdgelistMetrics(typing.TypedDict, total=True):
    """TypedDict for edgelist metrics."""

    component_count: int
    molecule_count: int
    marker_count: int
    read_count: int
    a_pixel_count: int
    b_pixel_count: int

    read_count_per_molecule_stats: SummaryStatistics

    fraction_molecules_in_largest_component: float
    fraction_pixels_in_largest_component: float

    edges_with_colliding_upi_count: int
    edges_removed_in_multiplet_recovery_first_iteration: int
    edges_removed_in_multiplet_recovery_refinement: int
    fraction_edges_removed_in_refinement: float


MetricsDict = typing.TypeVar(
    "MetricsDict", Dict[str, Union[int, float]], EdgelistMetrics
)


def _edgelist_metrics_lazyframe(edgelist: pl.LazyFrame) -> EdgelistMetrics:
    metrics: EdgelistMetrics = {}  # type: ignore

    unique_counts = edgelist.select(
        pl.col("upia").n_unique(),
        pl.col("upib").n_unique(),
        pl.col("marker").n_unique(),
    ).collect()

    metrics["a_pixel_count"] = int(unique_counts["upia"][0])
    metrics["b_pixel_count"] = int(unique_counts["upib"][0])
    metrics["marker_count"] = int(unique_counts["marker"][0])
    # Note that we get upi here and count that, because otherwise just calling count
    # here confuses polars since there is a column with that name.
    metrics["molecule_count"] = int(
        edgelist.select(pl.col("upia").count()).collect()["upia"][0]
    )

    counts_per_molecule = edgelist.select(pl.col("count")).collect()["count"]
    metrics["read_count"] = int(counts_per_molecule.sum())
    metrics["read_count_per_molecule_stats"] = SummaryStatistics.from_series(
        counts_per_molecule
    )

    component_stats = (
        edgelist.group_by("component")
        .agg(
            pl.col("upia").n_unique().alias("n_upia"),
            pl.col("upib").n_unique().alias("n_upib"),
            pl.len().alias("n_molecule"),
        )
        .collect()
    )
    metrics["component_count"] = component_stats.shape[0]
    component_stats = component_stats.with_columns(
        n_upi=pl.col("n_upia") + pl.col("n_upib")
    )
    largest_component = component_stats.filter(
        pl.col("n_molecule") == component_stats["n_molecule"].max()
    )[0]
    metrics["fraction_molecules_in_largest_component"] = (
        largest_component["n_molecule"][0] / component_stats["n_molecule"].sum()
    )
    metrics["fraction_pixels_in_largest_component"] = (
        largest_component["n_upi"][0] / component_stats["n_upi"].sum()
    )
    return metrics


def edgelist_metrics(
    edgelist: pl.DataFrame | pd.DataFrame | pl.LazyFrame,
) -> EdgelistMetrics:
    """Compute edgelist metrics.

    A simple function that computes a dictionary of basic metrics
    from an edge list (pl.DataFrame).

    :param edgelist: the edge list (pl.DataFrame)
    :param graph: optionally add the graph instance that corresponds to the
                  edgelist (to not have to re-compute it)
    :returns: a dataclass of metrics
    :rtype: EdgelistMetrics
    :raises TypeError: if edgelist is not either a pl.LazyFrame
    , pl.DataFrame or a pd.DataFrame
    """
    if isinstance(edgelist, pl.LazyFrame):
        logger.debug("Computing edgelist metrics where edgelist type is pl.LazyFrame")
        return _edgelist_metrics_lazyframe(edgelist)

    if isinstance(edgelist, pd.DataFrame):
        logger.debug("Computing edgelist metrics where edgelist type is pd.DataFrame")
        edgelist = pl.from_pandas(edgelist)

    if isinstance(edgelist, pl.DataFrame):
        metrics: EdgelistMetrics = {}  # type: ignore

        metrics["a_pixel_count"] = edgelist.n_unique("upia")
        metrics["b_pixel_count"] = edgelist.n_unique("upib")
        metrics["marker_count"] = edgelist.n_unique("marker")
        metrics["molecule_count"] = edgelist.shape[0]

        counts_per_molecule = edgelist["count"]
        metrics["read_count"] = int(counts_per_molecule.sum())
        metrics["read_count_per_molecule_stats"] = SummaryStatistics.from_series(
            counts_per_molecule
        )

        component_stats = edgelist.group_by("component").agg(
            pl.col("upia").n_unique().alias("n_upia"),
            pl.col("upib").n_unique().alias("n_upib"),
            pl.len().alias("n_molecule"),
        )
        metrics["component_count"] = component_stats.shape[0]
        component_stats = component_stats.with_columns(
            n_upi=pl.col("n_upia") + pl.col("n_upib")
        )
        largest_component = component_stats.filter(
            pl.col("n_molecule") == component_stats["n_molecule"].max()
        )[0]
        metrics["fraction_molecules_in_largest_component"] = (
            largest_component["n_molecule"][0] / component_stats["n_molecule"].sum()
        )
        metrics["fraction_pixels_in_largest_component"] = (
            largest_component["n_upi"][0] / component_stats["n_upi"].sum()
        )
        return metrics
    raise TypeError(
        "edgelist was not of type `pd.LazyFrame`, `pd.DataFrame` or `pl.DataFrame"
    )


def map_upis_to_components(
    edgelist: pl.LazyFrame,
    node_component_map: pd.Series,
    node_depth_map: Optional[pd.Series] = None,
) -> pl.LazyFrame:
    """Update the edgelist with component names corresponding to upia/upib.

    Using the node_component_map, this function will add the component
    information to the edgelist. Two columns are added to the edgelist,
    i.e. component_a and component_b respectively for upia and upib.
    The component names are then determined by calculating a hash of
    the nodes in that component.

    :param edgelist: the edge list
    :param node_component_map: a pd.Series mapping the nodes to their components
    :returns: the remaining_edgelist and the removed_edgelist
    :rtype: pl.LazyFrame
    :raises TypeError: if edgelist is not either a pl.LazyFrame or a pd.DataFrame
    """
    # Create a mapping of the components to a hash of its UPIs
    node_component_map = node_component_map.astype(str)
    components = node_component_map.groupby(node_component_map)
    for _, comp in components:
        comp_nodes = sorted(comp.index)
        comp_hash = xxhash.xxh64()
        for node in comp_nodes:
            comp_hash.update(str(node))
        node_component_map[comp_nodes] = comp_hash.hexdigest()
    node_component_dict = node_component_map.to_dict()
    logger.debug("Mapping components on the edge list")
    edgelist_with_component_info = edgelist.with_columns(
        component_a=pl.col("upia")
        .cast(pl.String)
        .replace_strict(
            node_component_dict,
            default="",
        ),
    ).with_columns(
        component_b=pl.col("upib")
        .cast(pl.String)
        .replace_strict(
            node_component_dict,
            default="",
        ),
    )
    if node_depth_map is not None:
        node_depth_dict = node_depth_map.to_dict()
        edgelist_with_component_info = (
            edgelist_with_component_info.with_columns(
                upia_depth=pl.col("upia")
                .cast(pl.String)
                .replace_strict(node_depth_dict, default=0),
                upib_depth=pl.col("upib")
                .cast(pl.String)
                .replace_strict(node_depth_dict, default=0),
            )
            .with_columns(
                depth=pl.min_horizontal([pl.col("upia_depth"), pl.col("upib_depth")])
            )
            .drop(["upia_depth", "upib_depth"])
        )

    return edgelist_with_component_info


def update_edgelist_membership(
    edgelist: pl.LazyFrame | pd.DataFrame,
    node_component_map: Optional[pd.Series] = None,
) -> pl.LazyFrame | pd.DataFrame:
    """Update the edgelist with component names.

    Using the node_component_map, this function will add the component
    information to the edgelist. If for an edge, components of UPIA and
    UPIB do not match, that edge will be removed. If node_component_map
    is missing, it will be constructed based on the connected component
    in the graph made from the edgelist.

    :param edgelist: the edge list
    :param node_component_map: a pd.Series mapping the nodes to their components
    if missing, it will be constructed based on the connected components in the
    graph made from the edgelist.
    :returns: the remaining_edgelist and the removed_edgelist
    :rtype: pl.LazyFrame | pd.DataFrame
    :raises TypeError: if edgelist is not either a pl.LazyFrame or a pd.DataFrame
    """
    if isinstance(edgelist, pd.DataFrame):
        was_dataframe = True
        edgelist = pl.LazyFrame(edgelist)
    else:
        was_dataframe = False

    if node_component_map is None:
        edges = (
            edgelist.select(["upia", "upib"])
            .group_by(["upia", "upib"])
            .len()
            .sort(["upia", "upib"])  # sort to make sure the graph is the same
            .collect()
        )
        graph = nx.from_edgelist(edges.select(["upia", "upib"]).iter_rows())

        node_component_map = pd.Series(index=graph.nodes())
        for i, cc in enumerate(nx.connected_components(graph)):
            node_component_map[list(cc)] = i
        del graph

    if isinstance(edgelist, pl.LazyFrame):
        logger.debug("Updating edgelist where type is pl.LazyFrame")
        if "component" in edgelist.collect_schema().names():
            logger.info("The input edge list already contains a component column")

        edgelist_with_component_info = map_upis_to_components(
            edgelist, node_component_map
        )
        edgelist_with_component_info, _ = split_remaining_and_removed_edgelist(
            edgelist_with_component_info
        )
    else:
        raise TypeError("edgelist was not of type pl.LazyFrame or pd.DataFrame")

    if was_dataframe:
        return edgelist_with_component_info.collect().to_pandas()
    else:
        return edgelist_with_component_info


def split_remaining_and_removed_edgelist(
    edgelist: pl.LazyFrame,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split the edgelist into remaining and removed.

    Inputs an edgelist with component_a and component_b columns.
    If the two columns match in a row and they are not invalid,
    i.e. empty string. They will be added to the remaining_edgelist.
    Otherwise they will be added to the removed_edgelist.

    :param edgelist: the edge list
    :returns: the remaining_edgelist and the removed_edgelist
    """
    if "component" in edgelist.collect_schema().names():
        logger.info("The input edge list already contains a component column")
        edgelist = edgelist.drop("component")

    remaining_edgelist = (
        edgelist.filter(pl.col("component_a") != "")
        .filter(pl.col("component_a") == pl.col("component_b"))
        .rename({"component_a": "component"})
        .drop("component_b")
    )
    columns = remaining_edgelist.collect_schema().names()
    if "depth" in columns:
        remaining_edgelist = remaining_edgelist.drop("depth")

    removed_edgelist = edgelist.filter(
        (pl.col("component_a") == "") | (pl.col("component_a") != pl.col("component_b"))
    )
    return remaining_edgelist, removed_edgelist
