"""Functions related to perform community detection on the pixelator graph step.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from copy import copy
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from graspologic_native import leiden

from pixelator.mpx.graph.constants import (
    LEIDEN_RESOLUTION,
    MIN_PIXELS_TO_REFINE,
)
from pixelator.mpx.graph.utils import (
    edgelist_metrics,
    map_upis_to_components,
    split_remaining_and_removed_edgelist,
)
from pixelator.mpx.report.models.graph import GraphSampleReport

logger = logging.getLogger(__name__)


def connect_components(
    input: str,
    output: str,
    sample_name: str,
    metrics_file: str,
    multiplet_recovery: bool,
    max_refinement_recursion_depth: int = 5,
    max_edges_to_split: int = 5,
    min_count: int = 2,
) -> None:
    """Retrieve all connected components from an edgelist.

    This function takes as input an edge list in `parquet` format that has
    been generated with `pixelator collapse`. The function filters the
    edge list by count (`min_count`) and then adds a column to the edge list
    with the respective connected components ids obtained from the graph. The
    column is named "component". The edge list is then processed to recover
    big components (technical multiplets) into smaller components if only if
    `multiplet_recovery` is True. The recovery is done using community
    detection to detect and remove problematic edges using the Leiden [1]_ community
    detection algorithm. Information about the recovered components is written to
    a CSV file (`<sample_id>.components_recovered.csv`) and the filtered edge list
    is written to a parquet file (`<sample_id>`.edgelist.parquet) alongside the
    discarded edges (`<sample_id>`.discarded_edgelist.parquet).

    The following files are generated:

    - edge list (parquet) after multiplets recovery (if any)
    - metrics (json) with information


    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s4q:598-019-41695-z

    :param input: the path to the edge list dataframe (parquet)
    :param output: the path to the output folder
    :param sample_name: the prefix to prepend to the files (sample name)
    :param metrics_file: the path to a JSON file to write metrics
    :param multiplet_recovery: set to true to activate multiplet recovery
    :param max_refinement_recursion_depth: The number of times a component can be broken down into
                             smaller components during the recovery process.
    :param max_edges_to_split: The maximum number of edges between the product components
                                when splitting during multiplet recovery.
    :param min_count: the minimum number of counts (molecules) an edge must have
    :returns: None
    :rtype: None
    :raises RuntimeError: if the edge list is empty after filtering
    """
    logger.debug("Parsing edge list %s", input)

    # load data (edge list in data frame format)
    edgelist = pl.scan_parquet(input, low_memory=True)

    nbr_of_rows = edgelist.select(pl.count()).collect()[0, 0]
    # filter data by count
    if min_count > 1:
        logger.debug(
            "Filtering edge list with %i rows using %i as minimum count",
            nbr_of_rows,
            min_count,
        )
        edgelist = edgelist.filter(pl.col("count") >= min_count)
        nbr_of_rows = edgelist.select(pl.count()).collect()[0, 0]
        logger.debug("Filtered edge list has %i elements", nbr_of_rows)

    if nbr_of_rows == 0:
        raise RuntimeError(
            f"The edge list has 0 elements after filtering by %{min_count}"
        )

    # check if the are problematic edges (same upib and upia)
    problematic_edges = set(
        np.intersect1d(
            edgelist.select("upib").unique().collect().to_numpy(),
            edgelist.select("upia").unique().collect().to_numpy(),
        )
    )
    if len(problematic_edges) > 0:
        logger.warning(
            "The edge list has %i intersecting UPIA and UPIB, these will be removed",
            len(problematic_edges),
        )
        edgelist_no_prob = edgelist.filter(
            (~pl.col("upib").is_in(problematic_edges))
            & (~pl.col("upia").is_in(problematic_edges))
        )
    else:
        edgelist_no_prob = edgelist

    nbr_of_rows = edgelist_no_prob.select(pl.count()).collect()[0, 0]
    if nbr_of_rows == 0:
        raise RuntimeError(
            "The edge list has 0 elements after removing problematic edges"
        )

    edges = (
        edgelist_no_prob.select(["upia", "upib"])
        .group_by(["upia", "upib"])
        .len()
        .sort(["upia", "upib"])  # sort to make sure the graph is the same
        .collect()
        .to_pandas()
    )
    graph = nx.from_pandas_edgelist(edges, source="upia", target="upib")

    node_component_map = pd.Series(index=graph.nodes())
    for i, cc in enumerate(nx.connected_components(graph)):
        node_component_map[list(cc)] = i
    del graph

    # get raw metrics before multiplets recovery
    logger.debug("Calculating raw edgelist metrics")

    if multiplet_recovery:
        recovered_node_component_map, node_depth_map = recover_technical_multiplets(
            edgelist=edges,
            node_component_map=node_component_map.astype(np.int64),
            max_refinement_recursion_depth=max_refinement_recursion_depth,
            max_edges_to_split=max_edges_to_split,
        )
    else:
        recovered_node_component_map = node_component_map
        node_depth_map = pd.Series(index=node_component_map.index, data=0)

    del edges

    # assign component column to edge list
    edgelist_with_component_info = map_upis_to_components(
        edgelist=edgelist,
        node_component_map=recovered_node_component_map.astype(np.int64),
        node_depth_map=node_depth_map.astype(np.int64),
    )
    remaining_edgelist, removed_edgelist = split_remaining_and_removed_edgelist(
        edgelist_with_component_info
    )

    # save the edge list (discarded)
    logger.debug("Save discarded edge list")
    removed_edgelist_df = removed_edgelist.collect()
    removed_edgelist_df.write_parquet(
        Path(output) / f"{sample_name}.discarded_edgelist.parquet"
    )

    # save the edge list (recovered)
    logger.debug("Save the edgelist")
    graph_output_edgelist = remaining_edgelist.collect()
    graph_output_edgelist.write_parquet(
        Path(output) / f"{sample_name}.edgelist.parquet",
        compression="zstd",
    )

    logger.debug("Generate graph report")
    result_metrics = edgelist_metrics(graph_output_edgelist)

    result_metrics["edges_with_colliding_upi_count"] = int(
        (removed_edgelist_df["depth"] == 0).sum()
    )
    result_metrics["edges_removed_in_multiplet_recovery_first_iteration"] = int(
        (removed_edgelist_df["depth"] == 1).sum()
    )
    result_metrics["edges_removed_in_multiplet_recovery_refinement"] = int(
        (removed_edgelist_df["depth"] > 1).sum()
    )
    result_metrics["fraction_edges_removed_in_refinement"] = float(
        (removed_edgelist_df["depth"] > 1).sum() / max(len(removed_edgelist_df), 1)
    )

    del graph_output_edgelist
    del removed_edgelist_df

    report = GraphSampleReport(
        sample_id=sample_name,
        **result_metrics,
    )
    report.write_json_file(Path(metrics_file), indent=4)


def merge_strongly_connected_communities(
    edgelist: pd.DataFrame,
    node_community_dict: dict,
    n_edges: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Merge strongly connected communities in an edge list.

    This function takes an edge list and a dictionary with the community mapping
    for each node. It then computes the number of edges between communities and
    if they are higher than the given n_edges. It assigns the same community id
    to the nodes in the connected strongly connected communities. If `n_edges` is
    None, the split communities are not considered for merging.

    :param edgelist: The edge list to process
    :param node_community_dict: A dictionary with the community mapping for each node
    :param n_edges: The threshold for the number of edges to be found between communities to merge or None to avoid merging
    :returns: A tuple with the modified edge list and the updated community mapping
    """
    community_serie = pd.Series(node_community_dict)
    edgelist["upia_community"] = community_serie[edgelist["upia"]].values
    edgelist["upib_community"] = community_serie[edgelist["upib"]].values

    if n_edges is None or (community_serie.nunique() == 1):
        return edgelist, community_serie

    edge_counts = (
        edgelist.groupby(["upia_community", "upib_community"])["upia"]
        .count()
        .unstack(fill_value=0)
    )
    edge_counts = edge_counts.add(edge_counts.T, fill_value=0)
    cross_community_edges = edge_counts.where(
        np.tril(np.ones(edge_counts.shape), k=-1).astype(bool)
    ).stack()
    connected_communities = cross_community_edges[cross_community_edges > n_edges].index
    communities_graph = nx.from_edgelist(connected_communities)
    for cc in nx.connected_components(communities_graph):
        new_tag = min(cc)
        community_serie[community_serie.isin(cc)] = new_tag
        edgelist.loc[edgelist["upia_community"].isin(cc), "upia_community"] = new_tag
        edgelist.loc[edgelist["upib_community"].isin(cc), "upib_community"] = new_tag
    return edgelist, community_serie


def recover_technical_multiplets(
    edgelist: pd.DataFrame,
    node_component_map: pd.Series,
    max_refinement_recursion_depth: int = 5,
    max_edges_to_split: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Perform component recovery by deleting spurious edges.

    The molecular pixelation assay may under some conditions introduce spurious
    edges. This creates components that are made up of two or more internally
    well-connected graphs, that are connected to each other by a low number of edges
    (typically in the single digits). We call these components technical multiplets.
    This method will attempt to break up these components by performing community
    detection on the whole graph using the Leiden algorithm [1]_.

    The community detection algorithm will attempt to find communities (i.e. subgraphs)
    that are internally well-connected. Edges between these communities are computed
    and consequently removed from the edge list to create new components that are
    internally well connected, and that should represent real cells.


    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s4q:598-019-41695-z

    :param edgelist: The edge list used to create the graph
    :param node_component_map: A series with the component mapping for each
                            node where the index is the upi (i.e. node name)
                            and the value is the component id.
    :param max_refinement_recursion_depth: The number of times a component can be broken down
                            into smaller components during the recovery process.
    :param max_edges_to_split: The maximum number of edges between the product components
                            when splitting during multiplet recovery.
    :return: A tuple with the updated node component map and the iteration depth at which each
             node is re-assigned to a component.
    :rtype: Tuple[pd.Series, pd.Series]
    """
    logger.debug(
        "Starting multiplets recovery in edge list with %i rows",
        edgelist.shape[0],
    )

    def id_generator(start=0):
        next_id = start
        while True:
            yield next_id
            next_id += 1

    id_gen = id_generator(node_component_map.max() + 1)
    comp_sizes = node_component_map.groupby(node_component_map).count()

    n_edges_to_remove = 0
    to_be_refined_next = comp_sizes[comp_sizes > MIN_PIXELS_TO_REFINE].index
    node_depth_map = pd.Series(index=node_component_map.index, data=0)
    for depth in range(max_refinement_recursion_depth):
        edgelist["component_a"] = node_component_map[edgelist["upia"]].values
        edgelist["component_b"] = node_component_map[edgelist["upib"]].values
        component_groups = edgelist.groupby("component_a")
        to_be_refined = copy(to_be_refined_next)
        to_be_refined_next = []
        for component, component_edgelist in component_groups:
            if component not in to_be_refined:
                continue

            component_edgelist = component_edgelist[
                component_edgelist["component_b"] == component
            ].sort_values(["upia", "upib"])

            edgelist_tuple = list(
                map(tuple, np.array(component_edgelist[["upia", "upib", "len"]]))
            )

            # run the leiden algorithm to get the communities
            _, community_dict = leiden(
                edgelist_tuple,
                resolution=LEIDEN_RESOLUTION
                if depth > 0
                else 1.0,  # Higher initial resolution to break up the mega-cluster
                seed=42,
                trials=5,
                # These parameters are used to sync up the native implementation with
                # the python implementation we originally used.
                iterations=1,
                randomness=0.001,
                use_modularity=True,
                starting_communities=None,
            )

            component_edgelist, community_serie = merge_strongly_connected_communities(
                component_edgelist,
                community_dict,
                n_edges=max_edges_to_split if depth > 0 else None,
            )

            if community_serie.nunique() == 1:
                continue
            n_edges_to_remove += (
                component_edgelist["upia_community"]
                != component_edgelist["upib_community"]
            ).sum()
            community_size_map = community_serie.groupby(community_serie).count()
            node_depth_map[community_serie.index] = depth + 1

            if (community_size_map > MIN_PIXELS_TO_REFINE).sum() > 1:
                further_refinement = True
            else:
                further_refinement = False

            for new_community in community_size_map.index:
                new_id = next(id_gen)
                node_component_map[
                    community_serie[community_serie == new_community].index
                ] = new_id
                if (
                    further_refinement
                    and community_size_map[new_community] > MIN_PIXELS_TO_REFINE
                ):
                    to_be_refined_next.append(new_id)

    logger.info(
        "Obtained %i components after removing %i edges",
        node_component_map.nunique(),
        n_edges_to_remove,
    )
    return node_component_map, node_depth_map
