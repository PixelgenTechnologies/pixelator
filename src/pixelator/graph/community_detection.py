"""Functions related to perform community detection on the pixelator graph step.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from time import time
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from graspologic.partition import leiden

from pixelator.graph.constants import (
    LEIDEN_RESOLUTION,
    MIN_PIXELS_TO_REFINE,
    STRONG_EDGE_THRESHOLD,
)
from pixelator.graph.utils import (
    edgelist_metrics,
    map_upis_to_components,
    split_remaining_and_removed_edgelist,
    update_edgelist_membership,
)
from pixelator.report.models.graph import GraphSampleReport
from pixelator.types import PathType

logger = logging.getLogger(__name__)


def connect_components(
    input: str,
    output: str,
    sample_name: str,
    metrics_file: str,
    multiplet_recovery: bool,
    max_refinement_recursion_depth: int = 5,
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
        recovered_node_component_map, component_refinement_history = (
            recover_technical_multiplets(
                edgelist=edges,
                node_component_map=node_component_map.astype(np.int64),
                max_refinement_recursion_depth=max_refinement_recursion_depth,
                removed_edges_edgelist_file=Path(output)
                / f"{sample_name}.discarded_edgelist.parquet",
            )
        )

        # save the recovered components info to a file
        component_refinement_history.to_csv(
            Path(output) / f"{sample_name}.components_recovered.csv"
        )
    else:
        recovered_node_component_map = node_component_map

    del edges

    # assign component column to edge list
    edgelist_with_component_info = map_upis_to_components(
        edgelist=edgelist,
        node_component_map=recovered_node_component_map.astype(np.int64),
    )
    remaining_edgelist, removed_edgelist = split_remaining_and_removed_edgelist(
        edgelist_with_component_info
    )

    # save the edge list (discarded)
    logger.debug("Save discarded edge list")
    removed_edgelist.collect().write_parquet(
        Path(output) / f"{sample_name}.discarded_edgelist.parquet"
    )

    # save the edge list (recovered)
    logger.debug("Save the edgelist")
    remaining_edgelist.collect().write_parquet(
        Path(output) / f"{sample_name}.edgelist.parquet",
        compression="zstd",
    )

    logger.debug("Generate graph report")
    result_metrics = edgelist_metrics(remaining_edgelist)

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
    to the nodes in the connected strongly connected communities. If n_edges is
    None, the split communities are not considered for merging.

    :param edgelist: The edge list to process
    :param node_community_dict: A dictionary with the community mapping for each node
    :param n_edges: The threshold for the number of edges between communities
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
    removed_edges_edgelist_file: Optional[PathType] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Perform component recovery by deleting spurious edges.

    The molecular pixelation assay may under some conditions introduce spurious
    false edges. This creates components that are made up of two or more internally
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
    :param node_component_map: A series with the component mapping for each node
    :param max_refinement_recursion_depth: The number of times a component can be broken down
                            into smaller components during the recovery process.
    :return: A tuple with the updated node component map the history of component
             breakdowns.
    :rtype: Tuple[pd.Series, pd.DataFrame]
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
    community_annotation_history = []
    to_be_refined_next = comp_sizes[comp_sizes > MIN_PIXELS_TO_REFINE].index
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

            component_nodes = list(
                set(component_edgelist["upia"]).union(set(component_edgelist["upib"]))
            )

            edgelist_tuple = list(
                map(tuple, np.array(component_edgelist[["upia", "upib", "len"]]))
            )

            # run the leiden algorithm to get the communities
            community_dict = leiden(
                edgelist_tuple,
                resolution=LEIDEN_RESOLUTION
                if depth > 0
                else 1.0,  # Higher initial resolution to break up the mega-cluster
                random_seed=42,
                trials=5,
            )

            component_edgelist, community_serie = merge_strongly_connected_communities(
                component_edgelist,
                community_dict,
                n_edges=STRONG_EDGE_THRESHOLD if depth > 0 else None,
            )

            if community_serie.nunique() == 1:
                continue
            n_edges_to_remove += (
                component_edgelist["upia_community"]
                != component_edgelist["upib_community"]
            ).sum()
            community_size_map = community_serie.groupby(community_serie).count()

            if (community_size_map > MIN_PIXELS_TO_REFINE).sum() > 1:
                further_refinement = True
            else:
                further_refinement = False

            for new_community in community_size_map.index:
                new_id = next(id_gen)
                node_component_map[
                    community_serie[community_serie == new_community].index
                ] = new_id
                community_annotation_history.append(
                    (
                        component,
                        new_id,
                        len(component_nodes),
                        community_size_map[new_community],
                        depth,
                    )
                )
                if (
                    further_refinement
                    and community_size_map[new_community] > MIN_PIXELS_TO_REFINE
                ):
                    to_be_refined_next.append(new_id)

    component_refinement_history = pd.DataFrame(
        community_annotation_history,
        columns=["old", "new", "old_size", "new_size", "depth"],
    ).astype(int)

    logger.info(
        "Obtained %i components after removing %i edges",
        node_component_map.nunique(),
        n_edges_to_remove,
    )
    return node_component_map, component_refinement_history


def write_recovered_components(
    recovered_components: pd.DataFrame, filename: PathType
) -> None:
    """Help to write the recovered component info to a CSV file.

    A helper function that writes to a CSV file the information
    of the recovered components that is an edgelist between old
    component annotations and new ones resulting from multiplet
    recovery.

    :returns: None
    :rtype: None
    """
    logger.debug("Saving recovered components to %s", filename)
    recovered_components.to_csv(filename, index=True)
    logger.debug("Recovered components saved")
