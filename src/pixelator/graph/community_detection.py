"""Functions related to perform community detection on the pixelator graph step.

Copyright © 2022 Pixelgen Technologies AB.
"""

import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from graspologic.cluster import leiden

from pixelator.graph.constants import (
    DEFAULT_COMPONENT_PREFIX,
    DEFAULT_COMPONENT_PREFIX_RECOVERY,
    MIN_PIXELS_TO_REFINE,
    STRONG_EDGE_THRESHOLD,
)
from pixelator.graph.graph import Graph
from pixelator.graph.utils import (
    edgelist_metrics,
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
    refinement_depth: int = 10,
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
    :param refinement_depth: The number of times a component can be broken down into
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
            edgelist.select("upib").collect().to_numpy(),
            edgelist.select("upia").collect().to_numpy(),
        )
    )
    if len(problematic_edges) > 0:
        logger.warning(
            "The edge list has %i intersecting UPIA and UPIB, these will be removed",
            len(problematic_edges),
        )
        edgelist = edgelist.filter(
            (~pl.col("upib").is_in(problematic_edges))
            & (~pl.col("upia").is_in(problematic_edges))
        )

    nbr_of_rows = edgelist.select(pl.count()).collect()[0, 0]
    if nbr_of_rows == 0:
        raise RuntimeError(
            "The edge list has 0 elements after removing problematic edges"
        )

    weighted_edgelist = (
        edgelist.select(["upia", "upib"])
        .group_by(["upia", "upib"])
        .len()
        .sort(["upia", "upib"])  # sort to make sure the graph is the same
        .collect()
    )
    graph = nx.Graph()
    for row in weighted_edgelist.iter_rows():
        graph.add_edge(row[0], row[1], weight=row[2])

    node_component_map = pd.Series(index=graph.nodes())
    for i, cc in enumerate(nx.connected_components(graph)):
        node_component_map[list(cc)] = i

    # assign component column to edge list
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        node_component_map=node_component_map.astype(np.int64),
        prefix=DEFAULT_COMPONENT_PREFIX,
    )

    # get raw metrics before multiplets recovery
    logger.debug("Calculating raw edgelist metrics")

    if multiplet_recovery:
        edgelist, info = recover_technical_multiplets(
            edgelist=edgelist,
            graph=graph,
            node_component_map=node_component_map.astype(np.int64),
            refinement_depth=refinement_depth,
            removed_edges_edgelist_file=Path(output)
            / f"{sample_name}.discarded_edgelist.parquet",
        )

        # Update the graph with the new edgelist after multiplet recovery
        graph = Graph.from_edgelist(
            edgelist=edgelist,
            add_marker_counts=False,
            simplify=False,
            use_full_bipartite=True,
        )

        # save the recovered components info to a file
        write_recovered_components(
            info,
            filename=Path(output) / f"{sample_name}.components_recovered.csv",
        )

    result_metrics = edgelist_metrics(edgelist, graph)
    del graph

    # save the edge list (recovered)
    logger.debug("Save the edgelist")
    edgelist.collect(streaming=True, no_optimization=True).write_parquet(
        Path(output) / f"{sample_name}.edgelist.parquet",
        compression="zstd",
    )

    report = GraphSampleReport(
        sample_id=sample_name,
        **result_metrics,
    )
    report.write_json_file(Path(metrics_file), indent=4)


def recovered_component_info(
    old_edgelist: pl.LazyFrame, new_edgelist: pl.LazyFrame
) -> Dict[str, List[str]]:
    """Map the old component naming to new component naming.

    Translate the old component memberships to the new ones
    to enable tracking, e.g. to see how many components a large
    component was broken up into.
    :param old_edgelist: the old edgelist
    :param new_edgelist: the new edgelist
    :returns: a dictionary with the old component as key and a list of
              new components as value
    :rtype: Dict[str, List[str]]
    """
    old_memberships = old_edgelist.select(pl.col("upi"), pl.col("component"))
    new_memberships = new_edgelist.select(pl.col("upi"), pl.col("component"))
    merged = old_memberships.join(new_memberships, how="left", on="upi", suffix="_new")
    return dict(
        merged.select(pl.col("component"), pl.col("component_new"))  # type: ignore
        .group_by(pl.col("component"))
        .agg(pl.col("component_new").unique().drop_nulls())
        .collect(streaming=True)
        .iter_rows()
    )


def recover_technical_multiplets(
    edgelist: pl.LazyFrame,
    graph: nx.Graph,
    node_component_map: pd.Series,
    refinement_depth: int = 10,
    removed_edges_edgelist_file: Optional[PathType] = None,
) -> Tuple[pl.LazyFrame, Dict[str, List[str]]]:
    """Perform mega-cluster recovery by deleting spurious edges.

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
    :param graph: optionally add the graph corresponding to the edgelist, this will
                 speed up the function by bypassing the graph creation step
    :param refinement_depth: The number of times a component can be broken down into
                             smaller components during the recovery process.
    :param removed_edges_edgelist_file: If not None, the edge list with the discarded
                                        edges will be saved to this file
    :return: A tuple with the modified edge list and a dictionary with the mapping of
             old component to new component ids
    :rtype: Tuple[pl.LazyFrame, Dict[str, List[str]]]
    """
    logger.debug(
        "Starting multiplets recovery in edge list with %i rows",
        edgelist.select(pl.count()).collect().item(0, 0),
    )

    # add the combined upi here as a way to make sure we can use it to
    # check edge membership for between the old and the (potential)
    # new components.
    edgelist = edgelist.with_columns(
        pl.concat_str(pl.col("upia"), pl.col("upib")).alias("upi")
    )

    def id_generator(min_id=0):
        internal_counter = min_id

        def get_id():
            nonlocal internal_counter
            internal_counter += 1
            return internal_counter

        return get_id

    def merge_strongly_connected_communities(
        graph, node_community_dict, n_edges=STRONG_EDGE_THRESHOLD
    ):
        community_serie = pd.Series(node_community_dict)
        edge_df = pd.DataFrame(graph.edges(), columns=["upia", "upib"])
        edge_df["cla"] = community_serie[edge_df["upia"]].values
        edge_df["clb"] = community_serie[edge_df["upib"]].values
        edge_counts = (
            edge_df.groupby(["cla", "clb"])["upia"].count().unstack(fill_value=0)
        )
        edge_counts = edge_counts.add(edge_counts.T, fill_value=0)
        cross_community_edges = edge_counts.where(
            np.tril(np.ones(edge_counts.shape), k=-1).astype(bool)
        ).stack()
        connected_communities = cross_community_edges[
            cross_community_edges > n_edges
        ].index
        communities_graph = nx.from_edgelist(connected_communities)
        for cc in nx.connected_components(communities_graph):
            community_serie[community_serie.isin(cc)] = min(cc)
        return community_serie

    id_gen = id_generator(node_component_map.max() + 1)

    component_refinement_list = []
    for component in node_component_map.unique():
        if (node_component_map == component).sum() > MIN_PIXELS_TO_REFINE:
            component_refinement_list.append((component, 0))

    while component_refinement_list:
        component, depth = component_refinement_list.pop(0)
        if depth >= refinement_depth:
            break

        logger.debug(
            "Processing component %i with %i pixels",
            component,
            (node_component_map == component).sum(),
        )

        # get the subgraph for the component
        component_nodes = list(
            node_component_map[node_component_map == component].index
        )
        component_graph = graph.subgraph(component_nodes)

        # run the leiden algorithm to get the communities
        community_dict = leiden(
            component_graph,
            resolution=0.01,
            seed=42,
        )
        community_serie = merge_strongly_connected_communities(
            component_graph, community_dict
        )
        if len(community_serie.unique()) == 1:
            continue
        for new_community in community_serie.unique():
            new_id = id_gen()
            node_component_map[
                community_serie[community_serie == new_community].index
            ] = new_id
            if (community_serie == new_community).sum() > MIN_PIXELS_TO_REFINE:
                component_refinement_list.append((new_id, depth + 1))

    edges_to_remove = list(
        edgelist.select(["upia", "upib", "upi"])
        .with_columns(
            pl.col("upia")
            .alias("comp_a")
            .replace_strict(
                old=list(node_component_map.index), new=node_component_map.values
            )
        )
        .with_columns(
            pl.col("upib")
            .alias("comp_b")
            .replace_strict(
                old=list(node_component_map.index), new=node_component_map.values
            )
        )
        .filter(pl.col("comp_a") != pl.col("comp_b"))
        .select("upi")
    )

    if removed_edges_edgelist_file is not None:
        logger.debug(
            "Saving edge list with discarded edges to %s", removed_edges_edgelist_file
        )
        # save the discarded edges to a file
        masked_df = edgelist.filter(pl.col("upi").is_in(edges_to_remove))
        masked_df.collect().write_parquet(
            removed_edges_edgelist_file,  # type: ignore
            compression="zstd",
        )

    logger.debug("Preparing edge list for computing new components")
    old_edgelist = edgelist.select(pl.col("upi"), pl.col("component"))

    logger.debug("Creating updated edgelist")
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        node_component_map=node_component_map,
        prefix=DEFAULT_COMPONENT_PREFIX_RECOVERY,
    )

    # get the info of recovered components ids to old ones
    info = recovered_component_info(old_edgelist, edgelist)

    # remove the upi column
    edgelist = edgelist.drop("upi")

    logger.info(
        "Obtained %i components after removing %i edges",
        edgelist.select(pl.col("component")).collect().n_unique(),
        len(edges_to_remove),
    )
    return edgelist, info


def write_recovered_components(
    recovered_components: Dict[str, List[str]], filename: PathType
) -> None:
    """Help to write the recovered component info to a CSV file.

    A helper function that writes to a CSV file the information
    of the recovered components (component of origin of multiplets
    detection algorithm) present in `recovered_components`. The input
    must be a dictionary of the form {old id: [new ids]}.
    :param recovered_components: dictionary of the form {old id: [new ids]}
    :param filename: the path to the output file
    :returns: None
    :rtype: None
    """
    logger.debug("Saving recovered components to %s", filename)
    with open(filename, "w") as fhandler:
        fhandler.write("cell_id,recovered_from\n")
        for (
            old_id,
            recovered,
        ) in recovered_components.items():
            for new_id in sorted(set(recovered)):
                fhandler.write(f"{new_id},{old_id}\n")
    logger.debug("Recovered components saved")
