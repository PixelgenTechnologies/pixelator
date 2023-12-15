"""Functions related to perform community detection on the pixelator graph step.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from pixelator.graph.constants import (
    DEFAULT_COMPONENT_PREFIX,
    DEFAULT_COMPONENT_PREFIX_RECOVERY,
)
from pixelator.graph.graph import Graph
from pixelator.graph.utils import (
    edgelist_metrics,
    update_edgelist_membership,
)
from pixelator.types import PathType
from pixelator.utils import np_encoder

logger = logging.getLogger(__name__)


def connect_components(
    input: str,
    output: str,
    output_prefix: str,
    metrics_file: str,
    multiplet_recovery: bool,
    leiden_iterations: int = 10,
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
    :param output_prefix: the prefix to prepend to the files (sample name)
    :param metrics_file: the path to a JSON file to write metrics
    :param multiplet_recovery: set to true to activate multiplet recovery
    :param leiden_iterations: the number of iterations for the leiden algorithm
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

    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )

    # assign component column to edge list
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        graph=graph,
        prefix=DEFAULT_COMPONENT_PREFIX,
    )

    # get raw metrics before multiplets recovery
    logger.debug("Calculating raw edgelist metrics")
    raw_metrics = edgelist_metrics(edgelist, graph)

    if multiplet_recovery:
        edgelist, info = recover_technical_multiplets(
            edgelist=edgelist,
            graph=graph,
            leiden_iterations=leiden_iterations,
            removed_edges_edgelist_file=Path(output)
            / f"{output_prefix}.discarded_edgelist.parquet",
        )

        # save the recovered components info to a file
        write_recovered_components(
            info,
            filename=Path(output) / f"{output_prefix}.components_recovered.csv",
        )

    del graph

    # save the edge list (recovered)
    logger.debug("Save the edgelist")
    edgelist.collect(streaming=True, no_optimization=True).write_parquet(
        Path(output) / f"{output_prefix}.edgelist.parquet",
        compression="zstd",
    )

    # save metrics raw (JSON)
    with open(metrics_file, "w") as outfile:
        # we want the metrics to be computed before recovery
        json.dump(raw_metrics, outfile, default=np_encoder)


def community_detection_crossing_edges(
    graph: Graph,
    leiden_iterations: int = 10,
    beta: float = 0.01,
) -> List[Set[str]]:
    """Detect spurious edges connecting components by community detection.

    Use the Leiden [1]_ community detection algorithm to detect communities in a graph.
    The function will then collect and returns the list of edges connecting the
    communities in order to obtain the optimal number of communities.

    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s4q:598-019-41695-z

    :param graph: a graph object
    :param leiden_iterations: the number of iterations for the leiden algorithm
    :param beta: parameter to control the randomness of the cluster refinement in
                 the Leiden algorithm. Must be a positive, non-zero float.
    :returns: a list of sets with the edges between communities (edges ids)
    :rtype: List[Set[str]]
    :raises AssertionError: if unsupported community detection options are found.
    """
    logger.debug(
        "Computing community detection using the leiden algorithm in a graph "
        "with %i nodes and %i edges",
        graph.vcount(),
        graph.ecount(),
    )

    # compute communities
    vertex_clustering = graph.community_leiden(
        n_iterations=leiden_iterations,
        beta=beta,
    )

    # obtain the list of edges connecting the communities (crossing edges)
    # get the crossing edges
    crossing_edges = vertex_clustering.crossing()
    # translate the edges to sets of their corresponding vertex names
    logger.debug("Iterating over crossing edges")
    edges = [
        {e.vertex_tuple[0]["name"], e.vertex_tuple[1]["name"]} for e in crossing_edges
    ]
    logger.debug("Finished iterating over crossing edges")
    logger.debug(
        "Community detection detected %i crossing edges in %i communities with a "
        "modularity of %f",
        len(edges),
        len(vertex_clustering),
        vertex_clustering.modularity,
    )
    return edges


# TODO Perhaps we can drop this method, since it doesn't
# add very much.
def detect_edges_to_remove(
    graph: Graph,
    leiden_iterations: int = 10,
) -> List[Set[str]]:
    """Use Leiden algorithm to detect communities from an edgelist.

    This method uses the community detection Leiden algorithm to detect
    communities in the whole graph corresponding to the edge list given as input.
    Edges connecting the communities are computed and returned.
    :param graph: The graph to detect edges to remove from.
    :param leiden_iterations: the number of iterations for the leiden algorithm
    :return: A list of edges (sets) that are connecting communities
    :rtype: List[Set[str]]
    """
    # perform community detection
    edges = community_detection_crossing_edges(
        graph=graph,
        leiden_iterations=leiden_iterations,
    )

    logger.debug("Found %i edges to remove", len(edges))
    return edges


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
        .collect(streaming=True, projection_pushdown=False)
        .iter_rows()
    )


def recover_technical_multiplets(
    edgelist: pl.LazyFrame,
    graph: Graph,
    leiden_iterations: int = 10,
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
    :param leiden_iterations: the number of iterations for the leiden algorithm
    :param removed_edges_edgelist_file: If not None, the edge list with the discarded
                                        edges will be saved to this file
    :return: A tuple with the modified edge list and a dictionary with the mapping of
             old component to new component ids
    :rtype: Tuple[pl.LazyFrame, Dict[str, List[str]]]
    """

    def vertex_name_pairs_to_upis(
        edge_tuples: List[Set[str]],
    ) -> Set[str]:
        """Translate each pair of vertices into full UPI info.

        Each pair of vertices should be translated into their UPI. Since we can not
        know which order they were in in the original edge list we create both, i.e.
        [(A,B), (C, D)] becomes [AB, BA, CD, DC].
        """
        return {
            "".join(combo)
            for edge in edge_tuples
            for combo in itertools.permutations(edge, 2)
        }

    logger.debug(
        "Starting multiplets recovery in edge list with %i rows",
        edgelist.select(pl.count()).collect().item(0, 0),
    )

    # perform community detection
    edges_to_remove = detect_edges_to_remove(
        graph=graph,
        leiden_iterations=leiden_iterations,
    )

    del graph

    if len(edges_to_remove) == 0:
        logger.info("Obtained 0 edges to remove, no recovery performed")
        return edgelist, {}

    # translate the edges to tuples of their corresponding vertex names
    edges_to_remove = vertex_name_pairs_to_upis(edges_to_remove)  # type: ignore

    # add the combined upi here as a way to make sure we can use it to
    # check edge membership for between the old and the (potential)
    # new components.
    edgelist = edgelist.with_columns(
        pl.concat_str(pl.col("upia"), pl.col("upib")).alias("upi")
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
    edgelist = edgelist.filter(~pl.col("upi").is_in(edges_to_remove))

    logger.debug("Creating updated edgelist")
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        graph=None,  # We need to make sure that a new graph is built
        # from the edgelist here,
        # otherwise the edge indexes will not match.
        prefix=DEFAULT_COMPONENT_PREFIX_RECOVERY,
    )

    # get the info of recovered components ids to old ones
    info = recovered_component_info(old_edgelist, edgelist)

    # remove the upi column and reset index
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
