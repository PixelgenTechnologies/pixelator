"""Functions related to perform community detection on the pixelator graph step.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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

    This function takes as input an edge list pd.Dataframe (csv) that has
    been generated with `pixelator collapse`. The function filters the
    edge list by count (`min_count`) and then adds a column to the edge list
    with the respective connected components ids obtained from the graph. The
    column is named "component". The edge list is then processed to recover
    big components (technical multiplets) into smaller components if only if
    `multiplet_recovery` is True. The recovery is done using community
    detection to detect and remove problematic edges using the Leiden [1]_ community
    detection algorithm. Information about the recovered components is written
    to a CSV file (recovered_components.csv) and the edge list containing only
    the removed edge is written to a CSV file (removed_edges.csv.gz).

    The following files are generated:

    - raw edge list (csv) updated with components membership
    - edge list (csv) after multiplets recovery (if any)
    - metrics (json) with information


    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s4q:598-019-41695-z

    :param input: the path to the edge list dataframe (csv)
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
    edgelist = pd.read_csv(input, dtype_backend="pyarrow")

    # filter data by count
    if min_count > 1:
        logger.debug(
            "Filtering edge list with %i rows using %i as minimum count",
            edgelist.shape[0],
            min_count,
        )
        edgelist = edgelist[edgelist["count"] >= min_count]
        logger.debug("Filtered edge list has %i elements", edgelist.shape[0])

    if edgelist.shape[0] == 0:
        raise RuntimeError(
            f"The edge list has 0 elements after filtering by %{min_count}"
        )

    # check if the are problematic edges (same upib and upia)
    problematic_edges = np.intersect1d(edgelist["upib"], edgelist["upia"])
    if len(problematic_edges) > 0:
        logger.warning(
            "The edge list has %i intersecting UPIA and UPIB, these will be removed",
            len(problematic_edges),
        )
        edgelist = edgelist[
            (~edgelist["upib"].isin(problematic_edges))
            & (~edgelist["upia"].isin(problematic_edges))
        ]

    if edgelist.shape[0] == 0:
        raise RuntimeError(
            "The edge list has 0 elements after removing problematic edges"
        )

    # assign component column to edge list
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        prefix=DEFAULT_COMPONENT_PREFIX,
    )

    # save the edge list (raw)
    edgelist.to_csv(
        Path(output) / f"{output_prefix}.raw_edgelist.csv.gz",
        header=True,
        index=False,
        sep=",",
        compression="gzip",
    )

    # get raw metrics before multiplets recovery
    raw_metrics = edgelist_metrics(edgelist)

    if multiplet_recovery:
        edgelist, info = recover_technical_multiplets(
            edgelist=edgelist,
            leiden_iterations=leiden_iterations,
            filename=Path(output) / f"{output_prefix}.discarded_edgelist.csv.gz",
        )

        # save the recovered components info to a file
        write_recovered_components(
            info,
            filename=Path(output) / f"{output_prefix}.components_recovered.csv",
        )

    # save the edge list (recovered)
    edgelist.to_csv(
        Path(output) / f"{output_prefix}.edgelist.csv.gz",
        header=True,
        index=False,
        sep=",",
        compression="gzip",
    )

    # save metrics raw (JSON)
    with open(metrics_file, "w") as outfile:
        # we want the metrics to be computed before recovery
        json.dump(raw_metrics, outfile, default=np_encoder)


def community_detection_crossing_edges(
    graph: Graph,
    leiden_iterations: int = 10,
    beta: float = 0,
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
    :param beta: parameter to control the randomness on cluster selection when Leiden
                 merges clustering (0 - maximize objective function i.e. modularity;
                 inf - uniform distribution to merge with any other cluster)
    :returns: a list of sets with the edges between communities (edges ids)
    :rtype: List[Set[str]]
    :raises AssertionError: if the method is not supported
    """
    logger.debug(
        "Computing community detection using the leiden algorithm in a graph "
        "with %i nodes and %i edges",
        graph.vcount(),
        graph.ecount(),
    )

    if beta < 0:
        raise ValueError(f"Beta parameter cannot be a negative value: {beta}")

    # compute communities
    # NOTE the default number of iterations is 2 but a higher number is needed to
    # mitigate the random variability between runs
    vertex_clustering = graph.community_leiden(
        objective_function="modularity",
        n_iterations=leiden_iterations,
        beta=beta,
    )

    # obtain the list of edges connecting the communities (crossing edges)
    edges = []
    if vertex_clustering is not None:
        # get the crossing edges
        graph.es["is_crossing"] = vertex_clustering.crossing()
        edges = graph.es.select(is_crossing_eq=True)
        # translate the edges to sets of their corresponding vertex names
        edges = [{e.vertex_tuple[0]["name"], e.vertex_tuple[1]["name"]} for e in edges]
        logger.debug(
            "Community detection detected %i crossing edges in %i communities with a "
            "modularity of %f",
            len(edges),
            len(vertex_clustering),
            vertex_clustering.modularity,
        )
    else:
        logger.debug("Community detection returned an empty list")
    return edges


def detect_edges_to_remove(
    edgelist: pd.DataFrame,
    leiden_iterations: int = 10,
) -> List[Set[str]]:
    """Use Leiden algorithm to detect communities from an edgelist.

    This method uses the community detection Leiden algorithm to detect
    communities in the whole graph corresponding to the edge list given as input.
    Edges connecting the communities are computed and returned.
    :param edgelist: The edge list used to create the graph
    :param leiden_iterations: the number of iterations for the leiden algorithm
    :return: A list of edges (sets) that are connecting communities
    :rtype: List[Set[str]]
    """
    logger.debug(
        "Detecting edges to remove using the leiden algorithm"
        " with an edge list with %i rows",
        edgelist.shape[0],
    )

    # build the graph from the edge list
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=False,
        simplify=False,
        use_full_bipartite=True,
    )

    # perform community detection
    edges = community_detection_crossing_edges(
        graph=graph,
        leiden_iterations=leiden_iterations,
    )

    logger.debug("Found %i edges to remove", len(edges))
    return edges


def recovered_component_info(
    old_edgelist: pd.DataFrame, new_edgelist: pd.DataFrame
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
    old_memberships = old_edgelist[["upi", "component"]]
    new_memberships = new_edgelist[["upi", "component"]]
    merged = pd.merge(
        old_memberships,
        new_memberships,
        on="upi",
        suffixes=["_old", "_new"],
        indicator=True,
    )
    return (
        merged[["component_old", "component_new"]]
        .groupby(["component_old"])
        .apply(lambda x: list(x["component_new"].value_counts().index))
        .to_dict()
    )


def recover_technical_multiplets(
    edgelist: pd.DataFrame,
    leiden_iterations: int = 10,
    filename: Optional[PathType] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
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
    :param leiden_iterations: the number of iterations for the leiden algorithm
    :param filename: If not None, the edge list with the discarded edges will
                     be saved to this file
    :return: A tuple with the modified edge list and a dictionary with the mapping of
             old component to new component ids
    :rtype: Tuple[pd.DataFrame, Dict[str, List[str]]]
    """

    def vertex_name_pairs_to_upis(
        edge_tuples: List[Set[str]],
    ) -> List[str]:
        """Translate each pair of vertices into full UPI info.

        Each pair of vertices should be translated into their UPI. Since we can not
        know which order they were in in the original edge list we create both, i.e.
        [(A,B), (C, D)] becomes [AB, BA, CD, DC].
        """
        return [
            "".join(combo)
            for edge in edge_tuples
            for combo in itertools.permutations(edge, 2)
        ]

    logger.debug(
        "Starting multiplets recovery in edge list with %i rows", edgelist.shape[0]
    )

    # perform community detection
    edges_to_remove = detect_edges_to_remove(
        edgelist=edgelist,
        leiden_iterations=leiden_iterations,
    )

    if len(edges_to_remove) == 0:
        logger.info("Obtained 0 edges to remove, no recovery performed")
        return edgelist, {}

    # translate the edges to tuples of their corresponding vertex names
    edges_to_remove = vertex_name_pairs_to_upis(edges_to_remove)  # type: ignore

    # add the combined upi here as a way to make sure we can use it to
    # check edge membership for between the old and the (potential)
    # new components.
    edgelist["upi"] = edgelist["upia"] + edgelist["upib"]
    filename = None
    if filename is not None:
        logger.debug("Saving edge list with discarded edges to %s", filename)
        # save the discarded edges to a file
        masked_df = edgelist[edgelist["upi"].isin(edges_to_remove)]
        masked_df.to_csv(
            filename,
            header=True,
            index=False,
            sep=",",
            compression="gzip",
        )

    old_edgelist = edgelist[["upi", "component"]]
    edgelist = edgelist[~edgelist["upi"].isin(edges_to_remove)]
    edgelist = update_edgelist_membership(
        edgelist=edgelist,
        prefix=DEFAULT_COMPONENT_PREFIX_RECOVERY,
    )

    # get the info of recovered components ids to old ones
    info = recovered_component_info(old_edgelist, edgelist)

    # remove the upi column and reset index
    edgelist = edgelist.drop("upi", axis=1, inplace=False)
    edgelist = edgelist.reset_index(drop=True, inplace=False)

    logger.info(
        "Obtained %i components after removing %i edges",
        edgelist["component"].nunique(),
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
