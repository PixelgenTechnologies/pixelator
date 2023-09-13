"""
This module contains functions for the colocalization analysis in pixelator

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
from functools import partial
from typing import Optional, get_args

import pandas as pd

from pixelator.analysis.colocalization.estimate import (
    estimate_observation_statistics,
    permutation_analysis_results,
)
from pixelator.analysis.colocalization.permute import permutations
from pixelator.analysis.colocalization.prepare import (
    filter_by_region_counts,
    filter_by_unique_values,
    prepare_from_graph,
)
from pixelator.analysis.colocalization.statistics import (
    Jaccard,
    Pearson,
    apply_multiple_stats,
)
from pixelator.analysis.colocalization.types import (
    MarkerColocalizationResults,
    TransformationTypes,
)
from pixelator.graph.utils import Graph
from pixelator.statistics import (
    clr_transformation,
    correct_pvalues,
    log1p_transformation,
    rel_normalization,
)

logger = logging.getLogger(__name__)


def colocalization_from_component_edgelist(
    edgelist: pd.DataFrame,
    component_id: str,
    transformation: TransformationTypes = "raw",
    neighbourhood_size: int = 0,
    n_permutations: int = 50,
    use_full_bipartite: bool = True,
    min_region_count: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get the colocalization scores for the component represented by the given
    `edgelist`.

    :param edgelist: edgelist to compute colocalization scores for
    :param component_id: name of the component
    :param transformation: transformation method to use, defaults to "raw"
    :param neighbourhood_size: size of the neighbourhood to consider, defaults to 0
    :param n_permutations: number of permutations used to calculate the
                           p-values and z-scores, defaults to 50
    :param use_full_bipartite: use the full bipartiate graph, if false use A-node
                               projection, defaults to True
    :param min_region_count: minimum number of counts in region to consider, defaults
                             to 5
    :param random_seed: Set the random seed for the permutation tests, defaults to None
    :return: a dataframe with computed colocalization scores
    """

    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=True,
        simplify=False,
        use_full_bipartite=use_full_bipartite,
    )
    return colocalization_from_component_graph(
        graph=graph,
        component_id=component_id,
        transformation=transformation,
        neighbourhood_size=neighbourhood_size,
        n_permutations=n_permutations,
        min_region_count=min_region_count,
        random_seed=random_seed,
    )


def _transform_data(
    data: MarkerColocalizationResults, transform: TransformationTypes
) -> MarkerColocalizationResults:
    if transform == "raw":
        return data
    if transform == "clr":
        # TODO Check that we want to do row-wise clr here.
        return clr_transformation(data, axis=0)
    if transform == "log1p":
        return log1p_transformation(data)
    if transform == "relative":
        return rel_normalization(data, axis=0)
    raise ValueError(
        f"`transform`must be one of: {'/'.join(get_args(TransformationTypes))}"
    )


def colocalization_from_component_graph(
    graph: Graph,
    component_id: str,
    transformation: TransformationTypes = "raw",
    neighbourhood_size: int = 1,
    n_permutations: int = 50,
    min_region_count: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the colocalization scores for this component graph

    :param graph: graph to compute scores for
    :param component_id: name of the component
    :param transformation: transformation method to use, defaults to "raw"
    :param neighbourhood_size: size of the neighbourhood to consider, defaults to 0
    :param n_permutations: number of permutations used to calculate the
                           p-values and z-scores, defaults to 50
    :param min_region_count: minimum number of counts in region to consider, defaults
                             to 5
    :param random_seed: Set the random seed for the permutation tests, defaults to None
    :return: a dataframe containing colocalization scores for this component
    """
    logger.debug("Computing colocalization for component: %s", component_id)
    logger.debug("Prepare the graph data for computing colocalization")
    marker_counts_by_region = prepare_from_graph(graph, n_neighbours=neighbourhood_size)

    marker_counts_by_region = filter_by_region_counts(
        marker_counts_by_region, min_region_counts=min_region_count
    )
    marker_counts_by_region = filter_by_unique_values(
        marker_counts_by_region, at_least_n_unique=2
    )

    nrow, ncols = marker_counts_by_region.shape
    if nrow == 0:
        logger.warning(
            "Component: %s had no valid regions with the filters you set", component_id
        )
        return pd.DataFrame()
    if ncols == 0:
        logger.warning(
            "Component: %s has no valid markers with the filters you set", component_id
        )
        return pd.DataFrame()

    transformed_marker_counts_by_region = _transform_data(
        marker_counts_by_region, transformation
    )

    functions_of_interest = (Pearson, Jaccard)
    logger.debug("Computing: %s", [f.name for f in functions_of_interest])
    observations = apply_multiple_stats(
        df=transformed_marker_counts_by_region, funcs=functions_of_interest
    )

    logger.debug("Running permutation analysis")
    permutation_results = permutation_analysis_results(
        marker_counts_by_region,
        funcs=functions_of_interest,
        permuter=permutations,
        transformer=partial(_transform_data, transform=transformation),
        n=n_permutations,
        random_seed=random_seed,
    )

    logger.debug("Estimating observation statistics")
    results = estimate_observation_statistics(
        observations=observations,
        permutation_results=permutation_results,
        funcs=functions_of_interest,
    )
    results["component"] = component_id
    logger.debug("Finished computing colocalization for component: %s", component_id)

    return results


def colocalization_scores(
    edgelist: pd.DataFrame,
    use_full_bipartite: bool = True,
    transformation: TransformationTypes = "raw",
    neighbourhood_size: int = 1,
    n_permutations: int = 50,
    min_region_count: int = 5,
) -> pd.DataFrame:
    """
    Computes colocalization scores (unique antibody pairs) for each component
    in the `edgelist` given as input. Only the unique combination of antibodies
    are included and the component id is present in the dataframe which has the
    following columns:
        - marker_1
        - marker_2
        - pearson
        - pearson_z
        - pearson_p_value
        - pearson_p_value_adjusted
        - jaccard
        - jaccard_z
        - jaccard_p_value
        - jaccard_p_value_adjusted
        - component
    The function iterates all the components to compute the scores for each
    pair of antibodies. The scores are computed using the Jaccard index on the
    binarize antibody counts (colocalization) or the Pearson Correlation
    Coefficient on the counts (coabundance). The scores should be high when
    two antibodies are located in the same area of the graph (taking into
    consideration size differences) and their expression is similar (Pearson).
    It also does permutation testing of each colocalizaiton measure to compute
    emprical p-values, corrected p-values, and z-score for each measure.
    :param edgelist: an edge list dataframe with a membership column
    :param use_full_bipartite: use the bipartite graph instead of the projection (UPIA)
    :param transformation: Select a transformation method to use for the colocalization
    :param neighbourhood_size: Set the size of the neighbourhood to
                               consider when computing the colocalization
    :param n_permutations: Select number of permutations used to
                           calculate empirical p-values of the
                           colocalization values
    :param min_region_count: The minimum size of the region (e.g. number
                             of counts in the neighbourhood) required
                             for it to be considered
    :returns: a pd.DataFrame of scores
    :raises: AssertionError when the input is not valid
    """
    if "component" not in edgelist.columns:
        raise AssertionError("edge list is missing the membership column")

    logger.debug(
        "Computing colocalization scores for edge list with %i elements",
        edgelist.shape[0],
    )

    def data():
        grouped = edgelist.groupby("component")
        unique_components = len(edgelist["component"].unique())
        for idx, (component_id, component_df) in enumerate(grouped):
            logger.debug(
                "Computing colocalization of %i/%i",
                idx + 1,
                unique_components,
            )
            # build the graph from the component
            graph = Graph.from_edgelist(
                edgelist=component_df,
                add_marker_counts=True,
                simplify=False,
                use_full_bipartite=use_full_bipartite,
            )

            yield colocalization_from_component_graph(
                graph=graph,
                component_id=component_id,
                transformation=transformation,
                neighbourhood_size=neighbourhood_size,
                n_permutations=n_permutations,
                min_region_count=min_region_count,
            )

    # create dataframe with all the scores
    scores = pd.concat(data(), axis=0)
    p_value_columns = filter(lambda x: "_p" in x, scores.columns)
    for p_value_col in p_value_columns:
        scores.insert(
            scores.columns.get_loc(p_value_col) + 1,
            f"{p_value_col}_adjusted",
            correct_pvalues(scores[p_value_col].to_numpy()),
        )

    logger.debug("Colocalization scores for dataset computed")
    return scores
