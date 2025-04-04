"""Functions for the colocalization analysis in pixelator.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
from functools import partial
from typing import Optional, get_args

import pandas as pd
from statsmodels.stats.multitest import multipletests

from pixelator.common.statistics import (
    correct_pvalues,
    log1p_transformation,
    rate_diff_transformation,
    wilcoxon_test,
)
from pixelator.mpx.analysis.analysis_engine import PerComponentAnalysis
from pixelator.mpx.analysis.colocalization.estimate import (
    estimate_observation_statistics,
    permutation_analysis_results,
)
from pixelator.mpx.analysis.colocalization.prepare import (
    filter_by_region_counts,
    filter_by_unique_values,
    prepare_from_graph,
)
from pixelator.mpx.analysis.colocalization.statistics import (
    Jaccard,
    Pearson,
    apply_multiple_stats,
)
from pixelator.mpx.analysis.colocalization.types import (
    MarkerColocalizationResults,
    TransformationTypes,
)
from pixelator.mpx.analysis.permute import permutations
from pixelator.mpx.graph.utils import Graph
from pixelator.mpx.pixeldataset import PixelDataset

logger = logging.getLogger(__name__)


def colocalization_from_component_edgelist(
    edgelist: pd.DataFrame,
    component_id: str,
    transformation: TransformationTypes = "raw",
    neighbourhood_size: int = 0,
    n_permutations: int = 50,
    use_full_bipartite: bool = True,
    min_region_count: int = 5,
    min_marker_count: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Get the colocalization scores for the component in the given `edgelist`.

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
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             colocalization
    :param random_seed: Set the random seed for the permutation tests, defaults to None
    :return: a dataframe with computed colocalization scores
    :rtype: pd.DataFrame
    """
    graph = Graph.from_edgelist(
        edgelist=edgelist,
        add_marker_counts=True,
        # If we do A-node projection, we will simplify anyway.
        # This just removes the warning.
        simplify=False if use_full_bipartite else True,
        use_full_bipartite=use_full_bipartite,
    )
    return colocalization_from_component_graph(
        graph=graph,
        component_id=component_id,
        transformation=transformation,
        neighbourhood_size=neighbourhood_size,
        n_permutations=n_permutations,
        min_region_count=min_region_count,
        min_marker_count=min_marker_count,
        random_seed=random_seed,
    )


def _transform_data(
    data: MarkerColocalizationResults, transform: TransformationTypes
) -> MarkerColocalizationResults:
    if transform == "raw":
        return data
    if transform == "log1p":
        return log1p_transformation(data)
    if transform == "rate-diff":
        return rate_diff_transformation(data)
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
    min_marker_count: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the colocalization scores for this component graph.

    :param graph: graph to compute scores for
    :param component_id: name of the component
    :param transformation: transformation method to use, defaults to "raw"
    :param neighbourhood_size: size of the neighbourhood to consider, defaults to 0
    :param n_permutations: number of permutations used to calculate the
                           p-values and z-scores, defaults to 50
    :param min_region_count: minimum number of counts in region to consider, defaults
                             to 5
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             colocalization
    :param random_seed: Set the random seed for the permutation tests, defaults to None
    :return: a dataframe containing colocalization scores for this component
    :rtype: pd.DataFrame
    """
    logger.debug("Computing colocalization for component: %s", component_id)
    logger.debug("Prepare the graph data for computing colocalization")

    raw_marker_counts = graph.node_marker_counts
    # Record markers to keep
    # Remove markers with zero variance and markers below minimum marker count
    markers_to_keep = raw_marker_counts.columns[
        (raw_marker_counts != 0).any(axis=0)
        & (raw_marker_counts.nunique() > 1)
        & (raw_marker_counts.sum() >= min_marker_count)
    ]

    marker_counts_by_region = prepare_from_graph(graph, n_neighbours=neighbourhood_size)
    marker_counts_by_region = marker_counts_by_region[markers_to_keep]
    marker_counts_by_region = filter_by_region_counts(
        marker_counts_by_region, min_region_counts=min_region_count
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
    min_marker_count: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute colocalization scores for antibody pairs.

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
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             colocalization
    :param random_seed: Set a random seed for the permutation function
    :returns: a pd.DataFrame of scores
    :rtype: pd.DataFrame
    :raises AssertionError: when the input is not valid
    :raises ValueError: when no components were found to be valid for
                        computing colocalization.

    """
    if "component" not in edgelist.columns:
        raise AssertionError("edge list is missing the membership column")

    logger.debug(
        "Computing colocalization scores for edge list with %i elements",
        edgelist.shape[0],
    )

    def data():
        grouped = edgelist.groupby("component", observed=True)
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
                # If we do A-node projection, we will simplify anyway.
                # This just removes the warning.
                simplify=False if use_full_bipartite else True,
                use_full_bipartite=use_full_bipartite,
            )
            if len(graph.vs) < 2:
                logger.warning(
                    "Component %s only had a single node. It will be skipped.",
                    component_id,
                )
                continue

            yield colocalization_from_component_graph(
                graph=graph,
                component_id=component_id,
                transformation=transformation,
                neighbourhood_size=neighbourhood_size,
                n_permutations=n_permutations,
                min_region_count=min_region_count,
                min_marker_count=min_marker_count,
                random_seed=random_seed,
            )

    # create dataframe with all the scores
    try:
        scores = pd.concat(data(), axis=0)
    except ValueError as error:
        logger.error(
            "No data was found to compute colocalization, probably "
            "because all components only had a single node."
        )
        raise error

    p_value_columns = filter(lambda x: "_p" in x, scores.columns)
    for p_value_col in p_value_columns:
        scores.insert(
            scores.columns.get_loc(p_value_col) + 1,
            f"{p_value_col}_adjusted",
            correct_pvalues(scores[p_value_col].to_numpy()),
        )

    logger.debug("Colocalization scores for dataset computed")
    return scores


def get_differential_colocalization(
    colocalization_data_frame: pd.DataFrame,
    reference: str,
    targets: str | list[str] | None = None,
    contrast_column: str = "sample",
    value_column: str = "pearson_z",
) -> pd.DataFrame:
    """Calculate the differential colocalization.

    :param colocalization_data_frame: The colocalization data frame.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param value_column: What colocalization metric to use. Defaults to "pearson_z".

    :return: The differential colocalization.
    :rtype: pd.DataFrame
    """
    if targets is None:
        targets = colocalization_data_frame[contrast_column].unique()
        targets = list(set(targets) - {reference})
    elif isinstance(targets, str):
        targets = [targets]

    if len(targets) > 5:
        logger.warning(
            "There are more than 5 targets in the dataset. This may take a while."
        )
    same_marker_mask = (
        colocalization_data_frame["marker_1"] == colocalization_data_frame["marker_2"]
    )
    data_frame = colocalization_data_frame.loc[~same_marker_mask, :]
    merged_differential_colocalization = pd.DataFrame()
    for target in targets:
        differential_colocalization = (
            data_frame.groupby(["marker_1", "marker_2"])
            .apply(
                lambda marker_data: wilcoxon_test(
                    marker_data,
                    reference=reference,
                    target=target,
                    contrast_column=contrast_column,
                    value_column=value_column,
                )
            )
            .reset_index()
        )

        # If a marker appears only in one of the datasets,
        # it's differential value will be NAN
        nan_values = differential_colocalization[
            differential_colocalization["median_difference"].isna()
        ].index
        differential_colocalization.drop(
            nan_values,
            axis="index",
            inplace=True,
        )

        _, pvals_corrected, *_ = multipletests(
            differential_colocalization["p_value"], method="bonferroni"
        )
        differential_colocalization["p_adj"] = pvals_corrected
        differential_colocalization["target"] = target
        merged_differential_colocalization = pd.concat(
            (merged_differential_colocalization, differential_colocalization), axis=0
        )

    return merged_differential_colocalization


class ColocalizationAnalysis(PerComponentAnalysis):
    """Run colocalization analysis on each component."""

    ANALYSIS_NAME = "colocalization"

    def __init__(
        self,
        transformation_type: TransformationTypes,
        neighbourhood_size: int,
        n_permutations: int,
        min_region_count: int,
        min_marker_count: int,
    ):
        """Initialize the ColocalizationAnalysis.

        :param transformation_type: transformation method to use
        :param neighbourhood_size: size of the neighbourhood to consider
        :param n_permutations: Select number of permutations used to
                               calculate empirical z-scores and p-values of the
                               colocalization values
        :param min_region_count: The minimum size of the region (e.g. number
                             of counts in the neighbourhood) required
                             for it to be considered for colocalization analysis
        :param min_marker_count: the minimum number of counts of a marker to calculate
                             colocalization
        """
        self.transformation_type = transformation_type
        self.neighbourhood_size = neighbourhood_size
        self.n_permutations = n_permutations
        self.min_region_count = min_region_count
        self.min_marker_count = min_marker_count

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        """Run colocalization analysis on the component."""
        logger.debug("Running colocalization analysis on component %s", component_id)
        return colocalization_from_component_graph(
            graph=component,
            component_id=component_id,
            transformation=self.transformation_type,
            neighbourhood_size=self.neighbourhood_size,
            n_permutations=self.n_permutations,
            min_region_count=self.min_region_count,
            min_marker_count=self.min_marker_count,
        )

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post process the colocalization data.

        This will adjust the p-values using the Benjamini-Hochberg method.
        """
        logger.debug("Post processing colocalization analysis data")
        if data.empty:
            return data
        p_value_columns = filter(lambda x: "_p" in x, data.columns)
        for p_value_col in p_value_columns:
            data.insert(
                data.columns.get_loc(p_value_col) + 1,
                f"{p_value_col}_adjusted",
                correct_pvalues(data[p_value_col].to_numpy()),
            )
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        """Add the colocalization data to the PixelDataset."""
        logger.debug("Adding colocalization analysis data to PixelDataset")
        pxl_dataset.colocalization = data
        return pxl_dataset
