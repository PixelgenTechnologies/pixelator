"""Copyright © 2023 Pixelgen Technologies AB."""

import logging
import time
from functools import partial
from typing import Iterable, get_args

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from pixelator.common.statistics import correct_pvalues, wilcoxon_test
from pixelator.common.utils import get_pool_executor
from pixelator.mpx.analysis.analysis_engine import PerComponentAnalysis
from pixelator.mpx.analysis.permute import permutations
from pixelator.mpx.analysis.polarization.types import PolarizationTransformationTypes
from pixelator.mpx.graph.utils import Graph
from pixelator.mpx.pixeldataset import MIN_VERTICES_REQUIRED, PixelDataset

logger = logging.getLogger(__name__)


def _compute_for_marker(
    marker, marker_count_matrix, weight_matrix, normalization_factor
):
    try:
        x_centered = np.array(
            marker_count_matrix[marker] - marker_count_matrix[marker].mean()
        )
        x_centered_squared_sum = (x_centered * x_centered).sum()
        element_wise_weighted_sum = (x_centered @ weight_matrix @ x_centered).sum()
        r = normalization_factor * (element_wise_weighted_sum / x_centered_squared_sum)
        return r
    except ValueError as e:
        logger.warning("Error computing Moran's I for marker %s: %s", marker, e)
        return np.nan


def _compute_morans_i(
    marker: str,
    marker_count_matrix: pd.DataFrame,
    marker_count_matrix_permuted: Iterable[pd.DataFrame],
    weight_matrix: csr_matrix,
    normalization_factor: float,
):
    r = _compute_for_marker(
        marker, marker_count_matrix, weight_matrix, normalization_factor
    )
    perm_rs = np.fromiter(
        (
            _compute_for_marker(marker, perm, weight_matrix, normalization_factor)
            for perm in marker_count_matrix_permuted
        ),
        dtype=np.float64,
    )
    perm_rs_mean = np.nanmean(perm_rs)
    perm_rs_std_dev = np.nanstd(perm_rs)
    perm_rs_z_score = (r - perm_rs_mean) / perm_rs_std_dev
    p_value = norm.sf(np.abs(perm_rs_z_score))
    return marker, r, perm_rs_z_score, perm_rs_mean, perm_rs_std_dev, p_value


def polarization_scores_component_graph(
    graph: Graph,
    component_id: str,
    transformation: PolarizationTransformationTypes = "log1p",
    n_permutations: int = 50,
    min_marker_count: int = 2,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Calculate Moran's I statistics for a component graph.

    Computes polarization statistics for all antibodies in the `graph` given
    as input (a single connected component). The statistics are computed using
    Moran's I autocorrelation to measure how clustered/localized the spatial
    patterns of the antibody is in the graph.

    The statistics contain the I value, as well as a p-value, the adjusted p-value
    and the z-score.The later are based on permuting the input graph to estimate a
    null-hypothesis for the Moran's I statistic.

    The function returns a pd.DataFrame with the following columns:
      morans_i, morans_p_value, morans_z, marker, component

    :param graph: a graph (it must be a single connected component)
    :param component_id: the id of the component
    :param transformation: the count transformation method to use (raw, log1p)
    :param n_permutations: the number of permutations to use to estimate the
                           null-hypothesis for the Moran's I statistic
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             the Moran's I statistic
    :param random_seed: the random seed to use to ensure that the permutations
                        are reproducible across runs
    :returns: a pd.DataFrame with the polarization statistics for each antibody
    :rtype: pd.DataFrame
    :raises: AssertionError when the input is not valid
    """
    if graph.vcount() < MIN_VERTICES_REQUIRED:
        logger.debug(
            (
                "Trying to compute polarization scores in component %s that has less"
                " than %i vertices which is the minimum required"
            ),
            component_id,
            MIN_VERTICES_REQUIRED,
        )
        return pd.DataFrame()

    def _compute_morans_on_current_graph():
        N = float(graph.vcount())

        W = graph.get_adjacency_sparse()
        # Normalize the weights by the degree of the node
        W = W / W.sum(axis=0)
        X = graph.node_marker_counts

        # Calculate normalization factor
        C = N / W.sum()

        # Record markers to keep
        # Remove markers with zero variance and markers below minimum marker count
        markers_to_keep = X.columns[
            (X != 0).any(axis=0) & (X.nunique() > 1) & (X.sum() >= min_marker_count)
        ]

        transform_func = np.log1p if transformation == "log1p" else lambda x: x
        X_perm = [
            transform_func(x)
            for x in permutations(X, n=n_permutations, random_seed=random_seed)
        ]
        X = transform_func(X)

        # Apply marker filter after transformation
        X = X.loc[:, markers_to_keep]
        X_perm = [x.loc[:, markers_to_keep] for x in X_perm]

        _compute_morans_i_per_marker = partial(
            _compute_morans_i,
            marker_count_matrix=X,
            marker_count_matrix_permuted=X_perm,
            weight_matrix=W,
            normalization_factor=C,
        )
        markers = X.columns

        results = pd.DataFrame(
            map(_compute_morans_i_per_marker, markers),
            columns=[
                "marker",
                "morans_i",
                "morans_z",
                "perm_mean",
                "perm_std",
                "morans_p_value",
            ],
        )
        return results

    start_time = time.perf_counter()
    logger.info("Computing Moran's I for component: %s", component_id)

    results = _compute_morans_on_current_graph()
    results["component"] = component_id

    run_time = time.perf_counter() - start_time
    logger.info(
        "Finished computing Moran's I for component: %s in %.2fs",
        component_id,
        run_time,
    )
    results = results.drop(columns=["perm_mean", "perm_std"])
    return results


def polarization_scores_component_df(
    component_id: str,
    component_df: pd.DataFrame,
    use_full_bipartite: bool,
    transformation: PolarizationTransformationTypes = "log1p",
    n_permutations: int = 50,
    min_marker_count: int = 2,
    random_seed: int | None = None,
):
    """Calculate Moran's I statistics for a component.

    See `polarization_scores_component_graph` for details.

    :param component_id: the id of the component
    :param component_df: A data frame with an edgelist for a single connected component
    :param use_full_bipartite: use the bipartite graph instead of the projection (UPIA)
    :param transformation: the count transformation method to use (raw, log1p)
    :param n_permutations: the number of permutations to use to estimate the
                           null-hypothesis for the Moran's I statistic
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             the Moran's I statistic
    :param random_seed: the random seed to use to ensure that the permutations
                        are reproducible across runs
    :returns: a pd.DataFrame with the polarization statistics for each antibody
    :rtype: pd.DataFrame
    :raises: AssertionError when the input is not valid
    """
    graph = Graph.from_edgelist(
        edgelist=component_df,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=use_full_bipartite,
    )

    component_result = polarization_scores_component_graph(
        graph=graph,
        component_id=component_id,
        transformation=transformation,
        n_permutations=n_permutations,
        min_marker_count=min_marker_count,
        random_seed=random_seed,
    )
    return component_result


def polarization_scores(
    edgelist: pd.DataFrame,
    use_full_bipartite: bool = False,
    transformation: PolarizationTransformationTypes = "log1p",
    n_permutations: int = 0,
    min_marker_count: int = 2,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Calculate Moran's I statistics for an edgelist.

    Compute polarization scores from an `edgelist`.
    The function iterates all the components to compute polarization scores
    for each antibody in the component. The scores are computed using Moran's
    I autocorrelation to measure how clustered/localised the spatial patterns of
    the antibody is in the component's graph. Spatial weights (`w`) are derived
    directly from the graph. The function returns a pd.DataFrame with the following
    columns:
      morans_i, morans_p_value, morans_z, morans_p_adjusted, marker, component
      (morans_p_value_sim and morans_z_sim if `permutations` > 0)
    :param edgelist: an edge list (pd.DataFrame) with a component column
    :param use_full_bipartite: use the bipartite graph instead of the projection (UPIA)
    :param transformation: the count transformation method to use (raw, log1p)
    :param n_permutations: the number of permutations for simulated Z-score (z_sim)
                           estimation (if n_permutations>0)
    :param min_marker_count: the minimum number of counts of a marker to calculate
                             the Moran's I statistic
    :param random_seed: the random seed to use for reproducibility
    :returns: a pd.DataFrames with all the polarization scores
    :rtype: pd.DataFrame
    :raises: AssertionError when the input is not valid
    """
    if transformation not in get_args(PolarizationTransformationTypes):
        raise AssertionError(
            f"incorrect value for count transformation {transformation}"
        )

    if "component" not in edgelist.columns:
        raise AssertionError("Edge list is missing the component column")

    if n_permutations < 0:
        logger.warning(
            "Setting `n_permutations < 0` will mean no z-scores and p-values will be calculated."
        )

    logger.debug(
        "Computing polarization for edge list with %i elements",
        edgelist.shape[0],
    )

    def data():
        with get_pool_executor() as pool:
            polarization_function = partial(
                polarization_scores_component_df,
                use_full_bipartite=use_full_bipartite,
                transformation=transformation,
                n_permutations=n_permutations,
                min_marker_count=min_marker_count,
                random_seed=random_seed,
            )
            yield from pool.starmap(
                polarization_function,
                (
                    (component_id, component_df)
                    for component_id, component_df in edgelist.groupby(
                        "component", observed=True
                    )
                ),
                chunksize=10,
            )

    # create dataframe with all the scores
    logger.debug("Concatenating the polarization dataframes")
    scores = pd.concat(data(), axis=0)
    if scores.empty:
        logger.warning("Polarization results were empty")
        return scores

    scores.insert(
        scores.columns.get_loc("morans_p_value") + 1,
        "morans_p_adjusted",
        correct_pvalues(scores["morans_p_value"].to_numpy()),
    )

    logger.debug("Polarization scores for edge list computed")
    return scores


class PolarizationAnalysis(PerComponentAnalysis):
    """Run polarization analysis on each component."""

    ANALYSIS_NAME = "polarization"

    def __init__(
        self,
        transformation_type: PolarizationTransformationTypes,
        n_permutations: int,
        min_marker_count: int,
        random_seed: int | None = None,
    ):
        """Initialize polarization analysis.

        :param transformation: the count transformation method to use (raw, log1p)
        :param n_permutations: the number of permutations to use to estimate the
                               null-hypothesis for the Moran's I statistic
        :param min_marker_count: the minimum number of counts of a marker to calculate
                                 the Moran's I statistic
        :param random_seed: set a random seed to ensure reproducibility when calculating z-scores
                            and p-values.
        """
        if transformation_type not in get_args(PolarizationTransformationTypes):
            raise AssertionError(
                f"incorrect value for count transformation {transformation_type}"
            )
        self.transformation = transformation_type
        self.permutations = n_permutations
        self.min_marker_count = min_marker_count
        self.random_seed = random_seed

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        """Run polarization analysis on component."""
        logger.debug("Running polarization analysis on component %s", component_id)
        return polarization_scores_component_graph(
            graph=component,
            component_id=component_id,
            transformation=self.transformation,
            n_permutations=self.permutations,
            min_marker_count=self.min_marker_count,
            random_seed=self.random_seed,
        )

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post process polarization analysis data.

        This will adjust the calculated p-values for the calculate Moran's I statistics.
        """
        logger.debug("Post processing polarization analysis data")
        if data.empty:
            return data
        data.insert(
            data.columns.get_loc("morans_p_value") + 1,
            "morans_p_adjusted",
            correct_pvalues(data["morans_p_value"].to_numpy()),
        )
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        """Add data to the polarization field of the PixelDataset."""
        logger.debug("Adding polarization analysis data to PixelDataset")
        pxl_dataset.polarization = data
        return pxl_dataset


def get_differential_polarity(
    polarity_data: pd.DataFrame,
    reference: str,
    targets: str | list[str] | None = None,
    contrast_column: str = "sample",
    value_column: str = "morans_z",
) -> pd.DataFrame:
    """Calculate the differential polarity.

    :param polarity_data: The polarity data frame.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param value_column: What polarity metric to use. Defaults to "morans_z".

    :return: The differential polarity.
    :rtype: pd.DataFrame
    """
    if targets is None:
        targets = polarity_data[contrast_column].unique()
        targets = list(set(targets) - {reference})
    elif isinstance(targets, str):
        targets = [targets]

    if len(targets) > 5:
        logger.warning(
            "There are more than 5 targets in the dataset. This may take a while."
        )
    merged_differential_polarity = pd.DataFrame()
    for target in targets:
        differential_polarity = (
            polarity_data.groupby("marker")
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
        nan_values = differential_polarity[
            differential_polarity["median_difference"].isna()
        ].index
        differential_polarity.drop(
            nan_values,
            axis="index",
            inplace=True,
        )

        _, pvals_corrected, *_ = multipletests(
            differential_polarity["p_value"], method="bonferroni"
        )
        differential_polarity["p_adj"] = pvals_corrected
        differential_polarity["target"] = target
        merged_differential_polarity = pd.concat(
            (merged_differential_polarity, differential_polarity), axis=0
        )

    return merged_differential_polarity
