"""Copyright (c) 2023 Pixelgen Technologies AB."""


import logging
import warnings
from concurrent import futures
from typing import get_args

import esda.moran
import numpy as np
import pandas as pd

from pixelator.analysis.polarization.types import PolarizationNormalizationTypes
from pixelator.graph.utils import Graph, create_node_markers_counts
from pixelator.pixeldataset import (
    MIN_VERTICES_REQUIRED,
)
from pixelator.statistics import (
    clr_transformation,
    correct_pvalues,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module="libpysal")
    from libpysal.weights import WSP


logger = logging.getLogger(__name__)


def morans_autocorr(
    w: np.ndarray, y: np.ndarray, permutations: int
) -> esda.moran.Moran:
    """Calculate Moran's I statistics.

    Computes the Moran's I autocorrelation statistics for the given spatial
    weights (`w`) and target variable (`y`). The target variable can be the counts
    of a specific antibody so the Moran's I value could indicate if the
    antibody has a localized spatial pattern. The function returns an object
    with the Moran's statistics: I, p_rand and z_rand (as well as p_sim and z_sim
    if `permutations` > 0).
    :param w: the weights matrix (connectivity matrix)
    :param y: the counts vector
    :param permutations: the number of permutations for simulated Z-score (z_sim)
                         estimation (if permutations > 0)
    :returns: the Moran's statistics
    :rtype: esda.moran.Moran
    """
    # Default Moran's transformation is row-standardized "r".
    # https://github.com/pysal/esda/blob/main/esda/moran.py
    return esda.moran.Moran(y, w, permutations=permutations)


def polarization_scores_component(
    graph: Graph,
    component_id: str,
    normalization: PolarizationNormalizationTypes = "clr",
    permutations: int = 0,
) -> pd.DataFrame:
    """Calculate Moran's I statistics for a component.

    Computes polarization statistics for all antibodies in the `graph` given
    as input (a single connected component). The statistics are computed using
    Moran's I autocorrelation to measure how clustered/localized the spatial
    patterns of the antibody is in the graph. Spatial weights (`w`) are derived
    directly from the graph. The statistics contain the I value, the p-value,
    the adjusted p-value and the z-score under the randomization assumption.
    The function returns a pd.DataFrame with the following columns:
      morans_i, morans_p_value, morans_z, marker, component (morans_p_value_sim
      and morans_z_sim if `permutations` > 0)
    :param graph: a graph (it must be a single connected component)
    :param component_id: the id of the component
    :param normalization: the normalization method to use (raw or clr)
    :param permutations: the number of permutations for simulated Z-score (z_sim)
                         estimation (if permutations>0)
    :returns: a pd.DataFrame with the polarization statistics for each antibody
    :rtype: pd.DataFrame
    :raises: AssertionError when the input is not valid
    """
    if len(graph.connected_components()) > 1:
        raise AssertionError("The graph given as input is not a single component")

    if normalization not in get_args(PolarizationNormalizationTypes):
        raise AssertionError(f"incorrect value for normalization {normalization}")

    logger.debug(
        "Computing polarization scores for component %s with %i nodes",
        component_id,
        graph.vcount(),
    )

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

    # create antibody node counts
    counts_df = create_node_markers_counts(graph=graph, k=0)

    # remove markers with zero variance
    counts_df = counts_df.loc[
        :, (counts_df != 0).any(axis=0) & (counts_df.nunique() > 1)
    ]

    # clr transformation
    if normalization == "clr":
        counts_df = clr_transformation(df=counts_df, non_negative=True, axis=0)

    # compute the spatial weights matrix (w) from the graph
    w = WSP(graph.get_adjacency_sparse()).to_W(silence_warnings=True)

    # compute polarization statistics for each marker using the spatial weights (w)
    # and the markers counts distribution (y) (Morans I autocorrelation)
    def data():
        for m in counts_df.columns:
            yield morans_autocorr(w, counts_df[m], permutations)

    if permutations > 0:
        df = pd.DataFrame(
            ((m.I, m.p_rand, m.z_rand, m.p_sim, m.z_sim) for m in data()),
            columns=[
                "morans_i",
                "morans_p_value",
                "morans_z",
                "morans_p_value_sim",
                "morans_z_sim",
            ],
        ).fillna(0)
    else:
        df = pd.DataFrame(
            ((m.I, m.p_rand, m.z_rand) for m in data()),
            columns=[
                "morans_i",
                "morans_p_value",
                "morans_z",
            ],
        ).fillna(0)

    df["marker"] = counts_df.columns.tolist()
    df["component"] = component_id

    logger.debug("Polarization scores for components %s computed", component_id)
    return df


def polarization_scores(
    edgelist: pd.DataFrame,
    use_full_bipartite: bool = False,
    normalization: PolarizationNormalizationTypes = "clr",
    permutations: int = 0,
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
    :param normalization: the normalization method to use (raw or clr)
    :param permutations: the number of permutations for simulated Z-score (z_sim)
                         estimation (if permutations>0)
    :returns: a pd.DataFrames with all the polarization scores
    :rtype: pd.DataFrame
    :raises: AssertionError when the input is not valid
    """
    if "component" not in edgelist.columns:
        raise AssertionError("Edge list is missing the component column")

    logger.debug(
        "Computing polarization for edge list with %i elements",
        edgelist.shape[0],
    )

    # we make the computation in parallel (for each component)
    with futures.ThreadPoolExecutor() as executor:
        tasks = []
        for component_id, component_df in edgelist.groupby("component", observed=True):
            # build the graph from the component
            graph = Graph.from_edgelist(
                edgelist=component_df,
                add_marker_counts=True,
                simplify=False,
                use_full_bipartite=use_full_bipartite,
            )
            tasks.append(
                executor.submit(
                    polarization_scores_component,
                    graph=graph,
                    component_id=component_id,
                    normalization=normalization,
                    permutations=permutations,
                )
            )
        results = futures.wait(tasks)
        data = []
        for result in results.done:
            if result.exception() is not None:
                raise result.exception()  # type: ignore
            scores = result.result()
            data.append(scores)

    # create dataframe with all the scores
    scores = pd.concat(data, axis=0)
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
