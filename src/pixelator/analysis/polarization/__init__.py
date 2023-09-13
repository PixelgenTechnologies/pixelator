"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

import warnings

from esda.moran import Moran

from pixelator.graph.utils import Graph, create_node_markers_counts
from pixelator.statistics import (
    binarize_counts,
    clr_transformation,
    correct_pvalues,
    denoise,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module="libpysal")
    from libpysal.weights import WSP

import logging
from concurrent import futures
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd

from pixelator.pixeldataset import (
    MIN_VERTICES_REQUIRED,
)

logger = logging.getLogger(__name__)


def morans_autocorr(w: np.ndarray, y: np.ndarray) -> Any:
    """
    A helper function that computes the Moran's I autocorrelation statistics
    for the given spatial weights (w) and target variable (y). The target
    variable can be the counts of a specific antibody so the Moran's I value
    could indicate if the antibody has a localized spatial pattern. The function
    returns an object with the Moran's statistics (I, p_rand and z_rand).
    :param w: the weights matrix (connectivity matrix)
    :param y: the counts vector
    :returns: the Moran's statistics
    """
    # Default Moran's transformation is row-standardized "r".
    # https://github.com/pysal/esda/blob/main/esda/moran.py
    return Moran(y, w, permutations=None)


def polarization_scores_component(
    graph: Graph,
    component_id: str,
    normalization: Literal["raw", "clr", "denoise"] = "clr",
    antibody_control: Optional[List[str]] = None,
    binarization: bool = False,
) -> pd.DataFrame:
    """
    A helper function that computes a matrix of polarization statistics
    (one for each antibody) for the `graph` given as input (it must be a
    single connected component). The statistics are computed using Moran's I
    autocorrelation to measure how clustered/localized the spatial
    patterns of the antibody is in the graph. Spatial weights (w) are derived
    directly from the graph. The statistics contain the I value, the p-value,
    the adjusted p-value and the z-score under the randomization assumption.
    The function returns a pd.DataFrame with the following columns:
      morans_i,morans_p_value,morans_z,marker,component
    :param graph: a graph (it must be a single connected component)
    :param component_id: the id of the component
    :param normalization: the normalization method to use (raw or clr or denoise)
    :param antibody_control: the antibodies to use as control for the denoise option
    :param binarization: transform the counts to 0-1 (binarize)
    :returns: a pd.DataFrame with the polarization statistics for each antibody
    :raises: AssertionError when the input is not valid
    """
    if len(graph.connected_components()) > 1:
        raise AssertionError("The graph given as input is not a single component")

    if normalization not in ["raw", "clr", "denoise"]:
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

    # remove the empty markers and any column where all values are the
    # same, since Morans I is not defined when there is no variance
    counts_df = counts_df.loc[
        :, (counts_df != 0).any(axis=0) & (counts_df.nunique() > 1)
    ]

    # clr transformation
    if normalization != "raw":
        counts_df = clr_transformation(df=counts_df, non_negative=True, axis=0)

    # denoise
    if normalization == "denoise":
        counts_df = denoise(
            df=counts_df,
            antibody_control=antibody_control,  # type: ignore[arg-type]
            quantile=1.0,
            axis=0,
        )

    # binarize
    if binarization:
        counts_df = binarize_counts(df=counts_df)

    # compute the spatial weights matrix (w) from the graph
    w = WSP(graph.get_adjacency_sparse()).to_W(silence_warnings=True)

    # compute polarization statistics for each marker using the spatial weights (w)
    # and the markers counts distribution (y) (Morans I autocorrelation)
    statistics = []
    for m in counts_df.columns:
        mir = morans_autocorr(w, counts_df[m])
        statistics.append(mir)

    # create the dataframe
    df = pd.DataFrame(
        data={
            "morans_i": [m.I for m in statistics],
            "morans_p_value": [m.p_rand for m in statistics],
            "morans_z": [m.z_rand for m in statistics],
        },
    ).fillna(0)
    # the p-values are adjusted per-component
    df["marker"] = counts_df.columns.tolist()
    df["component"] = component_id

    logger.debug("Polarization scores for components %s computed", component_id)
    return df


def polarization_scores(
    edgelist: pd.DataFrame,
    use_full_bipartite: bool = True,
    normalization: Literal["raw", "clr", "denoise"] = "clr",
    antibody_control: Optional[List[str]] = None,
    binarization: bool = False,
) -> pd.DataFrame:
    """
    A helper function that given an `edgelist` will compute polarization scores.
    The function iterates all the components to compute polarization scores
    for each antibody in the component. The scores are computed using Moran's
    I autocorrelation to measure how clustered/localised the spatial patterns of
    the antibody is in the component's graph. Spatial weights (w) are derived
    directly from the graph. The function returns a pd.DataFrame with the following
    columns:
      morans_i,morans_p_value,morans_z,morans_p_adjusted,marker,component
    :param edgelist: an edge list (pd.DataFrame) with a component column
    :param use_full_bipartite: use the bipartite graph instead of the projection (UPIA)
    :param normalization: the normalization method to use (raw or clr or denoise)
    :param antibody_control: the antibodies to use as control for the denoise method
    :param binarization: transform the counts to 0-1 (binarize)
    :returns: a pd.DataFrames with all the polarization scores
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
        for component_id, component_df in edgelist.groupby("component"):
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
                    antibody_control=antibody_control,
                    binarization=binarization,
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
