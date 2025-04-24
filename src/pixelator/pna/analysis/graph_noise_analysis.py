"""Plugin for computing post analysis metrics for R&D report.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
from copy import deepcopy
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

from pixelator.common.exceptions import PixelatorBaseException
from pixelator.pna.analysis_engine import PerComponentTask
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile

logger = logging.getLogger(__name__)


class SvdAnalysisError(PixelatorBaseException):
    """An exception for errors in SvdAnalysis.

    Currently used when trying to run SVD analysis on a graph with
    fewer nodes than the number of pivot points.
    """

    pass


DeadendOptions = Literal["remove", "ignore", "self-loop"]


def _mat_pow(mat, power):
    mat_power = mat
    for _ in range(power - 1):
        mat_power = mat @ mat_power
    return mat_power


def non_backtracking_transition_probabilities(
    graph: nx.Graph,
    n_steps: int = 3,
    cumulative: bool = True,
    deadend_action: DeadendOptions = "ignore",
    track_only_nonzero: bool = True,
) -> pd.Series:
    """Calculate the non-backtracking transition probabilities for an n_steps walk.

    A non-backtracking random walker can only transition on edges that are not the same as the
    edge it came from. The non-backtracking transition probabilities can be
    useful for identifying spurious edges in a graph. For example, a non-zero probability for
    a 3-step non-backtracking walk between two nodes i and j (connected by an edge) indicates
    that there is at least one path of length 3 connecting i and j. This can be an indication that
    i and j are located in the same neighborhood. If the probability is zero, there is no path of
    length 3 between i and j which is an indication that i and j are located in different neighborhoods.
    In the latter case, the edge between i and j is potentially spurious.

    One limitation with this method is that edges located in sparse regions of the graph often
    have zero probabilities. For example, when transitioning from i to j, if the end node j has
    a degree of 1, there are no paths of length > 1 and the probability will be zero for a k>1
    step non-backtracking walk. However, this is not an indication that i and j are located in
    different neighborhoods.

    :param graph: a networkx graph object.
    :param n_steps: the number of steps to take for the random walk.
    :param cumulative: if True, calculate the cumulative transition probabilities for all steps up to n_steps.
    :param deadend_action: the action to take for deadend nodes. Options are: "remove", "ignore", "self-loop".
    :param track_only_nonzero: if True, only track edges with non-zero probabilities without keeping the actual probablity.
    :return: a pandas series containing the transition probabilities.
    :raises ValueError: if the deadend_action is not one of the valid options.
    :raises ValueError: if the number of steps is less than 3.
    """
    valid_deadend_actions = ["remove", "ignore", "self-loop"]
    if deadend_action not in valid_deadend_actions:
        raise ValueError(
            f"Invalid value for deadend_action. Expected one of: {valid_deadend_actions}"
        )

    if n_steps < 3:
        raise ValueError("The number of steps must be at least 3.")

    if deadend_action == "remove":
        core_vals = nx.core_number(graph)
        to_remove = [i for i in core_vals.keys() if core_vals[i] == 1]
        graph = deepcopy(graph)
        graph.remove_nodes_from(to_remove)

    dir_g = nx.to_directed(graph)
    line_g = nx.line_graph(dir_g)

    # Remove edges that go back
    edgelist = list(line_g.edges)
    for edge in edgelist:
        if edge[0][0] == edge[1][1]:
            line_g.remove_edges_from([edge])

    adj_mat = nx.adjacency_matrix(line_g).astype(bool)

    if deadend_action == "self-loop":
        is_dead_end = (adj_mat.sum(axis=1) == 0).astype(float)
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(is_dead_end)
        adj_mat = adj_mat.tocsr().astype(bool)

    if track_only_nonzero:
        weighted_adj_mat = adj_mat
    else:
        weighted_adj_mat = (
            adj_mat / np.maximum(adj_mat.sum(axis=1).astype(float), 1)[:, None]
        )

    if cumulative:
        walk_probability = _mat_pow(weighted_adj_mat, 3)
        for i in range(3, n_steps):
            walk_probability += _mat_pow(weighted_adj_mat, i + 1)
    else:
        walk_probability = _mat_pow(weighted_adj_mat, n_steps)
    transition_probability = pd.Series(
        (adj_mat @ walk_probability).diagonal(), index=list(line_g.nodes)
    )

    return transition_probability


def svd_pivot_distances(
    g: nx.Graph, pivots: int = 100, seed: int | None = None
) -> tuple:
    """Single Value Decomposition of a graphs shortest path matrix.

    This function can be used to evaluate how well a graph can be represented
    in 3D space.

    Component graphs with high spatial resolution, i.e. where edges represent short distances
    in actual space, should be easier to embed in a euclidean space using multidimensional
    scaling techniques s.a. pMDS. In contrast, graphs with high spatial distortion or with
    a high amount of noise will be harder to embed. An example of such noise is the presence
    of edges which form long-range connections. Such connections will break the spatial
    structure of the graph and form shortest paths which are not representative of the actual
    spatial distances.

    The idea behind this method is that if a graph truly represents a spatial 3D structure,
    most of the variance should be explained by the first 3 singular vectors in the SVD. In
    the best case scenario, the first 3 singular vectors should explain 100% of the variance
    and all three singular vectors should contribute equally to the variance explained.

    The algorithm is outlined below:

    1. Select a number of pivot nodes at random.
    2. Calculate the shortest path length from the pivots to all other nodes.
    3. Double center the distance matrix.
    4. Perform SVD on the centered distance matrix.

    :param g: a networkx graph object
    :param pivots: the number of pivot nodes to use
    :param seed: the random seed to use
    :return: a tuple containing the results of the SVD
    :raises ValueError: the number of pivots must be less than the number of nodes in the graph
    """
    if pivots >= len(g.nodes):
        raise SvdAnalysisError(
            "'pivots' must be less than the number of nodes in the graph."
        )

    random_generator = np.random.default_rng(seed)

    node_list = list(g.nodes)

    # Select random pivot nodes
    pivs = random_generator.choice(node_list, pivots, replace=False)

    # Calculate the shortest path length from the pivots to all other nodes
    A = nx.to_scipy_sparse_array(g, weight=None, nodelist=node_list, format="csr")

    # Create shortest path matrix
    A.indices = A.indices.astype(np.intc, copy=False)
    A.indptr = A.indptr.astype(np.intc, copy=False)
    D = sp.sparse.csgraph.shortest_path(
        A,
        directed=False,
        unweighted=True,
        method="D",
        indices=np.where(np.isin(g.nodes, pivs))[0],
    ).T

    # Center values in rows and columns
    D2 = D**2
    cmean = np.mean(D2, axis=0)
    rmean = np.mean(D2, axis=1)
    D_pivs_centered = D2 - np.add.outer(rmean, cmean) + np.mean(D2)

    # Perform SVD on the centered distance matrix
    svd_res = np.linalg.svd(D_pivs_centered, full_matrices=False)

    return svd_res


def summarize_k_cores(g: nx.Graph) -> pd.DataFrame:
    """Summarize k-cores for a given component.

    Computes the coreness for all nodes in a given graph and summarizes the number
    of nodes per k-core in a table. The columns in the DataFrame are names 'k1', 'k2',
    ..., 'kM' where M is the maximum k-core value in the component. The k-core counts
    are useful for quantifying graph connectivity.

    :param g: a networkx Graph object
    :raises AssertionError: the input arguments are incorrect
    :return: a dataframe with the summarized node k-cores
    :rtype: pd.DataFrame
    """
    if not isinstance(g, nx.Graph):
        raise AssertionError("g must be a networkx graph object")

    k_core_nodes = pd.DataFrame.from_dict(
        nx.core_number(g), orient="index", columns=["k_core"]
    )

    average_k_core = k_core_nodes["k_core"].mean()

    # Summarize k-cores
    k_core_component_table = k_core_nodes.groupby("k_core").size()
    k_core_component_table = k_core_component_table.reset_index()
    k_core_component_table.columns = ["k_core", "count"]
    k_core_component_table["k_core"] = "k_core_" + k_core_component_table[
        "k_core"
    ].astype(str)
    k_core_component_table_wide = pd.pivot_table(
        k_core_component_table, values="count", columns="k_core"
    )
    k_core_component_table_wide.columns.name = None
    k_core_component_table_wide["average_k_core"] = average_k_core

    return k_core_component_table_wide


class KcoreAnalysis(PerComponentTask):
    """Run k-core decomposition on each component."""

    TASK_NAME = "k-core"

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pd.DataFrame:
        """Run k-core analysis on a single component.

        Calls summarize_k_cores on a component graph and returns a pandas DataFrame with k-core counts.
        The columns in the DataFrame are names 'k_core_1', 'k_core_2', ..., 'k_core_M' where M is the
        maximum k-core value in the component. The k-core counts are useful for quantifying graph connectivity.

        :param component: a networkx graph for a component to run the analysis on.
        :param component_id: the id of the component.
        :return: a pandas DataFrame containing k-core counts.
        """
        logger.debug(f"Running k-core analysis on component {component_id}")
        k_core_summary = summarize_k_cores(
            g=component.raw,
        )
        k_core_summary.index = [component_id]
        return k_core_summary

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile) -> None:
        """Add k-core count data for all components to adata.obs in pxl_dataset.

        :param data: a pandas DataFrame containing k-core counts for all components.
        :param pxl_file_target: the PxlFile to add the data to.
        """
        logger.debug("Adding k-core analysis data to PNAPixelDataset")
        data.fillna(0, inplace=True)
        data.sort_index(axis="columns", inplace=True)

        adata = PNAPixelDataset.from_files(pxl_file_target).adata()
        adata.obs.drop(
            columns=[col for col in adata.obs.columns if "k_core_" in col],
            inplace=True,
            errors="ignore",
        )
        adata.obs.drop(columns=["average_k_core"], inplace=True, errors="ignore")
        adata.obs = adata.obs.join(data)
        with PixelFileWriter(pxl_file_target.path) as writer:
            writer.write_adata(adata)


class SvdAnalysis(PerComponentTask):
    """Run SVD variance explained on each component."""

    TASK_NAME = "SVD"

    def __init__(self, pivots: int = 100):
        """Initialize SvdAnalysis.

        :param pivots: the number of pivot points to use for SVD analysis.
        """
        self.pivots = pivots

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pd.DataFrame:
        """Run SVD analysis on component.

        Calls svd_pivot_distances on a component graph and returns a pandas DataFrame with variance
        explained for the first three singular vectors. The columns in the DataFrame are named 'svd_var_expl_s1',
        'svd_var_expl_s2', 'svd_var_expl_s3'. The variance explained is useful for investigating spatial coherence
        in the graph. In the best case scenario, the first three singular vectors contribute equally to the variance
        explained, and together they should explain 100% of the total variance explained.

        :param component: a networkx graph for a component to run the analysis on.
        :param component_id: the id of the component.
        :return: a pandas DataFrame containing variance explained for the first three singular vectors.
        If the SVD computation fails, e.g. if the number of pivot points is >= the number of nodes,
        the function returns a DataFrame with nan values.
        """
        logger.debug(f"Running SVD analysis on component {component_id}")

        try:
            # Compute SVD and fetch singular values
            _, s, _ = svd_pivot_distances(component.raw, seed=123, pivots=self.pivots)

            # Compute variance explained from singular values
            s_var = (s**2) / sum(s**2)

            svd_variance_explained = pd.DataFrame(
                {
                    "svd_var_expl_s1": [s_var[0]],
                    "svd_var_expl_s2": [s_var[1]],
                    "svd_var_expl_s3": [s_var[2]],
                },
                index=[component_id],
            )
        except SvdAnalysisError as e:
            # If SVD fails, return a DataFrame with nan values
            logger.warning(
                f"Failed to run SVD analysis on component {component_id}: {e}"
            )
            svd_variance_explained = pd.DataFrame(
                {
                    "svd_var_expl_s1": [np.nan],
                    "svd_var_expl_s2": [np.nan],
                    "svd_var_expl_s3": [np.nan],
                },
                index=[component_id],
            )

        return svd_variance_explained

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile) -> None:
        """Add svd variance explained for the first three singular vectors.

        Add svd variance explained for the first three singular vectors across
        all components to adata.obs in pxl_dataset.

        :param data: a pandas DataFrame containing svd variance explained for the first three
         singular vectors across all components.
        :param pxl_dataset: the PNAPixelDataset to add the data to.
        """
        logger.debug("Adding SVD analysis data to PNAPixelDataset")
        adata = PNAPixelDataset.from_files(pxl_file_target.path).adata()
        adata.obs.drop(
            columns=[col for col in adata.obs.columns if "svd_var_expl_" in col],
            inplace=True,
        )
        adata.obs = adata.obs.join(data)
        with PixelFileWriter(pxl_file_target.path) as writer:
            writer.write_adata(adata)
