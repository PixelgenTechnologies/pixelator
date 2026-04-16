import logging
import warnings
from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

from pixelator.common.graph import Graph
from pixelator.common.graph.backends.implementations._networkx import _mat_pow

logger = logging.getLogger(__name__)


def normalize_counts(x):
    """Normalize counts using log-mean-exp normalization."""
    # counts_mat: shape (n_nodes, n_markers)
    x_log1p = np.log1p(x)
    x_mean = np.mean(x_log1p)
    norm = np.exp(x_mean)
    return np.log1p(x / norm)


def adaptive_core_expansion(
    cg: Graph,
    k: int = 3,
    max_k_core: int = 4,
    binding_thresholds: Sequence[float] = (0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3),
    max_iter: int = 200,
    min_seed_pct: float = 0.1,
    nodes_to_move_threshold: int = 10,
    select_LCC: bool = True,
    verbose: bool = True,
) -> Graph:
    """Performs a topology-aware graph partitioning by identifying a high-density
    k-core "seed" and iteratively expanding it.

    The algorithm uses transition probabilities to recruit nodes from the periphery ("low" layer)
    into the core ("high" layer) and selects the final partition that maximizes
    phenotypic dissimilarity (Bray-Curtis) between the two groups.

    :param cg: A `Graph` object containing the cell graph and node counts.
    :param k: The neighborhood radius (number of steps) used to calculate reachability
    :param max_k_core: Integer to cap the maximum k-core layer used for seeding.
    :param binding_thresholds: Sequence of thresholds for moving nodes from the low to high partition.
    :param max_iter: Maximum iterations per binding threshold.
    :param min_seed_pct: Minimum fraction of nodes required to form the initial seed partition.
    :param nodes_to_move_threshold: Convergence limit; stops iteration if fewer nodes move.
    :param select_LCC: Restricts the initial seed to the Largest Connected Component.
    :param verbose: Whether to print progress alerts.
    :return: The original graph object with an additional `partition` node attribute ("high" or "low").
    """
    assert 1 <= k <= 6
    assert 2 <= max_k_core <= 10
    assert all(0 <= t <= 1 for t in binding_thresholds)
    assert 1 <= max_iter <= 1000
    assert 0 <= min_seed_pct <= 1
    assert 0 <= nodes_to_move_threshold <= 1000

    try:
        counts = cg.node_marker_counts
    except AssertionError:
        raise ValueError("The Graph object must contain count data.")

    if counts.shape[1] < 2:
        raise ValueError("The 'counts' slot must contain at least two features.")

    raw_graph = cg.raw
    node_list = list(raw_graph.nodes())

    k_cores_dict = nx.core_number(raw_graph)
    k_cores = np.array([k_cores_dict[n] for n in node_list])
    max_k = k_cores.max()

    if max_k == 1:
        raise ValueError(
            "The graph does not contain any k-core layers above 1. The graph has too low connectivity."
        )

    if max_k > max_k_core:
        k_cores[k_cores > max_k_core] = max_k_core
        max_k = max_k_core

    pct_k_max = np.mean(k_cores == max_k)
    while pct_k_max < min_seed_pct and max_k > 1:
        max_k -= 1
        pct_k_max = np.mean(k_cores == max_k)

    if pct_k_max < min_seed_pct:
        raise ValueError(
            f"No k-core layer meets the required 'min_seed_pct' threshold of {min_seed_pct}."
        )

    k_cores[k_cores > max_k] = max_k

    # Compute transition probability matrix P from the adjacency matrix
    A = cg.get_adjacency_sparse(node_ordering=node_list)
    A = A + sp.diags_array([1] * A.shape[0], format="csr", dtype=None)

    row_sums = np.ravel(A.sum(axis=1))
    D_inv = sp.diags_array(1 / row_sums, format="csr")
    P = D_inv @ A

    min_weight = 0
    P_step = _mat_pow(P, k, prune_threshold=min_weight).T

    # Set diagonal to 0
    P_step.setdiag(0)

    row_sums = np.ravel(P_step.sum(axis=1))
    D_inv = sp.diags_array(1 / row_sums, format="csr")
    P_step = D_inv @ P_step

    if verbose:
        logger.info(
            f"Seed 'high' core layer contains {np.mean(k_cores == max_k)*100:.2f}% of all nodes."
        )

    if select_LCC:
        seed_nodes = [node_list[i] for i, k_val in enumerate(k_cores) if k_val == max_k]
        subgraph = raw_graph.subgraph(seed_nodes)
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            if verbose:
                logger.info(
                    f"The high k-core layer has {len(components)} connected components. Selecting the largest connected component."
                )
            largest_comp = max(components, key=len)
            largest_comp_set = set(largest_comp)
            for i, n in enumerate(node_list):
                if k_cores[i] == max_k and n not in largest_comp_set:
                    k_cores[i] = max_k - 1

    partitions = []
    current_partition = (k_cores == max_k).astype(int)
    # Sort thresholds descending to allow reuse of the partition vector
    sorted_thresholds = sorted(binding_thresholds, reverse=True)

    for binding_threshold in sorted_thresholds:
        if verbose:
            logger.info(
                f"Finding partition seeded with k-core layer >= {max_k} and a binding threshold of {binding_threshold}."
            )

        current_partition = _adaptive_core_expansion_inner(
            current_partition,
            P_step,
            max_iter,
            nodes_to_move_threshold,
            binding_threshold,
            verbose,
        )

        min_nodes_in_core = max(nodes_to_move_threshold, 10)
        num_high = current_partition.sum()

        if num_high < min_nodes_in_core or num_high == len(current_partition):
            bc_score = 0.0
        else:
            high_mask = current_partition == 1
            low_mask = ~high_mask

            # Use pandas direct indexing since node_marker_counts indices ordered same as node_list
            x = normalize_counts(counts[high_mask].sum(axis=0).values)
            y = normalize_counts(counts[low_mask].sum(axis=0).values)

            bc_score = float(np.sum(np.abs(x - y)) / np.sum(x + y))

        if verbose:
            logger.info("Completed")

        partitions.append((current_partition.copy(), bc_score))

    best_idx = int(np.argmax([p[1] for p in partitions]))
    best_partition_vec, best_score = partitions[best_idx]
    best_binding_threshold = sorted_thresholds[best_idx]

    if verbose:
        logger.info(
            f"Selected partition seeded with k-core layer >= {max_k} "
            f"and a binding threshold of {best_binding_threshold}. "
            f"Bray-Curtis dissimilarity score: {best_score:.4f}."
        )

    pixel_type_dict = {
        node_list[i]: ("high" if best_partition_vec[i] == 1 else "low")
        for i in range(len(node_list))
    }
    nx.set_node_attributes(raw_graph, pixel_type_dict, "partition")

    return cg


def _adaptive_core_expansion_inner(
    partition: np.ndarray,
    P: sp.csr_matrix,
    max_iter: int,
    nodes_to_move_threshold: int,
    binding_threshold: float,
    verbose: bool
) -> np.ndarray:
    for i in range(max_iter):
        tp_total = P @ partition

        to_move_mask = (partition == 0) & (tp_total > binding_threshold)
        num_to_move = to_move_mask.sum()

        if num_to_move < nodes_to_move_threshold:
            if verbose:
                logger.info(f"Convergence reached at iteration {i}.")
            break

        partition[to_move_mask] = 1

    if verbose:
        logger.info(f"Final 'high' core layer contains {partition.sum() / len(partition) * 100:.2f}% of all nodes.")

    return partition
