"""Adaptive Core Expansion (ACE) graph partitioning implementation.

Copyright © 2026 Pixelgen Technologies AB
"""

from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

from pixelator.common.graph import Graph
from pixelator.common.graph.backends.implementations._networkx import _mat_pow
from pixelator.common.utils import logger


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
    binding_thresholds: Sequence[float] = (
        0.5,
        0.475,
        0.45,
        0.425,
        0.4,
        0.375,
        0.35,
        0.325,
        0.3,
    ),
    max_iter: int = 200,
    min_seed_pct: float = 0.1,
    nodes_to_move_threshold: int = 10,
) -> Graph:
    """Perform Adaptive Core Expansion (ACE) graph partitioning.

    Adaptive Core Expansion (ACE) is a topology-aware graph partitioning by
    identifying a high-density k-core "seed" and iteratively expanding it.
    The algorithm uses transition probabilities to recruit nodes from the
    periphery ("low" layer) into the core ("high" layer) and selects the
    final partition that maximizes phenotypic dissimilarity (Bray-Curtis)
    between the two groups.

    Args:
        cg: A `Graph` object containing the cell graph and node counts.
        k: The neighborhood radius (number of steps) used to calculate reachability
        max_k_core: Integer to cap the maximum k-core layer used for seeding.
        binding_thresholds: Sequence of thresholds for moving nodes from the low to high partition.
        max_iter: Maximum iterations per binding threshold.
        min_seed_pct: Minimum fraction of nodes required to form the initial seed partition.
        nodes_to_move_threshold: Convergence limit; stops iteration if fewer nodes move.

    Returns:
        The original graph object with an additional `partition` node attribute ("high" or "low").

    Raises:
        ValueError: If the graph does not contain any k-core layers above 1.
        ValueError: If no k-core layer meets the required 'min_seed_pct' threshold.
        ValueError: If the Graph object does not contain count data.
        ValueError: If the 'k' parameter is not between 1 and 6 inclusive.
        ValueError: If the 'max_k_core' parameter is not between 2 and 10 inclusive.
        ValueError: If the 'binding_thresholds' parameter is not a sequence of floats between 0 and 1.
        ValueError: If the 'max_iter' parameter is not between 1 and 1000 inclusive.
        ValueError: If the 'min_seed_pct' parameter is not between 0 and 1 inclusive.
        ValueError: If the 'nodes_to_move_threshold' parameter is not between 0 and 1000 inclusive.
        TypeError: If the 'binding_thresholds' parameter is not a sequence of floats between 0 and 1.

    """
    _validate_ace_parameters(
        k=k,
        max_k_core=max_k_core,
        binding_thresholds=binding_thresholds,
        max_iter=max_iter,
        min_seed_pct=min_seed_pct,
        nodes_to_move_threshold=nodes_to_move_threshold,
    )

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

    logger.debug(
        f"Seed 'high' core layer contains {np.mean(k_cores == max_k) * 100:.2f}% of all nodes."
    )

    # Compute transition probability matrix P from the adjacency matrix
    A = cg.get_adjacency_sparse(node_ordering=node_list)
    A = A + sp.diags_array([1] * A.shape[0], format="csr", dtype=None)

    row_sums = np.ravel(A.sum(axis=1))
    D_inv = sp.diags_array(1 / row_sums, format="csr")
    P = D_inv @ A

    min_weight = 1e-5  # To avoid having the sparse matrix grow too dense
    P_step = _mat_pow(P, k, prune_threshold=min_weight).T

    P_step.setdiag(0)

    row_sums = np.ravel(P_step.sum(axis=1))
    D_inv = sp.diags_array(1 / row_sums, format="csr")
    P_step = D_inv @ P_step

    partitions = []
    current_partition = (k_cores == max_k).astype(int)
    # Sort thresholds descending to allow reuse of the partition vector
    sorted_thresholds = sorted(binding_thresholds, reverse=True)

    for binding_threshold in sorted_thresholds:
        logger.debug(
            f"Finding partition seeded with k-core layer >= {max_k} and a binding threshold of {binding_threshold}."
        )

        current_partition = _adaptive_core_expansion_inner(
            current_partition,
            P_step,
            max_iter,
            nodes_to_move_threshold,
            binding_threshold,
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

        logger.debug("Completed")

        partitions.append((current_partition.copy(), bc_score))

    best_idx = int(np.argmax([p[1] for p in partitions]))
    best_partition_vec, best_score = partitions[best_idx]
    best_binding_threshold = sorted_thresholds[best_idx]

    logger.debug(
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
) -> np.ndarray:
    for i in range(max_iter):
        tp_total = P @ partition

        to_move_mask = (partition == 0) & (tp_total > binding_threshold)
        num_to_move = to_move_mask.sum()

        if num_to_move < nodes_to_move_threshold:
            logger.debug(f"Convergence reached at iteration {i}.")
            break

        partition[to_move_mask] = 1

    logger.debug(
        f"Final 'high' core layer contains {partition.sum() / len(partition) * 100:.2f}% of all nodes."
    )

    return partition


def _validate_ace_parameters(
    k, max_k_core, binding_thresholds, max_iter, min_seed_pct, nodes_to_move_threshold
):
    if not 1 <= k <= 6:
        raise ValueError(f"'k' must be between 1 and 6 inclusive, got {k}.")
    if not 2 <= max_k_core <= 10:
        raise ValueError(
            f"'max_k_core' must be between 2 and 10 inclusive, got {max_k_core}."
        )
    if not isinstance(binding_thresholds, Sequence):
        raise TypeError(
            "'binding_thresholds' must be a sequence of floats between 0 and 1."
        )
    if not all(0 <= t <= 1 for t in binding_thresholds):
        raise ValueError(
            "All values in 'binding_thresholds' must be between 0 and 1 inclusive."
        )
    if not 1 <= max_iter <= 1000:
        raise ValueError(
            f"'max_iter' must be between 1 and 1000 inclusive, got {max_iter}."
        )
    if not 0 <= min_seed_pct <= 1:
        raise ValueError(
            f"'min_seed_pct' must be between 0 and 1 inclusive, got {min_seed_pct}."
        )
    if not 0 <= nodes_to_move_threshold <= 1000:
        raise ValueError(
            f"'nodes_to_move_threshold' must be between 0 and 1000 inclusive, got {nodes_to_move_threshold}."
        )
    return True
