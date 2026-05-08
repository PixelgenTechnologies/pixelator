"""Adaptive Core Expansion (ACE) graph partitioning implementation.

Copyright © 2026 Pixelgen Technologies AB
"""

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

from pixelator.common.graph import Graph
from pixelator.common.graph.backends.implementations._networkx import _mat_pow
from pixelator.common.utils import logger


@dataclass(frozen=True)
class _PartitionCandidate:
    partition: np.ndarray
    bc_score: float
    nodes_pct: float
    binding_threshold: float


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
    min_allowed_nodes_pct: float = 0.8,
    select_LCC: bool = True,
) -> Graph:
    """Perform Adaptive Core Expansion (ACE) graph partitioning.

    ACE performs a topology-aware graph partitioning by identifying a high-density
    k-core "seed" and iteratively expanding it. The algorithm uses transition
    probabilities to recruit nodes from the periphery ("low" layer) into the
    core ("high" layer) and selects the final partition that maximizes
    phenotypic dissimilarity (Bray-Curtis) between the two groups.

    The algorithm proceeds in four main stages:
    1. Seed Identification: Computes node k-coreness and identifies the maximum
       core layer (k_max). To ensure a robust starting point, the seed is capped
       at max_k_core and automatically downgraded to lower core levels until it
       meets the min_seed_pct threshold.
    2. k-Step Influence Modeling: Constructs a transition probability matrix P
       derived from a k-step reachability matrix. This captures the cumulative
       "binding strength" of a node based on its connectivity within a k-distance
       radius, accounting for indirect paths and local density.
    3. Iterative Expansion: For a range of binding_thresholds, nodes are moved
       from the "low" to the "high" partition if their total transition probability
       to the current "high" set exceeds the threshold.
    4. Dissimilarity Optimization: Evaluates each result using the Bray-Curtis
       dissimilarity score. The partition with the highest score that meets the
       min_allowed_nodes_pct requirement is selected.

    Args:
        cg: A `Graph` object containing the cell graph and node count data.
        k: The neighborhood radius (number of steps) used for reachability.
            Larger values increase the "reach" of the core.
        max_k_core: Cap for the maximum k-core layer used for seeding.
        binding_thresholds: Thresholds for moving nodes to the high partition.
                               If None, a default sequence from 0.5 to 0.3 is used.
        max_iter: Maximum iterations per binding threshold.
        min_seed_pct: Minimum fraction of nodes required for the initial seed.
        nodes_to_move_threshold: Convergence limit; iteration stops if fewer
                                    nodes move.
        min_allowed_nodes_pct: Minimum fraction of nodes required in the final
                                  "high" core partition.
        select_LCC:  If True, restricts the initial seed to the Largest
                       Connected Component.

    Returns:
        The original Graph object with an additional `partition` node attribute ("high" or "low").

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
        ValueError: If the 'min_allowed_nodes_pct' parameter is not between 0 and 1 (exclusive).
        TypeError: If the 'binding_thresholds' parameter is not a sequence of floats between 0 and 1.

    """
    _validate_ace_parameters(
        k=k,
        max_k_core=max_k_core,
        binding_thresholds=binding_thresholds,
        max_iter=max_iter,
        min_seed_pct=min_seed_pct,
        nodes_to_move_threshold=nodes_to_move_threshold,
        min_allowed_nodes_pct=min_allowed_nodes_pct,
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

    if select_LCC:
        seed_nodes = [node_list[i] for i, k_val in enumerate(k_cores) if k_val == max_k]
        subgraph = raw_graph.subgraph(seed_nodes)
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            logger.debug(
                f"The high k-core layer has {len(components)} connected components. "
                f"Selecting the largest connected component."
            )
            largest_comp = max(components, key=len)
            largest_comp_set = set(largest_comp)
            for i, k_val in enumerate(k_cores):
                if k_val == max_k and node_list[i] not in largest_comp_set:
                    k_cores[i] = max_k - 1

    partitions: list[_PartitionCandidate] = []
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

        partitions.append(
            _PartitionCandidate(
                partition=current_partition.copy(),
                bc_score=bc_score,
                nodes_pct=float(num_high / len(current_partition)),
                binding_threshold=binding_threshold,
            )
        )

    best_idx = int(np.argmax([p.bc_score for p in partitions]))
    best_candidate: _PartitionCandidate | None = partitions[best_idx]

    if best_candidate is not None and best_candidate.nodes_pct < min_allowed_nodes_pct:
        logger.warning(
            f"The selected partition has less than {min_allowed_nodes_pct * 100:.2f}% "
            f"of nodes in the 'high' core layer. Selecting the partition with the highest "
            f"binding threshold that has at least {min_allowed_nodes_pct * 100:.2f}% "
            f"of nodes in the 'high' core layer instead."
        )

        # Select partitions that meet the minimum node percentage
        valid_partitions: list[_PartitionCandidate] = [
            p for p in partitions if p.nodes_pct >= min_allowed_nodes_pct
        ]

        if valid_partitions:
            # Pick the one with the highest BC score among those that meet the threshold
            best_candidate = max(valid_partitions, key=lambda x: x.bc_score)
        else:
            logger.warning(
                f"Found no partition with at least {min_allowed_nodes_pct * 100:.2f}% of "
                f"nodes in the 'high' core layer. Setting all nodes as 'high'."
            )
            best_candidate = None

    if best_candidate is not None:
        best_partition_vec = best_candidate.partition
        logger.debug(
            f"Selected partition seeded with k-core layer >= {max_k} "
            f"and a binding threshold of {best_candidate.binding_threshold}. "
            f"Bray-Curtis dissimilarity score: {best_candidate.bc_score:.4f}. "
            f"Percent of nodes in 'high' core layer: {best_candidate.nodes_pct * 100:.2f}%."
        )
    else:
        best_partition_vec = np.ones(len(node_list), dtype=int)
        logger.debug("No suitable partition found. All nodes set to 'high'.")

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
    k: int,
    max_k_core: int,
    binding_thresholds: Sequence[float],
    max_iter: int,
    min_seed_pct: float,
    nodes_to_move_threshold: int,
    min_allowed_nodes_pct: float,
) -> bool:
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
    if not 0 <= min_allowed_nodes_pct < 1:
        raise ValueError(
            f"'min_allowed_nodes_pct' must be between 0 and 1 (exclusive), got {min_allowed_nodes_pct}."
        )
    return True
