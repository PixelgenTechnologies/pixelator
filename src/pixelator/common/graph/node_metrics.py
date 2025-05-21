"""Functions related to computing node metrics on graphs.

Copyright © 2025 Pixelgen Technologies AB.
"""

from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp

from pixelator.common.marks import experimental


@experimental
def local_g(
    A: sp.sparse.csr_array,
    counts: pd.DataFrame,
    k: int = 1,
    use_weights: bool = True,
    normalize_counts: bool = False,
    W: sp.sparse.csr_array | None = None,
    method: Literal["gi", "gstari"] = "gi",
) -> pd.DataFrame:
    """Compute local G-scores for each node and marker.

    Local G([1]_) is a spatial node metric that measures the spatial association.
    The metric can for instance be used to detect hot spots for marker counts in a graph, where nodes
    that are close to each other have similar values. The metric is a Z-score that
    measures the deviation of the observed local marker expression from the expected marker
    expression under the null hypothesis of no spatial association. The sign of the score
    indicates whether the observed marker counts are higher or lower than expected
    and can therefore be used to identify hot and/or cold spots.

    The observed local marker expression for a node is computed by aggregating the weighted marker
    expression within its local neighborhood. The local G metric is largely influenced by the choice
    of edge weights and the size of the local neighborhood (`k`). The method implemented here uses
    incoming transition probabilities for a k-step random walk as edge weights. By increasing `k`,
    the local neighborhood of a node is expanded, increasing the "smoothing effect" of the metric,
    which can be useful to increase the scale of spatial association.

    Local G can also be useful for more interpretable visualization of polarized marker expression
    as it enhances spatial trends across neighborhoods in the graph, even if the marker counts within
    individual nodes are sparse.

    Note that it is important that the node ordering in the sparse adjacency matrix, and
    the counts matrix line up. If calling this method directly the caller is responsible for
    ensuring that this contract is fulfilled.

    .. [1] Bivand, R.S., Wong, D.W.S. Comparing implementations of global and
    local indicators of spatial association. TEST 27, 716–748 (2018).
    https://doi.org/10.1007/s11749-018-0599-x

    :param A: A sparse adjacency matrix representing the graph.
    :param counts: A DataFrame of marker counts for each node.
    :param k: The number of steps in the k-step random walk. Default is 1.
    :param use_weights: Whether to use weights in the computation. When turned off, all
    edge weights will be qeual to 1. Default is True.
    :param normalize_counts: Whether to normalize counts to proportions. Default is False.
    :param W: A sparse matrix of custom edge weights. This will override the automated
    computation of edge weights. `W` must have the same dimensions as A. Note that weights can
    be defined for any pair of nodes, not only the pairs represented by edges in `A`. Default is None.
    :param method: The method to use for computing local G. Must be one of 'gi' or 'gstari'.
    'gi' is the original local G metric, which does not consider self-loops, meaning that the
    local marker expression for a node is computed by aggregating the weighted expression of
    its neighbors. 'gstari' is a simplified version of local G that does consider self-loops.
    In other words, the local marker expression of a node also includes the weighted marker
    expression of the node itself. Default is 'gi'.
    :return: A DataFrame of local G-scores for each node and marker.
    :rtype: pd.DataFrame
    """
    # Check that type is one of 'gi' or 'gstari'
    if method not in ["gi", "gstari"]:
        raise ValueError("type must be one of 'gi' or 'gstari'")

    n_nodes = A.shape[0]

    marker_columns = counts.columns
    node_indices = counts.index
    counts = sp.sparse.csr_array(counts.values, dtype=np.float64)

    # Number of nodes in G must match the number of rows in counts
    if n_nodes != counts.shape[0]:
        raise ValueError(
            "The number of nodes in G must match the number of rows in counts"
        )

    # Check that k is a positive integer
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    # Convert marker counts to proportions if normalize_counts is True
    if normalize_counts:
        row_sums = counts.sum(axis=1)
        row_sums[np.abs(row_sums) < np.finfo(counts.dtype).eps] = 1
        counts = counts.multiply(1 / row_sums[:, None]).tocsr()

    if W is not None:
        # Raise ValueError if W is not a sparse matrix
        if not sp.sparse.issparse(W):
            raise ValueError("W must be a sparse matrix")
        # Raise ValueError if the dimensions of W differ from the number of nodes in G
        if W.shape[0] != n_nodes or W.shape[1] != n_nodes:
            raise ValueError("The dimensions of W must match the number of nodes in G")
    if not W:
        if use_weights:
            if method == "gstari":
                # Set diagonal of A to 1 for gstari which expects self-loops
                A = A + sp.sparse.eye(n_nodes)
            W = compute_transition_probabilities(
                A, k=k, remove_self_loops=(method == "gi")
            ).tocsr()
            # Transpose W to get incoming transition probabilities
            # This eliminates the correlation between local G and node degree
            W = W.T
        else:
            W = A
            if k > 1:
                # Expand local neighborhood using matrix powers
                W = sp.sparse.linalg.matrix_power(W, k)
                # Set all positive elements to 1
                W.data = np.ones_like(W.data)
                if method == "gstari":
                    # Set diagonal of A to 1 for gstari which expects self-loops
                    W = W + sp.sparse.eye(n_nodes)
                else:
                    # Set diagonal of W to 0 for gi to avoid self-loops
                    W.setdiag(values=0)

    # Compute lag matrix
    lag_mat = W @ counts

    if method == "gstari":
        # Compute xibar for each node and marker
        xibar_mat = np.tile(counts.mean(axis=0), n_nodes).reshape(counts.shape)
        # Compute si for each node and marker
        s_mat = np.tile(
            ((counts**2).sum(axis=0) / n_nodes) - (counts.sum(axis=0) / n_nodes) ** 2,
            n_nodes,
        ).reshape(counts.shape)

    if method == "gi":
        # Compute xibar for each node and marker, excluding i
        xibar_mat = (
            np.tile(counts.sum(axis=0), n_nodes).reshape(counts.shape) - counts
        ) / (n_nodes - 1)

        # Compute si for each node and marker, excluding i
        s_mat = (
            (
                np.tile((counts**2).sum(axis=0), n_nodes).reshape(counts.shape)
                - (counts**2)
            )
            / (n_nodes - 1)
        ) - (xibar_mat**2)

    # Calculate weights for each node. If k = 1 and use_weights=False,
    # the weights should be equal to the node degree
    weights_i = W.sum(axis=1)[:, None]
    square_weights_i = (W**2).sum(axis=1)[:, None]

    # Calculate expected value
    E_G = xibar_mat * weights_i

    # Calculate numerator G - E(G)
    numerator_mat = lag_mat - E_G

    # Calculate denomenator Var(G)
    if method == "gstari":
        var_weights = ((n_nodes * square_weights_i) - weights_i**2) / (n_nodes - 1)
        denominator_mat = np.sqrt(s_mat * var_weights)

    if method == "gi":
        var_weights = (((n_nodes - 1) * square_weights_i) - weights_i**2) / (
            n_nodes - 2
        )
        denominator_mat = np.sqrt(s_mat * var_weights)

    # Calculate Z-score
    gi_mat = numerator_mat / denominator_mat

    gi_mat[~np.isfinite(gi_mat)] = 0

    return pd.DataFrame(gi_mat, index=node_indices, columns=marker_columns)


def compute_transition_probabilities(
    A: sp.sparse.csr_array,
    k: int = 1,
    remove_self_loops: bool = False,
) -> sp.sparse.csr_array:
    """Compute transition probabilities of a graph.

    This function computes the transition probabilities for node pairs in a graph.
    Transition probabilities can for example be useful for weighting marker counts
    when computing local spatial node metrics such as local G. The transition
    probability is the probability of going from one node to another in a k-step walk,
    where `k` is the number of steps in the walk. When `k=1`, the transition probabilities
    are defined for the edges in the graph. With `k>1`, the transition probabilities can
    be positive for node pairs that are not directly connected by an edge, including
    transitions from a node to itself. In some situations, it may be desirable to ignore
    these transitions (self-loops), which can be achieved by setting `remove_self_loops=True`.
    In this case, the diagonal of the transition probability matrix is set to 0 and the
    probabilities are renormalized (row-wise) to sum to 1.

    :param A: A sparse adjacency matrix representing the graph.
    :param k: The number of steps in the random walk. Default is 1.
    :param remove_self_loops: Whether to remove self-loops from the transition probability
    matrix. Default is False.
    :return: A sparse matrix with transition probabilities.
    :rtype: sp.sparse.csr_matrix
    """
    # Check that k is a positive integer
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    W_out = A / A.sum(axis=0)[:, None]

    # Multiply W by itself k times
    if k > 1:
        W_out = sp.sparse.linalg.matrix_power(W_out, k)

    if remove_self_loops and k > 1:
        # Set diagonal of W to 0 if k > 1 for gi to avoid self-loops
        W_out.setdiag(values=0)
        # Renormalize transition probabilities to sum to 1
        row_sums = np.array(W_out.sum(axis=1)).flatten()
        inv_row_sums = np.reciprocal(row_sums, where=row_sums != 0)
        W_out = W_out.multiply(inv_row_sums[:, np.newaxis])

    return W_out
