"""Module with helpers for calculating the adjacency matrix of a graph.

Given a list of close neigbours from a search on a FAISS index.

Copyright © 2024 Pixelgen Technologies AB.
"""

import numpy as np
import numpy.typing as npt
import scipy


def build_network_directional(
    distances: npt.NDArray[np.int64],
    indices: npt.NDArray[np.uint64],
    read_counts: npt.NDArray[np.int64],
    cutoff: int,
) -> scipy.sparse.csr_matrix:
    """Create a sparse directed adjacency matrix from a FAISS k-nearest query result.

    Uses the "Directional" approach as described by UMI-tools.

    :param distances: The distances of the k-nearest neighbors.
        A 2D array with shape (n_queries, k).
    :param indices: The indices of the k-nearest neighbors.
        A 2D array with shape (n_queries, k).
    :param read_counts: The read support for each molecule.
        A 1D array with shape (n_queries,).
    :param cutoff: The distance cutoff to consider an edge between two nodes.
    :return: A sparse directed adjacency matrix in CSR format.
    :rtype: scipy.sparse.csr_matrix
    """
    if distances.shape != indices.shape:
        raise ValueError(
            f"The distances and indices matrix must have the same shape."
            f"D({distances.shape}) != I({indices.shape})"
        )

    n_queries = distances.shape[0]

    # Hamming distance must be less than or equal to the cutoff
    # Each nucleotide difference will result in a hamming distance of 2
    # The cutoff is already in hamming space so the caller must have adjusted for this
    edges = distances <= cutoff

    # Retrieve the non-zero indices for the candidate edges
    i, j = np.nonzero(edges)  # type: ignore

    # row indices so we can reuse those directly as the node_a index
    node_a_index = i
    # retrieve the column (node_b) indices from the Indices matrix from the FAISS query result
    node_b_index = indices[i, j]

    # Retrieve read counts for the edge candidates
    node_a_read_counts = read_counts[node_a_index]
    node_b_read_counts = read_counts[node_b_index]

    # Only keep edges where the read counts are at least twice as high for node A
    nodes_mask = node_a_read_counts >= (2 * node_b_read_counts - 1)

    row = node_a_index[nodes_mask]
    col = node_b_index[nodes_mask]
    data = np.ones(len(row), dtype=np.uint8)

    # Build a CSR matrix from the adjacency information in COO format
    directional_adjacency = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(n_queries, n_queries), dtype=np.uint8
    )
    # Convert to CSR format for efficient connected components calculation later
    directional_adjacency = directional_adjacency.tocsr()
    return directional_adjacency


# Note: read_counts is not used in this function, but we keep it here to
# make the signature the same as build_network_directional


def build_network_cluster(
    distances: npt.NDArray[np.int64],
    indices: npt.NDArray[np.uint64],
    read_counts: npt.NDArray[np.int64],
    cutoff: int,
) -> scipy.sparse.csr_matrix:
    """Create a sparse adjacency matrix from a FAISS k-nearest query result.

    Use the "Cluster" approach as described by UMI-tools.

    :param distances: The distances of the k-nearest neighbors.
        A 2D array with shape (n_queries, k).
    :param indices: The indices of the k-nearest neighbors.
        A 2D array with shape (n_queries, k).
    :param read_counts: The read support for each molecule.
        A 1D array with shape (n_queries,).
    :param cutoff: The distance cutoff to consider an edge between two nodes.
    """
    n_queries = distances.shape[0]

    offsets = np.zeros((n_queries + 1), dtype=np.uint64)
    edges = distances <= cutoff
    edge_weights = np.ones_like(distances[edges])
    edge_indices = indices[edges]

    # Build the offsets array to directly construct a CSR matrix
    np.cumsum(np.count_nonzero(edges, axis=1), out=offsets[1:])

    adjacency = scipy.sparse.csr_matrix(
        (edge_weights, edge_indices, offsets), shape=(n_queries, n_queries)
    )

    return adjacency


def build_network_cluster_from_range_query(lims, distances, indices, index_map=None):
    """Create a sparse adjacency matrix from a FAISS range query result.

    FAISS range queries already returns a CSR matrix

    So for query i:
     - lims[i], lims[i+1] is the range in I and D that lists the neighbors of i
     - I[lims[i]:lims[i+1]] are the indices from the database vectors
     - D[lims[i]:lims[i+1]] are the corresponding distances


    :param lims: The range limits for the neighbors of each query.
    :param distances: The distances of the neighbors.
    :param indices: The indices of the neighbors.
    :param index_map: A transformation of the indices
    """
    indices = indices

    if index_map:
        indices = index_map[indices]

    # The FAISS result is actually identical to the CSR sparse matrix format
    # We can just load it into a scipy sparse matrix without copying where the data is stored

    adjacency = scipy.sparse.csr_matrix(
        (distances, indices, lims), shape=(len(indices), len(indices))
    )
    return adjacency
