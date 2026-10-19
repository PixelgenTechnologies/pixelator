"""Segment a cell:cell conjugate graph into two cell types.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

from pixelator.common.utils import logger
from pixelator.pna.graph import PNAGraph

_COMPARTMENT_ATTR = "compartment"
_INTERFACE_LABEL = "interface"
_OTHER_LABEL = "other"
_K_LIMITS = (1, 6)
_K_INTERFACE_LIMITS = (0, 4)
_SMOOTHING_LIMITS = (0, 50)
_PSEUDOCOUNT = 1e-8
# Relative cutoff on det(W.T W). Below this the two weight columns are
# linearly dependent and the 2×2 solve is numerically meaningless.
_NNLS_DET_TOL = 1e-12


def segment_cell(
    graph: PNAGraph,
    w: pd.DataFrame,
    k: int = 2,
    detect_interface: bool = True,
    k_interface_expansion: int = 0,
    keep_largest_comp: bool = True,
    min_comp_size: int = 10,
    spatial_smoothing_iter: int = 5,
    verbose: bool = True,
    random_state: int = 0,
) -> PNAGraph:
    """Classify conjugate-graph nodes into two cell types.

    Uses NMF protein weights (``w``, typically from
    :func:`~pixelator.pna.analysis.cc_protein_weights`) to score each node
    from its k-hop neighborhood, then labels the two cell types and
    optionally the interface between them. Nodes that are not kept in a
    cell-type component are labeled ``"other"``.

    The result is stored as the node attribute ``compartment``, replacing
    that attribute if it already exists. The same :class:`~pixelator.pna.graph.PNAGraph`
    instance is updated in place and returned.

    Args:
        graph: Component graph of a cell:cell conjugate, typically
            ``component.graph`` from a ``PNAPixelDataset`` edgelist
            iterator.
        w: Protein weights with one row per protein and exactly two
            columns named after the two cell types. Usually the return
            value of ``cc_protein_weights`` (each column sums to 1).
        k: Hop distance used to build local neighborhood abundance
            profiles before projecting ``w``. Default 2. Must be between
            1 and 6.
        detect_interface: If True, label nodes on a crossing edge between
            the two retained cell-type components as ``"interface"``.
            Default True.
        k_interface_expansion: Extra hops of neighbors to include in the
            interface. ``0`` (default) keeps only nodes that sit on a
            crossing edge. Must be between 0 and 4. Ignored when
            ``detect_interface`` is False.
        keep_largest_comp: If True (default), keep only the largest
            connected component per cell type. If False, keep every
            component with at least ``min_comp_size`` nodes.
        min_comp_size: Minimum component size when
            ``keep_largest_comp`` is False. Default 10.
        spatial_smoothing_iter: Smoothing iterations on the NNLS scores
            using the graph adjacency. Default 5. ``0`` skips smoothing.
        verbose: If True, log progress. Default True.
        random_state: Seed for the 2-means threshold on population-1
            scores, so labels can be reproduced. Default 0.

    Returns:
        The same ``PNAGraph``, with a ``compartment`` node attribute.
        Values are the two column names of ``w``, ``"interface"``, or
        ``"other"``.

    Raises:
        TypeError: If ``graph`` is not a ``PNAGraph``, ``w`` is not a
            DataFrame, or a flag/integer argument has the wrong type.
        ValueError: If ``w`` does not have two named columns and named
            protein rows, if no proteins are shared with the graph, or if
            a numeric argument is outside its allowed range.

    Examples:
        ``cc_protein_weights`` needs cell type labels in ``adata.obs``
        (here ``"cell_type"``). Assigning those labels is not shown::

            from pixelator.pna.analysis import cc_protein_weights, segment_cell
            from pixelator.pna.pixeldataset import read

            dataset = read("sample.pxl")
            adata = dataset.adata()
            # here cell types need to be annotated in adata.obs["cell_type"] before calling cc_protein_weights
            w = cc_protein_weights(
                adata,
                group_by="cell_type",
                population_1="B",
                population_2="T",
            )
            for component in dataset.edgelist().iterator():
                segment_cell(component.graph, w)

    See Also:
        ``segment_cell`` in pixelatorR, the equivalent function for
        R users.

    """
    cell_names = _validate_segment_cell_params(
        graph=graph,
        w=w,
        k=k,
        detect_interface=detect_interface,
        k_interface_expansion=k_interface_expansion,
        keep_largest_comp=keep_largest_comp,
        min_comp_size=min_comp_size,
        spatial_smoothing_iter=spatial_smoothing_iter,
        verbose=verbose,
        random_state=random_state,
    )
    k = int(k)
    k_interface_expansion = int(k_interface_expansion)
    min_comp_size = int(min_comp_size)
    spatial_smoothing_iter = int(spatial_smoothing_iter)
    random_state = int(random_state)

    node_order = list(graph.raw.nodes())
    adjacency = graph.get_adjacency_sparse(node_ordering=node_order).tocsr()

    classification = _classify_nodes_nnls(
        marker_labels=_node_marker_labels(graph, node_order),
        adjacency=adjacency,
        w=w,
        cell_names=cell_names,
        k=k,
        spatial_smoothing_iter=spatial_smoothing_iter,
        random_state=random_state,
        verbose=verbose,
    )

    c1_nodes = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=classification == 1,
        keep_largest_comp=keep_largest_comp,
        min_comp_size=min_comp_size,
    )
    c2_nodes = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=classification == 2,
        keep_largest_comp=keep_largest_comp,
        min_comp_size=min_comp_size,
    )

    if verbose:
        logger.info("Defining interface nodes between %s", cell_names)

    if detect_interface:
        interface_nodes = _fetch_interface_nodes(
            adjacency=adjacency,
            node_order=node_order,
            c1_nodes=c1_nodes,
            c2_nodes=c2_nodes,
            k_interface_expansion=k_interface_expansion,
        )
    else:
        if verbose:
            logger.info("Skipping interface detection.")
        interface_nodes = []

    if verbose:
        logger.info("Mapping nodes to compartments")

    compartments = _map_compartments(
        node_order=node_order,
        c1_nodes=c1_nodes,
        c2_nodes=c2_nodes,
        interface_nodes=interface_nodes,
        cell_names=cell_names,
    )
    nx.set_node_attributes(graph.raw, compartments, _COMPARTMENT_ATTR)
    return graph


def _validate_segment_cell_params(
    *,
    graph: PNAGraph,
    w: pd.DataFrame,
    k: int,
    detect_interface: bool,
    k_interface_expansion: int,
    keep_largest_comp: bool,
    min_comp_size: int,
    spatial_smoothing_iter: int,
    verbose: bool,
    random_state: int,
) -> list[str]:
    if not isinstance(graph, PNAGraph):
        raise TypeError("graph must be a PNAGraph.")
    if not isinstance(w, pd.DataFrame):
        raise TypeError("w must be a pandas DataFrame.")
    if w.shape[1] != 2:
        raise ValueError(
            "w must have exactly two columns corresponding to the two "
            "interacting cell types."
        )
    if w.empty:
        raise ValueError("w must have at least one protein row.")
    if any(name is None or str(name) == "" for name in w.columns):
        raise ValueError("w must have two character column names.")
    if w.index.hasnans or any(str(name) == "" for name in w.index):
        raise ValueError("w must have protein names as its index.")
    if w.columns[0] == w.columns[1]:
        raise ValueError("w column names must be distinct cell type names.")

    _require_bool(detect_interface, "detect_interface")
    _require_bool(keep_largest_comp, "keep_largest_comp")
    _require_bool(verbose, "verbose")
    _require_int_in_range(k, "k", *_K_LIMITS)
    _require_int_in_range(
        k_interface_expansion, "k_interface_expansion", *_K_INTERFACE_LIMITS
    )
    _require_int_in_range(min_comp_size, "min_comp_size", 1, None)
    _require_int_in_range(
        spatial_smoothing_iter, "spatial_smoothing_iter", *_SMOOTHING_LIMITS
    )
    if not isinstance(random_state, (int, np.integer)) or isinstance(
        random_state, bool
    ):
        raise TypeError("random_state must be an int.")

    return [str(name) for name in w.columns]


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _require_int_in_range(
    value: Any, name: str, minimum: int, maximum: int | None
) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")


def _node_marker_labels(graph: PNAGraph, node_order: Sequence[Hashable]) -> list[Any]:
    """Marker name of each node, or ``None`` when the node has no marker.

    PNA nodes carry a single protein. Callers use these names instead of a
    dense nodes × proteins count table.
    """
    markers = dict(graph.raw.nodes(data="marker", default=None))
    return [markers.get(node) for node in node_order]


def _shared_markers(marker_labels: Sequence[Any], weight_index: pd.Index) -> list[Any]:
    """Markers present on the graph and in ``w``, in pivot-table order."""
    weight_markers = set(weight_index)
    present = {marker for marker in marker_labels if marker in weight_markers}
    return sorted(present)


def _marker_indicator(
    marker_labels: Sequence[Any], shared_markers: Sequence[Any]
) -> sparse.csr_matrix:
    """Sparse one-hot of ``shared_markers`` with one entry per labeled node."""
    column_of = {marker: i for i, marker in enumerate(shared_markers)}
    columns = np.fromiter(
        (column_of.get(marker, -1) for marker in marker_labels),
        dtype=np.int32,
        count=len(marker_labels),
    )
    rows = np.flatnonzero(columns >= 0)
    return sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns[rows])),
        shape=(len(marker_labels), len(shared_markers)),
    )


def _classify_nodes_nnls(
    *,
    marker_labels: Sequence[Any],
    adjacency: sparse.csr_matrix,
    w: pd.DataFrame,
    cell_names: Sequence[str],
    k: int,
    spatial_smoothing_iter: int,
    random_state: int,
    verbose: bool,
) -> np.ndarray:
    if verbose:
        logger.info("Classifying %s nodes using NNLS", list(cell_names))

    shared_markers = _shared_markers(marker_labels, w.index)
    if not shared_markers:
        raise ValueError(
            "No proteins in w are present in the graph node marker counts."
        )

    a_k = _expand_adjacency_matrix(adjacency, k)
    a_k.setdiag(1)
    # Each node contributes its one marker. The dense result is only the
    # proteins in w, after the neighborhood sum.
    neighborhood_counts = a_k @ _marker_indicator(marker_labels, shared_markers)
    if sparse.issparse(neighborhood_counts):
        neighborhood_counts = neighborhood_counts.toarray()
    profiles = _l1_normalize_rows(np.asarray(neighborhood_counts, dtype=float))
    weights = w.loc[shared_markers].to_numpy(dtype=float)
    scores = _project_weights(profiles, weights)

    if spatial_smoothing_iter > 0:
        if verbose:
            logger.info(
                "Performing spatial smoothing of projection scores with %s iterations",
                spatial_smoothing_iter,
            )
        scores = _spatial_smoothing(adjacency, scores, n_iter=spatial_smoothing_iter)

    scores = scores + _PSEUDOCOUNT
    scores = scores / scores.sum(axis=1, keepdims=True)
    threshold = _kmeans_midpoint(scores[:, 0], random_state=random_state)
    return np.where(scores[:, 0] >= threshold, 1, 2)


def _expand_adjacency_matrix(adjacency: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Boolean k-hop expansion matching R ``expand_adjacency_matrix``.

    Self-loops are added for the matrix power when ``k > 1``, then removed
    from the result. ``k == 1`` returns a copy of ``adjacency`` unchanged.
    """
    if k == 1:
        return adjacency.copy()

    expanded = adjacency.tocsr(copy=True)
    expanded.setdiag(1)
    expanded.eliminate_zeros()
    _binarize_inplace(expanded)
    result = expanded
    for _ in range(k - 1):
        result = result @ expanded
        _binarize_inplace(result)
    result.setdiag(0)
    result.eliminate_zeros()
    return result


def _binarize_inplace(matrix: sparse.spmatrix) -> None:
    if matrix.nnz:
        matrix.data = np.ones(matrix.nnz, dtype=np.float64)


def _l1_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _project_weights(profiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Non-negative projection of ``weights`` onto neighborhood profiles.

    ``profiles`` is nodes × proteins and ``weights`` is proteins × 2.
    Each row ``p`` solves ``min ||W h - p||^2`` with ``h >= 0``, the same
    unpenalized problem previously solved with
    ``LinearRegression(positive=True, fit_intercept=False)``. The two
    coefficients are solved together from the normal equations. R
    ``RcppML::project`` uses ``L1 = 0.2``; that penalty is not applied here.
    """
    weight_inner_products = weights.T @ weights
    profile_weight_products = profiles @ weights
    return _nonnegative_least_squares_two(
        weight_inner_products, profile_weight_products
    )


def _nonnegative_least_squares_two(
    weight_inner_products: np.ndarray,
    profile_weight_products: np.ndarray,
) -> np.ndarray:
    """Lawson-Hanson NNLS for a two-column design.

    ``weight_inner_products`` is ``W.T @ W``, the dot products of the two
    weight columns. Each row of ``profile_weight_products`` is ``W.T @ p``,
    the dot product of one neighborhood profile with each weight column.
    At ``h = 0`` the loss decreases fastest along the column with the
    larger positive dot product, so that column is freed first. The other
    column is added only when its reduced gradient is positive and the
    two columns are linearly independent. A negative coefficient in that
    2×2 solution is set back to zero and the remaining column is re-solved.

    With only column ``i`` free, ``h_i = b_i / G_ii`` when that value is
    positive. With both columns free, ``h`` solves the normal equations
    ``G h = b``, where ``G`` is ``weight_inner_products`` and ``b`` is one
    row of ``profile_weight_products``:

    ``h_0 = (G_11 b_0 - G_01 b_1) / det(G)``

    ``h_1 = (G_00 b_1 - G_01 b_0) / det(G)``

    References:
        Lawson, C. L. and Hanson, R. J. *Solving Least Squares Problems*,
        Chapter 23, Algorithm NNLS. SIAM, 1995.
        doi:10.1137/1.9781611971217

        ``scipy.optimize.nnls`` implements that algorithm. The formulas
        above are the same active set for a two-column design.
    """
    n_rows = profile_weight_products.shape[0]
    coef = np.zeros((n_rows, 2), dtype=float)
    if n_rows == 0:
        return coef

    g00 = float(weight_inner_products[0, 0])
    g11 = float(weight_inner_products[1, 1])
    g01 = float(weight_inner_products[0, 1])
    column0_product = profile_weight_products[:, 0]
    column1_product = profile_weight_products[:, 1]
    h0 = column0_product / g00 if g00 > 0.0 else np.zeros(n_rows, dtype=float)
    h1 = column1_product / g11 if g11 > 0.0 else np.zeros(n_rows, dtype=float)

    # At h = 0 the gradient of the loss is -profile_weight_products, so
    # the larger positive dot product is the first variable freed.
    enter0 = (column0_product >= column1_product) & (column0_product > 0.0)
    enter1 = (column1_product > column0_product) & (column1_product > 0.0)
    reduced1 = column1_product - g01 * h0
    reduced0 = column0_product - g01 * h1
    det = g00 * g11 - g01 * g01
    inner_product_scale = max(g00, g11, abs(g01), 1.0)
    singular = abs(det) <= _NNLS_DET_TOL * inner_product_scale * inner_product_scale

    stay0 = enter0 & (singular | (reduced1 <= 0.0))
    stay1 = enter1 & (singular | (reduced0 <= 0.0))
    coef[stay0, 0] = h0[stay0]
    coef[stay1, 1] = h1[stay1]
    if singular:
        return coef

    solve = (enter0 & (reduced1 > 0.0)) | (enter1 & (reduced0 > 0.0))
    if not np.any(solve):
        return coef

    # Unconstrained solution of G h = b. A negative entry is not
    # feasible: drop that variable and keep the one-column coefficient.
    u0 = (g11 * column0_product - g01 * column1_product) / det
    u1 = (g00 * column1_product - g01 * column0_product) / det
    both = solve & (u0 >= 0.0) & (u1 >= 0.0)
    coef[both, 0] = u0[both]
    coef[both, 1] = u1[both]

    drop0 = solve & (u0 < 0.0)
    coef[drop0, 1] = np.maximum(h1[drop0], 0.0)
    drop1 = solve & (u1 < 0.0) & (u0 >= 0.0)
    coef[drop1, 0] = np.maximum(h0[drop1], 0.0)
    return coef


def _spatial_smoothing(
    adjacency: sparse.csr_matrix, scores: np.ndarray, n_iter: int
) -> np.ndarray:
    """Smooth node scores with a row-normalized adjacency, matching R.

    Neighbors get equal weight; the center node is then given the same
    weight as the sum of its neighbors and rows are renormalized.
    Isolated nodes keep their own score (row-sum 0 is treated as 1).
    """
    transition = adjacency.tocsr(copy=True).astype(float)
    row_sums = np.asarray(transition.sum(axis=1), dtype=float).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_row = 1.0 / row_sums
    transition = transition.multiply(inv_row[:, np.newaxis]).tocsr()
    transition.setdiag(1.0)
    row_sums = np.asarray(transition.sum(axis=1), dtype=float).ravel()
    row_sums[row_sums == 0] = 1.0
    transition = transition.multiply((1.0 / row_sums)[:, np.newaxis]).tocsr()

    smoothed = np.asarray(scores, dtype=float)
    for _ in range(n_iter):
        smoothed = transition @ smoothed
    return smoothed


def _kmeans_midpoint(values: np.ndarray, random_state: int) -> float:
    """Return the midpoint of two k-means centers on a 1-d score vector."""
    if values.size < 2:
        raise ValueError("Need at least two nodes to threshold cell type scores.")
    model = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    model.fit(np.asarray(values, dtype=float).reshape(-1, 1))
    return float(np.mean(model.cluster_centers_))


def _edges_within_mask(
    adjacency: sparse.csr_matrix, keep_mask: np.ndarray
) -> sparse.csr_matrix:
    """Drop edges that touch a node outside ``keep_mask``.

    The result is still indexed like ``adjacency``. Nodes outside the mask
    become isolated and are ignored by the caller.
    """
    restricted = adjacency.tocsr(copy=True)
    if restricted.nnz == 0:
        return restricted
    rows = np.repeat(np.arange(restricted.shape[0]), np.diff(restricted.indptr))
    inside = keep_mask[rows] & keep_mask[restricted.indices]
    restricted.data = np.where(inside, restricted.data, 0)
    restricted.eliminate_zeros()
    return restricted


def _retained_component_nodes(
    *,
    adjacency: sparse.csr_matrix,
    node_order: list[Hashable],
    keep_mask: np.ndarray,
    keep_largest_comp: bool,
    min_comp_size: int,
) -> list[Hashable]:
    """Nodes kept after filtering connected components of ``keep_mask``.

    Uses the CSR adjacency already built for scoring. Equal-sized components
    are broken by the earliest node in ``node_order``.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    member_idx = np.flatnonzero(keep_mask)
    if member_idx.size == 0:
        return []

    if member_idx.size == adjacency.shape[0]:
        component_graph = adjacency
    else:
        component_graph = _edges_within_mask(adjacency, keep_mask)
    labels = connected_components(component_graph, directed=False, return_labels=True)[
        1
    ][member_idx]

    if keep_largest_comp:
        sizes = np.bincount(labels)
        earliest = np.full(sizes.shape[0], np.iinfo(np.intp).max, dtype=np.intp)
        np.minimum.at(earliest, labels, member_idx)
        candidates = np.flatnonzero(sizes == sizes.max())
        # Same size: keep the component whose earliest node comes first.
        chosen = candidates[np.argmin(earliest[candidates])]
        selected = member_idx[labels == chosen]
    else:
        sizes = np.bincount(labels)
        selected = member_idx[sizes[labels] >= min_comp_size]
    return [node_order[i] for i in selected]


def _fetch_interface_nodes(
    *,
    adjacency: sparse.csr_matrix,
    node_order: list[Hashable],
    c1_nodes: Sequence[Hashable],
    c2_nodes: Sequence[Hashable],
    k_interface_expansion: int,
) -> list[Hashable]:
    if not c1_nodes or not c2_nodes:
        return []

    index = {node: i for i, node in enumerate(node_order)}
    c1_idx = np.fromiter((index[node] for node in c1_nodes), dtype=int)
    c2_idx = np.fromiter((index[node] for node in c2_nodes), dtype=int)
    crossing = adjacency[c1_idx][:, c2_idx]
    c1_touch = np.asarray(crossing.sum(axis=1)).ravel() > 0
    c2_touch = np.asarray(crossing.sum(axis=0)).ravel() > 0
    interface = [node for node, touch in zip(c1_nodes, c1_touch) if touch]
    interface.extend(node for node, touch in zip(c2_nodes, c2_touch) if touch)

    if k_interface_expansion > 0 and interface:
        expanded = _expand_adjacency_matrix(adjacency, k_interface_expansion)
        iface_idx = np.fromiter((index[node] for node in interface), dtype=int)
        reached = np.asarray(expanded[iface_idx, :].sum(axis=0)).ravel() > 0
        extra = [node for node, hit in zip(node_order, reached) if hit]
        interface = list(dict.fromkeys(interface + extra))
    return interface


def _map_compartments(
    *,
    node_order: list[Hashable],
    c1_nodes: Sequence[Hashable],
    c2_nodes: Sequence[Hashable],
    interface_nodes: Sequence[Hashable],
    cell_names: Sequence[str],
) -> dict[Hashable, str]:
    interface = set(interface_nodes)
    cell1 = set(c1_nodes)
    cell2 = set(c2_nodes)
    compartments: dict[Hashable, str] = {}
    for node in node_order:
        if node in interface:
            compartments[node] = _INTERFACE_LABEL
        elif node in cell1:
            compartments[node] = cell_names[0]
        elif node in cell2:
            compartments[node] = cell_names[1]
        else:
            compartments[node] = _OTHER_LABEL
    return compartments
