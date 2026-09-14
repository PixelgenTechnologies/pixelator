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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from pixelator.common.utils import logger
from pixelator.pna.graph import PNAGraph

_COMPARTMENT_ATTR = "compartment"
_INTERFACE_LABEL = "interface"
_OTHER_LABEL = "other"
_K_LIMITS = (1, 6)
_K_INTERFACE_LIMITS = (0, 4)
_SMOOTHING_LIMITS = (0, 50)
_PSEUDOCOUNT = 1e-8


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
            value of ``cc_protein_weights``.
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
        Segment a conjugate component after fitting protein weights::

            from pixelator.pna.analysis import cc_protein_weights, segment_cell
            from pixelator.pna.pixeldataset import read

            dataset = read("sample.pxl")
            w = cc_protein_weights(
                dataset.adata(),
                group_by="cell_type",
                population_1="B",
                population_2="T",
            )
            component = next(dataset.edgelist().iterator())
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
    counts = graph.node_marker_counts.reindex(node_order).fillna(0)
    adjacency = graph.get_adjacency_sparse(node_ordering=node_order).tocsr()

    classification = _classify_nodes_nnls(
        counts=counts,
        adjacency=adjacency,
        w=w,
        cell_names=cell_names,
        k=k,
        spatial_smoothing_iter=spatial_smoothing_iter,
        random_state=random_state,
        verbose=verbose,
    )

    c1_nodes = _retained_component_nodes(
        graph=graph,
        node_order=node_order,
        keep_mask=classification == 1,
        keep_largest_comp=keep_largest_comp,
        min_comp_size=min_comp_size,
    )
    c2_nodes = _retained_component_nodes(
        graph=graph,
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


def _classify_nodes_nnls(
    *,
    counts: pd.DataFrame,
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

    shared_markers = [m for m in counts.columns if m in w.index]
    if not shared_markers:
        raise ValueError(
            "No proteins in w are present in the graph node marker counts."
        )

    a_k = _expand_adjacency_matrix(adjacency, k)
    a_k.setdiag(1)
    neighborhood_counts = a_k @ counts.loc[:, shared_markers].to_numpy(dtype=float)
    profiles = _l1_normalize_rows(neighborhood_counts)
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
    Matches the internal ``project_weights`` draft
    (``LinearRegression(positive=True, fit_intercept=False)``), which is
    NNLS-like. R ``RcppML::project`` uses ``L1 = 0.2``; that penalty is
    not applied here.
    """
    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(weights, profiles.T)
    return np.asarray(model.coef_, dtype=float)


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


def _retained_component_nodes(
    *,
    graph: PNAGraph,
    node_order: list[Hashable],
    keep_mask: np.ndarray,
    keep_largest_comp: bool,
    min_comp_size: int,
) -> list[Hashable]:
    nodes = [node for node, keep in zip(node_order, keep_mask) if keep]
    if not nodes:
        return []

    components = list(nx.connected_components(graph.raw.subgraph(nodes)))
    if keep_largest_comp:
        order_index = {node: i for i, node in enumerate(node_order)}
        largest = max(
            components,
            key=lambda component: (
                len(component),
                -min(order_index[node] for node in component),
            ),
        )
        return [node for node in node_order if node in largest]

    keep_ids = {
        node
        for component in components
        if len(component) >= min_comp_size
        for node in component
    }
    return [node for node in node_order if node in keep_ids]


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
