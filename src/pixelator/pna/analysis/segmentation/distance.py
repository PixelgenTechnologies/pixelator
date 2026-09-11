"""Hop distance from a set of seed nodes on a cell graph.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np

from pixelator.common.utils import logger
from pixelator.pna.graph import PNAGraph

_DISTANCE_ATTR = "distance_from_seed"


def distance_from_node_set(
    graph: PNAGraph,
    seed_nodes: Hashable | Sequence[Hashable],
    max_iter: int = 40,
    verbose: bool = False,
) -> PNAGraph:
    """Compute integer hop distance from seed nodes on a cell graph.

    Runs a multi-source breadth-first search on ``graph``. Each seed has
    distance 0. Every other node gets the length of the shortest unweighted
    path to the nearest seed, up to ``max_iter`` hops. Nodes that are never
    reached keep a missing value.

    The result is stored as the node attribute ``distance_from_seed``,
    replacing that attribute if it already exists. The same ``PNAGraph``
    instance is updated in place and returned.

    Args:
        graph: Component graph to annotate, typically
            ``component.graph`` from a ``PNAPixelDataset`` edgelist
            iterator.
        seed_nodes: One node name or a sequence of node names that must
            all be present in ``graph``.
        max_iter: Maximum hop distance to compute. Nodes farther than this
            (and disconnected nodes) stay missing. Default 40.
        verbose: If True, log how many new nodes are reached at each
            iteration. Default False.

    Returns:
        The same ``PNAGraph``, with integer ``distance_from_seed`` on
        reached nodes and ``None`` on unreached nodes.

    Raises:
        TypeError: If ``graph`` is not a ``PNAGraph``, or if ``max_iter``
            or ``verbose`` have the wrong type.
        ValueError: If ``seed_nodes`` is empty, a seed is missing from
            the graph, or ``max_iter`` is negative.

    Examples:
        Distances from one node on a component graph::

            from pixelator.pna.analysis import distance_from_node_set
            from pixelator.pna.pixeldataset import read

            component = next(read("sample.pxl").edgelist().iterator())
            seed = next(iter(component.graph.raw.nodes))
            distance_from_node_set(component.graph, seed)

    See Also:
        ``distance_from_node_set`` in pixelatorR, the equivalent function
        for R users.

    """
    seeds = _validate_distance_from_node_set_params(
        graph=graph,
        seed_nodes=seed_nodes,
        max_iter=max_iter,
        verbose=verbose,
    )
    max_iter = int(max_iter)

    raw = graph.raw
    distances: dict[Any, int | None] = {node: None for node in raw.nodes}
    for seed in seeds:
        distances[seed] = 0

    frontier = list(dict.fromkeys(seeds))
    for iteration in range(1, max_iter + 1):
        next_frontier: list[Any] = []
        seen_next: set[Any] = set()
        for node in frontier:
            for neighbor in raw.neighbors(node):
                if distances[neighbor] is None and neighbor not in seen_next:
                    distances[neighbor] = iteration
                    next_frontier.append(neighbor)
                    seen_next.add(neighbor)
        if not next_frontier:
            break
        if verbose:
            logger.info(
                "Iteration %s: %s new nodes reached.",
                iteration,
                len(next_frontier),
            )
        frontier = next_frontier

    nx.set_node_attributes(raw, distances, _DISTANCE_ATTR)
    return graph


def _validate_distance_from_node_set_params(
    *,
    graph: PNAGraph,
    seed_nodes: Hashable | Sequence[Hashable],
    max_iter: int,
    verbose: bool,
) -> list[Hashable]:
    if not isinstance(graph, PNAGraph):
        raise TypeError("graph must be a PNAGraph.")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an int.")
    if int(max_iter) < 0:
        raise ValueError("max_iter must be >= 0.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool.")

    seeds = _as_seed_list(seed_nodes)
    missing = [seed for seed in seeds if seed not in graph.raw]
    if missing:
        raise ValueError(
            "All seed nodes must be present in the graph. "
            f"The following seed nodes are not present in the graph: {missing}"
        )
    return seeds


def _as_seed_list(seed_nodes: Hashable | Sequence[Hashable]) -> list[Hashable]:
    if isinstance(seed_nodes, (str, bytes)):
        seeds: list[Hashable] = [seed_nodes]
    elif isinstance(seed_nodes, Sequence):
        seeds = list(seed_nodes)
    else:
        seeds = [seed_nodes]
    if not seeds:
        raise ValueError("seed_nodes must contain at least one node.")
    return seeds
