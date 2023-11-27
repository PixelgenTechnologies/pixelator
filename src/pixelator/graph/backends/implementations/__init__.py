"""Graph backend implementations used by pixelator.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import os
from typing import Literal, Optional, Union

import networkx as nx
import igraph

from pixelator.graph.backends.protocol import GraphBackend
from pixelator.graph.backends.implementations._igraph import IgraphGraphBackend
from pixelator.graph.backends.implementations._networkx import NetworkXGraphBackend

__all__ = ["IgraphGraphBackend", "NetworkXGraphBackend"]

logger = logging.getLogger(__name__)


def graph_backend(
    graph_backend_class: Optional[
        Literal["IgraphGraphBackend", "NetworkXGraphBackend"]
    ] = None
) -> GraphBackend:
    """Get a graph backend.

    Pick up a GraphBackend. Defaults to `IgraphGraphBackend` unless
    the the following variable is set in the environment:
    `PIXELATOR_GRAPH_BACKEND=True`
    :param graph_backend_class: name of the graph backend class to try to pickup.
    :returns: A concrete graph backend instance
    :rtype: GraphBackend
    :raises ValueError: when `graph_backend_class` is not recognized
    """
    # TODO Later on we could use this as an entry point for loading
    # a graph backend from plugins if we would like.

    def _load_nx():
        logger.debug("Setting up a networkx based backend")
        return NetworkXGraphBackend

    def _load_ig():
        logger.debug("Setting up an igraph based backend")
        return IgraphGraphBackend

    if graph_backend_class:
        if graph_backend_class == "NetworkXGraphBackend":
            return _load_nx()

        if graph_backend_class == "IgraphGraphBackend":
            return _load_ig()

        raise ValueError(
            f"Class name {graph_backend_class} not recognized as `GraphBackend`"
        )

    if str(os.environ.get("PIXELATOR_GRAPH_BACKEND", None)) == "NetworkXGraphBackend":
        return _load_nx()

    return _load_ig()


def graph_backend_from_graph_type(graph: Union[nx.Graph, nx.MultiGraph, igraph.Graph]):
    """Pick the correct backend type based on the graph type."""
    if isinstance(graph, nx.Graph):
        return graph_backend("NetworkXGraphBackend")
    if isinstance(graph, igraph.Graph):
        return graph_backend("IgraphGraphBackend")
    raise ValueError("Cannot recognize type of `graph`")
