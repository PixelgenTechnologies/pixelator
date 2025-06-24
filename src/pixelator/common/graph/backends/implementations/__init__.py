"""Graph backend implementations used by pixelator.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
from typing import Literal, Optional, Union

import networkx as nx

from pixelator.common.graph.backends.implementations._networkx import (
    NetworkXGraphBackend,
)
from pixelator.common.graph.backends.protocol import GraphBackend

__all__ = ["NetworkXGraphBackend"]

logger = logging.getLogger(__name__)


def graph_backend(
    graph_backend_class: Optional[Literal["NetworkXGraphBackend"]] = None,
) -> GraphBackend:
    """Get a graph backend.

    Pick up a GraphBackend. Defaults to `NetworkXGraphBackend`
    (at this time the only available backend).
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

    if graph_backend_class:
        if graph_backend_class == "NetworkXGraphBackend":
            return _load_nx()

        raise ValueError(
            f"Class name {graph_backend_class} not recognized as `GraphBackend`"
        )

    return _load_nx()


def graph_backend_from_graph_type(graph: Union[nx.Graph, nx.MultiGraph]):
    """Pick the correct backend type based on the graph type."""
    if isinstance(graph, nx.Graph):
        return graph_backend("NetworkXGraphBackend")
    raise ValueError("Cannot recognize type of `graph`")
