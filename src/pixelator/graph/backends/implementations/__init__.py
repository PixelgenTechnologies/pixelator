"""Graph backend implementations used by pixelator.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import os
from typing import Literal, Optional

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
    `ENABLE_NETWORKX_BACKEND=True`
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

    if not graph_backend_class:
        if str(os.environ.get("ENABLE_NETWORKX_BACKEND", False)) in (
            "True",
            "true",
            "1",
        ):
            return _load_nx()

        return _load_ig()

    if graph_backend_class == "NetworkXGraphBackend":
        return _load_nx()

    if graph_backend_class == "IgraphGraphBackend":
        return _load_ig()

    raise ValueError(
        f"Class name {graph_backend_class} not recognized as `GraphBackend`"
    )
