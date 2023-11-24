"""Graph backend implementations used by pixelator.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import os

from pixelator.graph.backends.implementations._igraph import IgraphGraphBackend
from pixelator.graph.backends.implementations._networkx import NetworkXGraphBackend


logger = logging.getLogger(__name__)

__all__ = ["IgraphGraphBackend", "NetworkXGraphBackend"]


def graph_backend():
    """Get a graph backend depending on which config is active."""
    if os.environ.get("ENABLE_NETWORKX_BACKEND", False):
        logger.debug("Setting up a networkx based backend")
        return NetworkXGraphBackend

    logger.debug("Setting up an igraph based backend")
    return IgraphGraphBackend
