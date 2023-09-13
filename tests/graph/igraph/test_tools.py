"""
Functionality from igraph to produce pixelator Graph
of certain types required for testing

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import random
from typing import Optional

import igraph
import numpy as np
from pixelator.graph.utils import Graph

from tests.graph.test_graph_utils import add_random_names_to_vertexes


def create_random_graph(n_nodes: int, prob: float) -> Graph:
    """
    create a random graph with n_nodes nodes
    and a probability prob of connecting two
    nodes with an edge
    """
    rng = np.random.default_rng(2)
    edge_list = []
    for i in range(0, n_nodes):
        for j in range(i, n_nodes):
            if rng.uniform(0, 1) < prob:
                edge_list.append((i, j))
    g = igraph.Graph(directed=False)
    g.add_vertices(n_nodes)
    g.add_edges(edge_list)
    add_random_names_to_vertexes(g)
    return Graph.from_raw(g)


def create_fully_connected_bipartite_graph(n_nodes: int) -> Graph:
    """
    create a fully connected bipartite graph of
    size n_nodes * 2
    """
    graph = igraph.Graph.Full_Bipartite(n1=n_nodes, n2=n_nodes, directed=False)
    add_random_names_to_vertexes(graph)
    return Graph.from_raw(graph)


def create_randomly_connected_bipartite_graph(
    n1: int, n2: int, p: float, random_seed: Optional[int]
) -> Graph:
    if random_seed:
        random.seed(random_seed)
    graph = igraph.Graph.Random_Bipartite(n1, n2, p)
    add_random_names_to_vertexes(graph)
    return Graph.from_raw(graph)


def full_graph(n: int) -> Graph:
    return Graph.from_raw(igraph.Graph.Full(n=n))
