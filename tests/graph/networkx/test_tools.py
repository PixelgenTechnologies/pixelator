"""
Functionality from networkx to produce pixelator Graph
of certain types required for testing

Copyright © 2023 Pixelgen Technologies AB.
"""

import random
from typing import Optional

import networkx as nx
import numpy as np

from pixelator.graph.utils import Graph


def create_random_graph(n_nodes: int, prob: float) -> Graph:
    """
    create a random graph with n_nodes nodes
    and a probability prob of connecting two
    nodes with an edge
    """
    random.seed(0)
    rng = np.random.default_rng(2)
    edge_list = []
    for i in range(0, n_nodes):
        for j in range(i, n_nodes):
            if rng.uniform(0, 1) < prob:
                edge_list.append((i, j))
    g = nx.Graph()
    g.add_nodes_from(range(0, n_nodes))
    g.add_edges_from(edge_list)
    g = Graph.from_raw(g)
    add_random_names_to_vertexes(g)
    return g


def create_fully_connected_bipartite_graph(n_nodes: int) -> Graph:
    """
    create a fully connected bipartite graph of
    size n_nodes * 2
    """
    graph = Graph.from_raw(nx.complete_bipartite_graph(n1=n_nodes, n2=n_nodes))
    add_random_names_to_vertexes(graph)
    return graph


def create_randomly_connected_bipartite_graph(
    n1: int, n2: int, p: float, random_seed: Optional[int]
) -> Graph:
    graph = Graph.from_raw(nx.bipartite.random_graph(n1, n2, p, seed=random_seed))
    add_random_names_to_vertexes(graph)
    return graph


def full_graph(n: int) -> Graph:
    return Graph.from_raw(nx.complete_graph(n))


def add_random_names_to_vertexes(graph: Graph) -> None:
    """Add some random names to vertices on the graph."""
    for vertex in graph.vs:
        vertex["name"] = random_sequence(21)


def random_sequence(size: int) -> str:
    """Create a random sequence of size (size)."""
    return "".join(random.choices("CGTA", k=size))
