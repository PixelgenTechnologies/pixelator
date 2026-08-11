"""Test configuration generally useful in the graph module.


Copyright © 2023 Pixelgen Technologies AB.
"""

import networkx as nx
import pandas as pd
import pytest

from pixelator.common.graph.backends.implementations import graph_backend
from pixelator.common.graph.graph import Graph


@pytest.fixture(name="pentagram_graph")
def pentagram_graph_fixture():
    """Build a graph in the shape of a five pointed star."""
    # Construct a graph in the shape of a five pointed
    # star with a single marker in each point
    edges = [
        (0, 2),
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 4),
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 2),
    ]
    edgelist = pd.DataFrame(
        edges, columns=["upia", "upib"], index=[str(i) for i in range(len(edges))]
    )
    GraphBackend = graph_backend()
    g = Graph(
        backend=GraphBackend.from_edgelist(
            edgelist=edgelist,
            add_marker_counts=False,
            simplify=True,
            use_full_bipartite=True,
        )
    )

    default_marker = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    g.vs.get_vertex(0)["markers"] = dict(default_marker, A=1)
    g.vs.get_vertex(1)["markers"] = dict(default_marker, B=1)
    g.vs.get_vertex(2)["markers"] = dict(default_marker, C=1)
    g.vs.get_vertex(3)["markers"] = dict(default_marker, D=1)
    g.vs.get_vertex(4)["markers"] = dict(default_marker, E=1)

    g.vs.get_vertex(0)["pixel_type"] = "A"
    g.vs.get_vertex(1)["pixel_type"] = "B"
    g.vs.get_vertex(2)["pixel_type"] = "A"
    g.vs.get_vertex(3)["pixel_type"] = "B"
    g.vs.get_vertex(4)["pixel_type"] = "A"

    g.vs.get_vertex(0)["name"] = "AAAA"
    g.vs.get_vertex(1)["name"] = "TTTT"
    g.vs.get_vertex(2)["name"] = "CCCC"
    g.vs.get_vertex(3)["name"] = "GGGG"
    g.vs.get_vertex(4)["name"] = "AATT"
    return g
