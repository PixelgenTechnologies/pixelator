"""Functions related to the graph dataclass used in pixelator graph operations.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
import os
from typing import Dict, Iterable, List, Tuple, Union

import igraph
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix

from pixelator.graph.backends.implementations import (
    IgraphGraphBackend,
    NetworkXGraphBackend,
)
from pixelator.graph.backends.protocol import _GraphBackend

logger = logging.getLogger(__name__)


class Graph:
    """`Graph` represents a graph, i.e. a collection of vertices and edges."""

    def __init__(self, backend: _GraphBackend):
        """Create a new Graph instance.

        Create a Graph instance (as an end-user this is probably not the interface
        you are looking for). Try `Graph.from_edgelist`.

        :param backend: The backend used to represent the graph
        """
        self._backend = backend

    @staticmethod
    def from_edgelist(
        edgelist: pd.DataFrame,
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
    ) -> "Graph":
        """Build a graph from an edgelist.

        Build a Graph from an edge list (pd.DataFrame). Multiple options are available
        to build the graph, `add_marker_counts` will add a dictionary of marker counts
        to each node, `simplify` will remove redundant edges and `use_full_bipartite`
        will not project the graph (UPIA).

        The graph will contain the edge attributes present in the edge list when
        `use_full_bipartite` is True and a dictionary of marker counts in each
        vertex (node) when `add_marker_counts` is True. If `use_full_bipartite` is
        False or `simplify` is True the edge attributes will be lost.

        :param edgelist: the edge list (dataframe) corresponding to the graph
        :param add_marker_counts: add a dictionary of marker counts to each node
        :param simplify: simplifies the graph (remove redundant edges)
        :param use_full_bipartite: use the bipartite graph instead of the projection
                                  (UPIA)
        :returns: a Graph instance
        :rtype: Graph
        :raises: AssertionError when the input edge list is not valid
        """
        if os.environ.get("ENABLE_NETWORKX_BACKEND", False):
            return Graph(
                backend=NetworkXGraphBackend.from_edgelist(
                    edgelist=edgelist,
                    add_marker_counts=add_marker_counts,
                    simplify=simplify,
                    use_full_bipartite=use_full_bipartite,
                )
            )

        return Graph(
            backend=IgraphGraphBackend.from_edgelist(
                edgelist=edgelist,
                add_marker_counts=add_marker_counts,
                simplify=simplify,
                use_full_bipartite=use_full_bipartite,
            )
        )

    @staticmethod
    def from_raw(graph: Union[igraph.Graph, nx.Graph]) -> "Graph":
        """Generate a Graph from an igraph.Graph object.

        :param graph: input igraph to use
        :return: A pixelator Graph object
        :rtype: Graph
        """
        if os.environ.get("ENABLE_NETWORKX_BACKEND", False):
            return Graph(backend=NetworkXGraphBackend(graph))
        return Graph(backend=IgraphGraphBackend(graph))

    @property
    def _raw(self):
        return self._backend._raw

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        return self._backend.vs

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        return self._backend.es

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        return self._backend.vcount()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        return self._backend.ecount()

    def get_adjacency_sparse(self) -> csr_matrix:
        """Get the sparse adjacency matrix."""
        return self._backend.get_adjacency_sparse()

    def connected_components(self):
        """Get the connected components in the Graph instance."""
        return self._backend.connected_components()

    def community_leiden(self, **kwargs):
        """Run community detection using the Leiden algorithm."""
        return self._backend.community_leiden(**kwargs)

    def layout_coordinates(
        self,
        layout_algorithm: str = "fruchterman_reingold",
        only_keep_a_pixels: bool = True,
        get_node_marker_matrix: bool = True,
    ) -> pd.DataFrame:
        """Generate coordinates and (optionally) node marker counts for plotting.

        Generate a dataframe with coordinates, and (optionally) node marker
        counts to use that can be used for plotting.

        The layout options are:
          - fruchterman_reingold
          - fruchterman_reingold_3d
          - kamada_kawai
          - kamada_kawai_3d


        The `fruchterman_reingold` options are in general faster, but less
        accurate than the `kamada_kawai` ones.

        :param layout_algorithm: the layout algorithm to use to generate the coordinates
        :param only_keep_a_pixels: If true, only keep the a-pixels
        :param get_node_marker_matrix: Add a matrix of marker counts to each
                                       node if True.
        :return: the coordinates and markers (if activated) as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if the provided `layout_algorithm` is not valid
        :raises: ValueError if the provided current graph instance is empty
        """
        return self._backend.layout_coordinates(
            layout_algorithm=layout_algorithm,
            only_keep_a_pixels=only_keep_a_pixels,
            get_node_marker_matrix=get_node_marker_matrix,
        )

    def get_edge_dataframe(self):
        """Get the edges as a pandas DataFrame."""
        return self._backend.get_edge_dataframe()

    def get_vertex_dataframe(self):
        """Get all vertices as a pandas DataFrame."""
        return self._backend.get_vertex_dataframe()

    def add_edges(self, edges: Iterable[Tuple[int]]) -> None:
        """Add edges to the graph instance.

        :param edges: Add the following edges to the graph instance.
        """
        if not self._backend.raw:
            self._backend.from_raw(igraph.Graph())
        self._backend.add_edges(edges)

    def add_vertices(self, n_vertices: int, attrs: Dict[str, List]) -> None:
        """Add some number of vertices to the graph instance.

        :param n_vertices: the number of vertices to be added to the graph instance.
        :param attrs: dict of sequences, all of length equal to the number of vertices
                      to be added, containing the attributes of the new vertices. If
                      `n_vertices=1` then they have to be lists of length 1.
        :raises IndexError: if the number of graph vertices to add and lists of
                            attributes are of different lengths
        """
        self._backend.add_vertices(n_vertices=n_vertices, attrs=attrs)

    def add_names_to_vertexes(self, vs_names: List[str]) -> None:
        """Rename the current vertices on the graph instance.

        :param vs_names: Add the following vertices to the graph instance.
        :raises ValueError: if the graph is empty
        :raises IndexError: if the number of graph vertices and list of names are
                            of different length
        """
        self._backend.add_names_to_vertexes(vs_names=vs_names)
