"""Protocol of graph backends used by pixelator.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Protocol, Tuple

import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix


class _GraphBackend(Protocol):
    """Protocol for graph backends."""

    @staticmethod
    def from_edgelist(
        edgelist: pd.DataFrame,
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
    ) -> _GraphBackend:
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
        :rtype: _GraphBackend
        :raises: AssertionError when the input edge list is not valid
        """
        ...

    @staticmethod
    def from_raw(graph: nx.Graph) -> _GraphBackend:
        """Generate a Graph from an networkx.Graph object.

        :param graph: input igraph to use
        :return: A pixelator Graph object
        :rtype: _GraphBackend
        """
        ...

    @property
    def raw(self):
        """Get the raw underlying graph representation."""
        ...

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        ...

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        ...

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        ...

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        ...

    def get_adjacency_sparse(self) -> csr_matrix:
        """Get the sparse adjacency matrix."""
        ...

    def connected_components(self):
        """Get the connected components in the Graph instance."""
        ...

    def community_leiden(self, **kwargs):
        """Run community detection using the Leiden algorithm."""
        ...

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
        ...

    def get_edge_dataframe(self):
        """Get the edges as a pandas DataFrame."""
        ...

    def get_vertex_dataframe(self):
        """Get all vertices as a pandas DataFrame."""
        ...

    def add_edges(self, edges: Iterable[Tuple[int]]) -> None:
        """Add edges to the graph instance.

        :param edges: Add the following edges to the graph instance.
        """
        ...

    def add_vertices(self, n_vertices: int, attrs: Dict[str, List]) -> None:
        """Add some number of vertices to the graph instance.

        :param n_vertices: the number of vertices to be added to the graph instance.
        :param attrs: dict of sequences, all of length equal to the number of vertices
                      to be added, containing the attributes of the new vertices. If
                      `n_vertices=1` then they have to be lists of length 1.
        :raises IndexError: if the number of graph vertices to add and lists of
                            attributes are of different lengths
        """
        ...

    def add_names_to_vertexes(self, vs_names: List[str]) -> None:
        """Rename the current vertices on the graph instance.

        :param vs_names: Add the following vertices to the graph instance.
        :raises ValueError: if the graph is empty
        :raises IndexError: if the number of graph vertices and list of names are
                            of different length
        """
        ...
