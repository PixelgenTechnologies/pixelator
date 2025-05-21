"""Functions related to the graph dataclass used in pixelator graph operations.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from functools import cached_property, lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd
import polars as pl
from scipy.sparse import csr_array, csr_matrix

from pixelator.common.graph.backends.implementations import (
    graph_backend,
    graph_backend_from_graph_type,
)
from pixelator.common.graph.backends.protocol import (
    GraphBackend,
    SupportedLayoutAlgorithm,
    VertexClustering,
)
from pixelator.common.graph.node_metrics import local_g

logger = logging.getLogger(__name__)


class Graph:
    """`Graph` represents a graph, i.e. a collection of vertices and edges."""

    def __init__(self, backend: GraphBackend):
        """Create a new Graph instance.

        Create a Graph instance (as an end-user this is probably not the interface
        you are looking for). Try `Graph.from_edgelist`.

        :param backend: The backend used to represent the graph
        """
        self._backend = backend
        self._connected_components_needs_recompute = False
        self._connected_components: Optional[VertexClustering] = None

    @staticmethod
    def from_edgelist(
        edgelist: Union[pd.DataFrame, pl.LazyFrame],
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
        convert_indices_to_integers: bool = True,
    ) -> Graph:
        """Build a graph from an edgelist.

        Build a Graph from an edge list (pd.DataFrame). Multiple options are available
        to build the graph, `add_marker_counts` will add a dictionary of marker counts
        to each node, `simplify` will remove redundant edges and `use_full_bipartite`
        will not project the graph (UPIA).

        The graph will contain the edge attributes present in the edge list when
        `use_full_bipartite` is True and a dictionary of marker counts in each
        vertex (node) when `add_marker_counts` is True. If `use_full_bipartite` is
        False or `simplify` is True the edge attributes will be lost.

        :param edgelist: the edge list (dataframe) corresponding to the graph either as
                         a pandas data frame or as a polars LazyFrame. To minimize the
                         memory usage the LazyFrame is preferred.
        :param add_marker_counts: add a dictionary of marker counts to each node
        :param simplify: simplifies the graph (remove redundant edges)
        :param use_full_bipartite: use the bipartite graph instead of the projection
                                  (UPIA)
        :param convert_indices_to_integers: convert the indices to integers (this is the default)
        :returns: a Graph instance
        :rtype: Graph
        :raises: AssertionError when the input edge list is not valid
        """
        backend = graph_backend().from_edgelist(
            edgelist=edgelist,
            add_marker_counts=add_marker_counts,
            simplify=simplify,
            use_full_bipartite=use_full_bipartite,
            convert_indices_to_integers=convert_indices_to_integers,
        )

        return Graph(backend=backend)

    @staticmethod
    def from_raw(graph: Any) -> Graph:
        """Generate a Graph from a graph object.

        Which graph object is valid depends on the underlying
        graph implementation. In general end users should use
        the `from_edgelist` method instead.

        :param graph: input graph to use
        :return: A pixelator Graph object
        :rtype: Graph
        :raises ValueError: if type of `graph` is not recognized.
        """
        Backend = graph_backend_from_graph_type(graph)
        return Graph(backend=Backend(graph))

    @property
    def raw(self):
        """Get the underlying raw graph implementation.

        The type of this will depend on the underlying backend.
        This property is useful when you need to inter-operate
        with other graph libraries that work e.g. with a
        networkx graph instance.
        """
        return self._backend.raw

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        return self._backend.vs

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        return self._backend.es

    @cached_property
    def node_marker_counts(self):
        """Get the node marker counts as a dataframe."""
        return self._backend.node_marker_counts()

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        return self._backend.vcount()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        return self._backend.ecount()

    def get_adjacency_sparse(
        self, node_ordering: Iterable[Any] | None = None
    ) -> csr_matrix:
        """Get the sparse adjacency matrix.

        :param node_ordering: Control the node ordering in the adjacency matrix
        :return: a sparse adjacency matrix
        """
        return self._backend.get_adjacency_sparse(node_ordering=node_ordering)

    def connected_components(self) -> VertexClustering:
        """Get the connected components in the Graph instance."""
        # Recompute if there has been changes, otherwise pick the
        # cached value
        if not self._connected_components:
            self._connected_components = self._backend.connected_components()
            self._connected_components_needs_recompute = False
        if self._connected_components and self._connected_components_needs_recompute:
            self._connected_components = self._backend.connected_components()
            self._connected_components_needs_recompute = False

        return self._connected_components  # type: ignore

    def layout_coordinates(
        self,
        layout_algorithm: SupportedLayoutAlgorithm = "wpmds_3d",
        only_keep_a_pixels: bool = True,
        get_node_marker_matrix: bool = True,
        cache: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate coordinates and (optionally) node marker counts for plotting.

        Generate a dataframe with coordinates, and (optionally) node marker
        counts to use that can be used for plotting.

        The layout options are:
          - fruchterman_reingold
          - fruchterman_reingold_3d
          - kamada_kawai
          - kamada_kawai_3d
          - pmds
          - pmds_3d
          - wpmds_3d


        The `pmds` options are much faster than the force-directed algorithms fruchterman_reingold
        and kamada_kawai. The `wpmds_3d` option is a weighted version of the `pmds_3d` algorithm.

        :param layout_algorithm: the layout algorithm to use to generate the coordinates
        :param only_keep_a_pixels: If true, only keep the a-pixels
        :param get_node_marker_matrix: Add a matrix of marker counts to each
                                       node if True.
        :param cache: set to `True` in order to cache one call of this this method.
                      It will make subsequent calls to the layout method
                      with the same settings much faster, at the cost of additional
                      memory usage. This can speed things up a lot when plotting
                      e.g. different markers across multiple markers.
        :param random_seed: used as the seed for graph layouts with a stochastic
                            element. Useful to get deterministic layouts across
                            method calls.
        :param **kwargs: will be passed to the underlying layout implementation
        :return: the coordinates and markers (if activated) as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if the provided `layout_algorithm` is not valid
        :raises: ValueError if the provided current graph instance is empty
        """
        if cache:
            return self._cached_layout_coordinates(
                layout_algorithm=layout_algorithm,
                only_keep_a_pixels=only_keep_a_pixels,
                get_node_marker_matrix=get_node_marker_matrix,
                random_seed=random_seed,
                **kwargs,
            )
        else:
            return self._backend.layout_coordinates(
                layout_algorithm=layout_algorithm,
                only_keep_a_pixels=only_keep_a_pixels,
                get_node_marker_matrix=get_node_marker_matrix,
                random_seed=random_seed,
                **kwargs,
            )

    @lru_cache(maxsize=1)
    def _cached_layout_coordinates(self, **kwargs):
        return self._backend.layout_coordinates(**kwargs)

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
        self._backend.add_edges(edges)
        self._connected_components_needs_recompute = True

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
        self._connected_components_needs_recompute = True

    def add_names_to_vertexes(self, vs_names: List[str]) -> None:
        """Rename the current vertices on the graph instance.

        :param vs_names: Add the following vertices to the graph instance.
        :raises ValueError: if the graph is empty
        :raises IndexError: if the number of graph vertices and list of names are
                            of different length
        """
        self._backend.add_names_to_vertexes(vs_names=vs_names)

    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        return f"Graph with {self.vcount()} vertices and {self.ecount()} edges"

    def local_g(
        self,
        k: int = 1,
        use_weights: bool = True,
        normalize_counts: bool = True,
        W: csr_array | None = None,
        method: Literal["gi", "gstari"] = "gi",
    ) -> pd.DataFrame:
        """Compute the local G metric for each node in the graph.

        The local G metric is a measure of the local clustering of a node in the graph.

        :param k: The number of steps in the k-step random walk. Default is 1.
        :param use_weights: Whether to use weights in the computation. When turned off, all
        edge weights will be equal to 1. Default is True.
        :param normalize_counts: Whether to normalize counts to proportions. Default is True.
        :param W: A sparse matrix of custom edge weights. This will override the automated
        computation of edge weights. `W` must have the same dimensions as A. Note that weights can
        be defined for any pair of nodes, not only the pairs represented by edges in `A`. Default is None.
        :param method: The method to use for computing local G. Must be one of 'gi' or 'gstari'.
        'gi' is the original local G metric, which does not consider self-loops, meaning that the
        local marker expression for a node is computed by aggregating the weighted expression of
        its neighbors. 'gstari' is a simplified version of local G that does consider self-loops.
        In other words, the local marker expression of a node also includes the weighted marker
        expression of the node itself. Default is 'gi'.
        :return: A DataFrame of local G-scores for each node and marker.
        """
        counts = self.node_marker_counts
        A = self.get_adjacency_sparse(node_ordering=counts.index)
        return local_g(
            A=A,
            counts=counts,
            k=k,
            use_weights=use_weights,
            normalize_counts=normalize_counts,
            W=W,
            method=method,
        )
