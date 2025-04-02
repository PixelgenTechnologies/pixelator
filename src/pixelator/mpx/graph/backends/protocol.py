"""Protocol of graph backends used by pixelator.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from pixelator.mpx.graph import Graph

SupportedLayoutAlgorithm = Literal[
    "fruchterman_reingold",
    "fruchterman_reingold_3d",
    "kamada_kawai",
    "kamada_kawai_3d",
    "pmds",
    "pmds_3d",
    "wpmds_3d",
]


class GraphBackend(Protocol):
    """Protocol for graph backends."""

    @staticmethod
    def from_edgelist(
        edgelist: Union[pd.DataFrame, pl.LazyFrame],
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
        convert_indices_to_integers: bool = True,
    ) -> GraphBackend:
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
        :param convert_indices_to_integers: convert the indices to integers (this is the default)
        :returns: a Graph instance
        :rtype: GraphBackend
        :raises: AssertionError when the input edge list is not valid
        """
        ...

    @staticmethod
    def from_raw(graph: Any) -> GraphBackend:
        """Generate a Graph from a valid Graph object.

        What the valid graph object is depends on the underlying
        GraphBackend implementation.

        Typically what you want to use is `from_edgelist`.

        :param graph: input graph to use
        :return: A pixelator GraphBackend object
        :rtype: GraphBackend
        """
        ...

    @property
    def raw(self):
        """Get the raw underlying graph representation."""
        ...

    @property
    def vs(self) -> VertexSequence:
        """Get a sequence of the vertices in the Graph instance."""
        ...

    @property
    def es(self) -> EdgeSequence:
        """A sequence of the edges in the Graph instance."""
        ...

    def vcount(self) -> int:
        """Get the total number of vertices in the Graph instance."""
        ...

    def ecount(self) -> int:
        """Get the total number of edges in the Graph instance."""
        ...

    def get_adjacency_sparse(
        self, node_ordering: Iterable[Any] | None = None
    ) -> csr_matrix:
        """Get the sparse adjacency matrix.

        :param node_ordering: Control the node ordering in the adjacency matrix
        :return: a sparse adjacency matrix
        """
        ...

    def connected_components(self) -> VertexClustering:
        """Get the connected components in the Graph instance."""
        ...

    def node_marker_counts(self) -> pd.DataFrame:
        """Get the marker counts of each node as a dataframe.

        :return: node markers as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if graph nodes don't include markers
        """
        ...

    def layout_coordinates(
        self,
        layout_algorithm: SupportedLayoutAlgorithm = "wpmds_3d",
        only_keep_a_pixels: bool = True,
        get_node_marker_matrix: bool = True,
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

        For most cases the `pmds` options should be about 10-100x faster
        than the force directed layout methods, i.e. `fruchterman_reingold`
        and `kamada_kawai`. Among the force directed layout methods,
        `fruchterman_reingold` is generally faster than `kamada_kawai`. The
        `wpmds_3d` method uses edge weights to improve the layout, but is slightly
        slower than `pmds_3d`.

        :param layout_algorithm: the layout algorithm to use to generate the coordinates
        :param only_keep_a_pixels: If true, only keep the a-pixels
        :param get_node_marker_matrix: Add a matrix of marker counts to each
                                       node if True.
        :param random_seed: used as the seed for graph layouts with a stochastic
                            element. Useful to get deterministic layouts across
                            method calls.
        :param **kwargs: will be passed to the underlying layout implementation
        :return: the coordinates and markers (if activated) as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if the provided `layout_algorithm` is not valid
        :raises: ValueError if the provided current graph instance is empty
        """
        ...

    def get_edge_dataframe(self) -> pd.DataFrame:
        """Get the edges as a pandas DataFrame."""
        ...

    def get_vertex_dataframe(self) -> pd.DataFrame:
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


class Vertex(Protocol):
    """Protocol for a single vertex in a graph."""

    @property
    def index(self) -> int:
        """Get the index of the vertex."""
        ...

    def __getitem__(self, attr: str) -> Any:
        """Get the value of the vertex attribute `attr`."""
        ...

    def __str__(self) -> str:
        """Get a string representation of the vertex."""
        return f"Vertex({self.index})"

    def __repr__(self) -> str:
        """Get a representation of the vertex."""
        return str(self)


class Edge(Protocol):
    """Protocol for edges in a graph."""

    @property
    def vertex_tuple(self) -> Tuple[Vertex, Vertex]:
        """Return the vertices the edge connects as a tuple."""
        ...

    @property
    def index(self) -> int:
        """The index of the edge."""
        ...

    def __str__(self) -> str:
        """Get a string representation of the edge."""
        return f"Edge({self.vertex_tuple})"

    def __repr__(self) -> str:
        """Get a representation of the edge."""
        return str(self)


class VertexSequence(Iterable[Vertex], Protocol):
    """A sequence of vertexes from a graph."""

    def vertices(self) -> Iterable[Vertex]:
        """Get an iterable of vertices."""
        ...

    def __len__(self) -> int:
        """Get the number of vertexes."""
        ...

    def __iter__(self) -> Iterator[Vertex]:
        """Get an iterator over the vertices in the sequence."""
        ...

    def attributes(self) -> Set[str]:
        """Get all attributes associated with the vertices."""
        ...

    def get_vertex(self, vertex_id: int) -> Vertex:
        """Get the Vertex corresponding to the vertex id."""
        ...

    def get_attribute(self, attr: str) -> Iterable[Any]:
        """Get the values of the attribute."""
        ...


class EdgeSequence(Iterable[Edge], Protocol):
    """A sequence of edges from a graph."""

    def __len__(self) -> int:
        """Get the number of edges."""
        ...

    def __iter__(self) -> Iterator[Edge]:
        """Get an iterator over the edges."""
        ...

    def select_where(self, key, value) -> EdgeSequence:
        """Select a subset of edges where `key == value`."""
        ...

    def select_within(self, component):
        """Select all edges that are part of `component`."""
        ...


class VertexClustering(Iterable[VertexSequence], Protocol):
    """A cluster of vertexes, such as a community in a graph."""

    def __len__(self) -> int:
        """Get the number of clusters."""
        ...

    def __iter__(self) -> Iterator[VertexSequence]:
        """Provide an iterator over the clusters."""
        ...

    @property
    def modularity(self) -> float:
        """Get the modularity of the clusters."""
        ...

    def crossing(self) -> EdgeSequence:
        """Get any crossing edges."""
        ...

    def giant(self) -> Graph:
        """Get the largest component."""
        ...

    def subgraphs(self) -> Iterable[Graph]:
        """Get subgraphs of each cluster."""
        ...
