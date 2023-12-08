"""Implementation of the pixelator Graph protocol based on igraph.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import igraph
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix

from pixelator.graph.backends.protocol import (
    Edge,
    EdgeSequence,
    GraphBackend,
    Vertex,
    VertexClustering,
    VertexSequence,
)

if TYPE_CHECKING:
    from pixelator.graph import Graph

logger = logging.getLogger(__name__)


class IgraphGraphBackend(GraphBackend):
    """`IGraphGraphBackend` represents a graph, using igraph."""

    def __init__(
        self,
        raw: Optional[igraph.Graph] = None,
    ):
        """Create a new Graph instance.

        Create a Graph instance (as an end-user this is probably not the interface
        you are looking for). Try `Graph.from_edgelist`.

        :param raw: The underlying raw representation of the graph, defaults to None
        """
        if not raw:
            raw = igraph.Graph()
        self._raw = raw

    @staticmethod
    def from_edgelist(
        edgelist: Union[pd.DataFrame, pl.LazyFrame],
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
    ) -> IgraphGraphBackend:
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
        :returns: a GraphBackend instance
        :rtype: IgraphGraphBackend
        :raises: AssertionError when the input edge list is not valid
        """
        if isinstance(edgelist, pl.LazyFrame):
            edgelist = edgelist.collect().to_pandas()

        logger.debug(
            "Creating graph from edge list with %i edges",
            edgelist.shape[0],
        )

        # igraph requires these types to work properly with indexing the vertices
        edgelist = edgelist.astype({"upia": "object", "upib": "object"})

        vertices = pd.DataFrame(
            set(edgelist["upia"].unique()).union(set(edgelist["upib"].unique()))
        )

        graph = igraph.Graph.DataFrame(
            edgelist,
            vertices=vertices,
            directed=False,
            use_vids=False,
        )
        if "sequence" in edgelist.columns:
            all_sequences = edgelist["sequence"].unique()
            logger.debug(
                "Edge list contains %i sequences",
                len(all_sequences),
            )
        else:
            logger.warning(
                "Edge list with no sequence found",
            )

        if add_marker_counts:
            if "marker" not in edgelist.columns:
                raise AssertionError("Edge list is missing the marker column")

            all_markers = edgelist.marker.unique()
            logger.debug(
                "Adding %i markers information to graph from edge list",
                len(all_markers),
            )

            graph.vs["markers"] = [{m: 0 for m in all_markers} for _ in graph.vs]
            for e in graph.es:
                marker = e["marker"]
                v1, v2 = e.vertex_tuple
                v1["markers"][marker] += 1
                v2["markers"][marker] += 1

        # the type attribute is needed to project (True means second projection
        # which in out case is upib, the first projection would be upia)
        dest_pixels = set(edgelist["upib"])
        graph.vs["type"] = [v["name"] in dest_pixels for v in graph.vs]
        graph.vs["pixel_type"] = list(
            map(lambda x: "B" if x else "A", graph.vs["type"])
        )

        if not use_full_bipartite:
            logger.debug("Projecting graph on UPIA")
            # which argument defines which projection to retrieve (first (0) or
            # second (1))
            graph = graph.simplify().bipartite_projection(
                types="type", multiplicity=False, which=0
            )

        logger.debug("Graph created")
        return IgraphGraphBackend(
            graph.simplify() if use_full_bipartite and simplify else graph
        )

    @staticmethod
    def from_raw(graph: igraph.Graph) -> "IgraphGraphBackend":
        """Generate a Graph from an igraph.Graph object.

        :param graph: input igraph to use
        :return: A pixelator Graph object
        :rtype: IgraphGraphBackend
        """
        return IgraphGraphBackend(graph)

    @property
    def raw(self):
        """Get the raw underlying graph representation."""
        return self._raw

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        return IgraphBasedVertexSequence(self._raw.vs)

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        return IgraphBasedEdgeSequence(self._raw.es)

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        return self._raw.vcount()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        return self._raw.ecount()

    def get_adjacency_sparse(self) -> csr_matrix:
        """Get the sparse adjacency matrix."""
        return self._raw.get_adjacency_sparse()

    def connected_components(self) -> VertexClustering:
        """Get the connected components in the Graph instance."""
        return IgraphBasedVertexClustering(
            self._raw.connected_components(), graph=self._raw
        )

    def community_leiden(
        self,
        n_iterations: int = 2,
        beta: float = 0.01,
        **kwargs,
    ) -> VertexClustering:
        """Run community detection using the Leiden algorithm."""
        return IgraphBasedVertexClustering(
            self._raw.community_leiden(
                objective_function="modularity",
                n_iterations=n_iterations,
                beta=beta,
                **kwargs,
            ),
            graph=self._raw,
        )

    def layout_coordinates(
        self,
        layout_algorithm: str = "fruchterman_reingold",
        only_keep_a_pixels: bool = True,
        get_node_marker_matrix: bool = True,
        random_seed: Optional[int] = None,
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
        :param random_seed: used as the seed for graph layouts with a stochastic
                            element. Useful to get deterministic layouts across
                            method calls.
        :return: the coordinates and markers (if activated) as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if the provided `layout_algorithm` is not valid
        :raises: ValueError if the provided current graph instance is empty
        """
        layout_options = [
            "fruchterman_reingold",
            "fruchterman_reingold_3d",
            "kamada_kawai",
            "kamada_kawai_3d",
        ]
        if layout_algorithm not in layout_options:
            raise AssertionError(
                (
                    f"{layout_algorithm} not allowed `layout_algorithm` option. "
                    f"Options are: {'/'.join(layout_options)}"
                )
            )

        if random_seed:
            random.seed(random_seed)

        if not self._raw:
            raise ValueError("Trying to get layout for empty Graph instance.")
        raw = self._raw  # type: igraph.Graph

        if layout_algorithm == "kamada_kawai":
            layout_inst = raw.layout_kamada_kawai(seed=raw.layout_random(dim=2))
        if layout_algorithm == "kamada_kawai_3d":
            layout_inst = raw.layout_kamada_kawai_3d(seed=raw.layout_random(dim=3))
        if layout_algorithm == "fruchterman_reingold":
            layout_inst = raw.layout_fruchterman_reingold()
        if layout_algorithm == "fruchterman_reingold_3d":
            layout_inst = raw.layout_fruchterman_reingold_3d()

        coordinates = pd.DataFrame(
            layout_inst.coords,
            columns=["x", "y"] if layout_inst.dim == 2 else ["x", "y", "z"],
            index=raw.vs["name"],
        )
        coordinates.index = coordinates.index.astype("str")

        # If we are doing a 3D layout we add the option of normalized
        # vectors where we scale the length of each point vector to be one, so that
        # we have the option of doing a spherical projection of the graph
        if layout_inst.dim == 3:
            coordinates[["x_norm", "y_norm", "z_norm"]] = (
                coordinates[["x", "y", "z"]]
                / (1 * np.linalg.norm(np.asarray(coordinates), axis=1))[:, None]
            )

        if get_node_marker_matrix:
            # Added here to avoid circular imports
            from pixelator.graph.utils import create_node_markers_counts

            node_marker_counts = create_node_markers_counts(self)  # type: ignore
            df = pd.concat([coordinates, node_marker_counts], axis=1)
        else:
            df = coordinates

        if only_keep_a_pixels:
            df = df[~np.array(raw.vs["type"])]

        return df

    def get_edge_dataframe(self):
        """Get the edges as a pandas DataFrame."""
        return self._raw.get_edge_dataframe()

    def get_vertex_dataframe(self):
        """Get all vertices as a pandas DataFrame."""
        return self._raw.get_vertex_dataframe()

    def add_edges(self, edges: Iterable[Tuple[int]]) -> None:
        """Add edges to the graph instance.

        :param edges: Add the following edges to the graph instance.
        """
        if not self._raw:
            self._raw = igraph.Graph()
        self._raw.add_edges(edges)

    def add_vertices(self, n_vertices: int, attrs: Dict[str, List]) -> None:
        """Add some number of vertices to the graph instance.

        :param n_vertices: the number of vertices to be added to the graph instance.
        :param attrs: dict of sequences, all of length equal to the number of vertices
                      to be added, containing the attributes of the new vertices. If
                      `n_vertices=1` then they have to be lists of length 1.
        :raises IndexError: if the number of graph vertices to add and lists of
                            attributes are of different lengths
        """
        if not self._raw:
            self._raw = igraph.Graph()
        for k, v in attrs.items():
            if n_vertices != len(v):
                raise IndexError(
                    (
                        "Number of vertices in graph to add not matching input "
                        f"attributes: ({n_vertices} vs {len(v)})"
                    )
                )
        # this is relying in the underlying igraph implementation
        self._raw.add_vertices(n_vertices, attrs)

    def add_names_to_vertexes(self, vs_names: List[str]) -> None:
        """Rename the current vertices on the graph instance.

        :param vs_names: Add the following vertices to the graph instance.
        :raises ValueError: if the graph is empty
        :raises IndexError: if the number of graph vertices and list of names are
                            of different length
        """
        if not self._raw:
            raise ValueError("Graph is empty")
        if len(self.vs) != len(vs_names):
            raise IndexError(
                (
                    "Number of vertices in graph different than input vertices "
                    f"({len(self.vs)} vs {len(vs_names)})"
                )
            )
        for vertex in self.vs:
            vertex["name"] = vs_names


class IgraphBasedVertex(Vertex):
    """A Vertex instance backed by an igraph.Vertex."""

    def __init__(self, vertex: igraph.Vertex):
        """Create a new IgraphBasedVertex instance."""
        if not isinstance(vertex, igraph.Vertex):
            raise TypeError(f"{vertex} is not `igraph.Vertex`")
        self._vertex = vertex

    @property
    def index(self):
        """Get the index of the vertex."""
        return self._vertex.index

    def __getitem__(self, attr: str) -> Any:
        """Get the attr of the provided vertex."""
        return self._vertex[attr]

    def __setitem__(self, attr: str, value: Any) -> None:
        """Get the attr of the provided vertex."""
        self._vertex[attr] = value

    def neighbors(self):
        """Fetch all the neighboring vertices."""
        return self._vertex.neighbors()


class IgraphBasedEdge(Edge):
    """An Edge instance backed by an igraph.Edge."""

    def __init__(self, edge: igraph.Edge):
        """Create an IgraphBasedEdge instance."""
        self._edge = edge

    @property
    def index(self) -> int:
        """The index of the edge."""
        return self._edge.index

    @property
    def vertex_tuple(self) -> Tuple[Vertex, Vertex]:
        """Return the vertices the edge connects as a tuple."""
        v1, v2 = self._edge.vertex_tuple
        return (IgraphBasedVertex(v1), IgraphBasedVertex(v2))


class IgraphBasedVertexSequence(VertexSequence):
    """Proxy for a igraph.VertexSeq."""

    def __init__(self, vertex_sequence: igraph.VertexSeq) -> None:
        """Instantiate a new IgraphBasedVertexSequence."""
        self._vertex_seq = vertex_sequence

    def vertices(self) -> Iterable[Vertex]:
        """Return an iterable of vertices."""
        return [IgraphBasedVertex(vertex) for vertex in self._vertex_seq]

    def select_where(self, key, value):
        """Select a subset of vertices where key == value."""
        kwargs = {key: value}
        return IgraphBasedVertexSequence(self._vertex_seq.select(**kwargs))

    def __len__(self) -> int:
        """Get the number of vertexes."""
        return len(self._vertex_seq)

    def __iter__(self):
        """Get an iterator over the vertices in the sequence."""
        return iter(self.vertices())

    def attributes(self) -> Set[str]:
        """Get all attributes associated with the vertices."""
        return set(self._vertex_seq.attributes())

    def get_vertex(self, vertex_id: int) -> Vertex:
        """Get the Vertex corresponding to the vertex id."""
        try:
            return IgraphBasedVertex({v.index: v for v in self._vertex_seq}[vertex_id])
        except KeyError as e:
            raise KeyError(
                (
                    f"Vertex {vertex_id} not found in VertexSequence. "
                    f"Contains: {self.vertices()}"
                )
            ) from e

    def get_attribute(self, attr: str) -> Iterable[Any]:
        """Get the values of the attribute."""
        return self._vertex_seq[attr]

    def __setitem__(self, attribute, attribute_vector):
        """Set the given vertex attribute to the values in the attribute vector."""
        self._vertex_seq[attribute] = attribute_vector


class IgraphBasedEdgeSequence(EdgeSequence):
    """Proxy for a igraph.EdgeSeq."""

    def __init__(self, edge_seq: igraph.EdgeSeq) -> None:
        """Instantiate a new IgraphBasedEdgeSequence."""
        self._edge_seq = edge_seq

    def __len__(self):
        """Get the number of edges."""
        return len(self._edge_seq)

    def __iter__(self):
        """Get an iterator over the edges."""
        for edge in self._edge_seq:
            yield IgraphBasedEdge(edge)

    def select_where(self, key, value) -> EdgeSequence:
        """Select a subset of edges."""
        kwargs = {f"{key}_eq": value}
        return IgraphBasedEdgeSequence(self._edge_seq.select(**kwargs))

    def select_within(self, component) -> EdgeSequence:
        """Select a subset of edges."""
        return IgraphBasedEdgeSequence(self._edge_seq.select(_within=component))


class IgraphBasedVertexClustering(VertexClustering):
    """Wrapper class for a cluster of vertexes."""

    def __init__(
        self, vertex_clustering: igraph.VertexClustering, graph: igraph.Graph
    ) -> None:
        """Instantiate a VertexClustering."""
        self._vertex_clustering = vertex_clustering
        self._graph = graph

    def __len__(self) -> int:
        """Get the number of clusters."""
        return len(self._vertex_clustering)

    def __iter__(self) -> Iterator[VertexSequence]:
        """Provide an iterator over the clusters."""
        for cluster in self._vertex_clustering:
            yield IgraphBasedVertexSequence(self._graph.vs[cluster])

    @property
    def modularity(self) -> float:
        """Get the modularity of the clusters."""
        return self._vertex_clustering.modularity

    def crossing(self) -> EdgeSequence:
        """Get crossing edges."""
        crossing_indexes = [
            idx
            for idx, is_crossing in enumerate(self._vertex_clustering.crossing())
            if is_crossing
        ]
        return IgraphBasedEdgeSequence(self._graph.es.select(crossing_indexes))

    def giant(self) -> Graph:
        """Get the largest component."""
        from pixelator.graph import Graph

        return Graph(IgraphGraphBackend(self._vertex_clustering.giant()))

    def subgraphs(self) -> Iterable[Graph]:
        """Get subgraphs of each cluster."""
        from pixelator.graph import Graph

        return [
            Graph(IgraphGraphBackend(g)) for g in self._vertex_clustering.subgraphs()
        ]
