"""Implementation of the pixelator Graph protocol based on networkx.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
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

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

with warnings.catch_warnings():
    # Graspologic raises a numba related warning here, that we can
    # safely ignore.
    warnings.filterwarnings("ignore", module="graspologic.models.edge_swaps")
    from graspologic.partition import leiden

from networkx.algorithms import bipartite as nx_bipartite
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


class NetworkXGraphBackend(GraphBackend):
    """`NetworkXGraphBackend` represents a graph, using networkx."""

    def __init__(
        self,
        raw: Optional[nx.Graph] = None,
    ):
        """Create a new Graph instance.

        Create a Graph instance (as an end-user this is probably not the interface
        you are looking for). Try `Graph.from_edgelist`.

        :param raw: The underlying raw representation of the graph, defaults to None
        """
        self._raw = raw

    @staticmethod
    def _build_plain_graph_from_edgelist(
        df: pl.LazyFrame,
        create_using: Union[nx.Graph, nx.MultiGraph],
    ) -> Union[nx.Graph, nx.MultiGraph]:
        g = nx.empty_graph(0, create_using)

        # TODO Look at how to deal with setting project_pushdown=False
        # here. If it is needed or not seems to depend on the
        # exact call context, so it might be that we can actually
        # enable it again here and improve the memory usage.
        for idx, row in enumerate(
            df.collect(streaming=True, projection_pushdown=False).iter_rows(
                named=False, buffer_size=1000
            )
        ):
            g.add_edge(row[0], row[1], index=idx)
        return g

    @staticmethod
    def _build_graph_with_node_counts_from_edgelist(
        df: pl.LazyFrame,
        create_using: Union[nx.Graph, nx.MultiGraph],
    ) -> Union[nx.Graph, nx.MultiGraph]:
        unique_markers = set(df.unique("marker").collect()["marker"].to_list())
        initial_marker_dict = {marker: 0 for marker in unique_markers}

        g: nx.Graph = nx.empty_graph(0, create_using)

        for idx, row in enumerate(
            (
                df.select(["upia", "upib", "marker"])
                .collect(streaming=True)
                .iter_rows(named=False, buffer_size=1000)
            )
        ):
            # We are duplicating code here, since it gives
            # a performance boost of roughly 10% here.
            # Which isn't that much, but will add up when
            # we have large edge lists.
            node_1 = row[0]
            node_2 = row[1]
            marker = row[2]

            existing_node_1 = g.nodes.get(node_1)
            if existing_node_1:
                existing_node_1["markers"][marker] += 1
            else:
                marker_dict = initial_marker_dict.copy()
                marker_dict[marker] += 1
                g.add_node(node_1, markers=marker_dict)

            existing_node_2 = g.nodes.get(node_2)
            if existing_node_2:
                existing_node_2["markers"][marker] += 1
            else:
                marker_dict = initial_marker_dict.copy()
                marker_dict[marker] += 1
                g.add_node(node_2, markers=marker_dict)

            g.add_edge(node_1, node_2, index=idx)

        return g

    @staticmethod
    def _add_node_attributes(
        graph: Union[nx.Graph, nx.MultiGraph], a_nodes: set[str]
    ) -> None:
        node_names = {node: node for node in graph.nodes()}
        pixel_type = {node: "A" if node in a_nodes else "B" for node in graph.nodes()}
        type_ = {node: node in a_nodes for node in graph.nodes()}
        nx.set_node_attributes(graph, node_names, "name")
        nx.set_node_attributes(graph, pixel_type, "pixel_type")
        nx.set_node_attributes(graph, type_, "type")

    @staticmethod
    def _project_on_a_nodes(
        graph: Union[nx.Graph, nx.MultiGraph], a_nodes: set[str]
    ) -> Union[nx.Graph, nx.MultiGraph]:
        if isinstance(graph, nx.MultiGraph):
            warnings.warn(
                "Using `use_full_bipartite=False` together with `simplify=False` "
                "will still implicitly simplify the graph, since the multi-edges "
                "will be lost upon A-node projection."
            )
            graph = nx.Graph(graph)

        return nx_bipartite.projected_graph(graph, a_nodes)

    @staticmethod
    def _build_graph_with_marker_counts(
        edgelist: pl.LazyFrame, simplify: bool, use_full_bipartite: bool
    ) -> Union[nx.Graph, nx.MultiGraph]:
        graph = NetworkXGraphBackend._build_graph_with_node_counts_from_edgelist(
            edgelist,
            create_using=nx.Graph if simplify else nx.MultiGraph,
        )
        a_nodes = set(edgelist.select(["upia"]).unique().collect()["upia"].to_list())
        NetworkXGraphBackend._add_node_attributes(graph, a_nodes)
        if use_full_bipartite:
            return graph
        return NetworkXGraphBackend._project_on_a_nodes(graph, a_nodes)

    @staticmethod
    def _build_plain_graph(
        edgelist: pl.LazyFrame, simplify: bool, use_full_bipartite: bool
    ) -> Union[nx.Graph, nx.MultiGraph]:
        graph = NetworkXGraphBackend._build_plain_graph_from_edgelist(
            edgelist.select(pl.col("upia"), pl.col("upib")),
            create_using=nx.Graph if simplify else nx.MultiGraph,
        )
        a_nodes = set(edgelist.select(["upia"]).unique().collect()["upia"].to_list())
        NetworkXGraphBackend._add_node_attributes(graph, a_nodes)
        if use_full_bipartite:
            return graph
        return NetworkXGraphBackend._project_on_a_nodes(graph, a_nodes)

    @staticmethod
    def from_edgelist(
        edgelist: Union[pd.DataFrame, pl.LazyFrame],
        add_marker_counts: bool,
        simplify: bool,
        use_full_bipartite: bool,
    ) -> NetworkXGraphBackend:
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
        :rtype: NetworkXGraphBackend
        :raises: AssertionError when the input edge list is not valid
        """
        if isinstance(edgelist, pd.DataFrame):
            edgelist: pl.LazyFrame = pl.LazyFrame(edgelist)  # type: ignore

        if add_marker_counts:
            graph = NetworkXGraphBackend._build_graph_with_marker_counts(
                edgelist, simplify, use_full_bipartite
            )
        else:
            graph = NetworkXGraphBackend._build_plain_graph(
                edgelist, simplify, use_full_bipartite
            )

        # TODO igraph uses integer indexing. This converts the networkx graph to using
        # the same-ish schema. We probably evaluate if this is really necessary later,
        # or potentially only do it on request.
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

        return NetworkXGraphBackend(raw=graph)

    @staticmethod
    def from_raw(graph: Union[nx.Graph, nx.MultiGraph]) -> NetworkXGraphBackend:
        """Generate a Graph from an networkx.Graph object.

        :param graph: input networkx graph to use
        :return: A pixelator Graph object
        :rtype: NetworkXGraphBackend
        """
        return NetworkXGraphBackend(graph)

    @property
    def raw(self):
        """Get the raw underlying graph representation."""
        return self._raw

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        return NetworkxBasedVertexSequence(
            vertices=[
                NetworkxBasedVertex(v[0], v[1], self.raw)
                for v in self.raw.nodes(data=True)
            ]
        )

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        return NetworkxBasedEdgeSequence(self._raw, self._raw.edges(data=True))

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        return self._raw.number_of_nodes()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        return self._raw.number_of_edges()

    def get_adjacency_sparse(self) -> csr_matrix:
        """Get the sparse adjacency matrix."""
        return nx.to_scipy_sparse_array(self._raw)

    def connected_components(self) -> NetworkxBasedVertexClustering:
        """Get the connected components in the Graph instance."""
        return NetworkxBasedVertexClustering(
            self._raw, nx.connected_components(self._raw)
        )

    def community_leiden(
        self,
        n_iterations: int = 10,
        beta: float = 0.01,
        **kwargs,
    ) -> VertexClustering:
        """Run community detection using the Leiden algorithm."""
        graph = self._raw

        # TODO This is probably not sufficient for
        # some cases, since it looses multi-edge information
        # without translating that to weights.
        # We should look into that once the rest of the code around
        # this has been cleaned up a bit.

        if isinstance(graph, nx.MultiGraph):
            graph = nx.Graph(graph)

        leiden_communities = leiden(
            graph,
            use_modularity=True,
            randomness=beta,
            extra_forced_iterations=n_iterations,
            **kwargs,
        )

        def clusters(leiden_communities):
            communities = defaultdict(set)
            for node, community in leiden_communities.items():
                communities[community].add(node)
            for _, nodes in communities.items():
                yield nodes

        return NetworkxBasedVertexClustering(graph, clusters(leiden_communities))

    def _layout_coordinates(
        self,
        layout_algorithm: str = "fruchterman_reingold",
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
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

        if not self._raw:
            raise ValueError("Trying to get layout for empty Graph instance.")
        raw = self._raw  # type: nx.Graph

        if layout_algorithm == "kamada_kawai":
            layout_inst = nx.kamada_kawai_layout(
                raw, pos=nx.random_layout(raw, seed=random_seed)
            )
        if layout_algorithm == "kamada_kawai_3d":
            layout_inst = nx.kamada_kawai_layout(
                raw, pos=nx.random_layout(raw, seed=random_seed, dim=3), dim=3
            )
        if layout_algorithm == "fruchterman_reingold":
            layout_inst = nx.spring_layout(raw, seed=random_seed)
        if layout_algorithm == "fruchterman_reingold_3d":
            layout_inst = nx.spring_layout(raw, dim=3, seed=random_seed)

        coordinates = pd.DataFrame.from_dict(
            layout_inst,
            orient="index",
            columns=["x", "y"] if len(layout_inst[0]) == 2 else ["x", "y", "z"],
        )
        coordinates.index = [
            str(raw.nodes[node_idx]["name"]) for node_idx in layout_inst.keys()
        ]
        return coordinates

    @staticmethod
    def _normalize_to_unit_sphere(coordinates):
        coordinates[["x_norm", "y_norm", "z_norm"]] = (
            coordinates[["x", "y", "z"]]
            / (1 * np.linalg.norm(np.asarray(coordinates), axis=1))[:, None]
        )
        return coordinates

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
        coordinates = self._layout_coordinates(
            layout_algorithm=layout_algorithm, random_seed=random_seed
        )

        # If we are doing a 3D layout we add the option of normalized
        # vectors where we scale the length of each point vector to be one, so that
        # we have the option of doing a spherical projection of the graph
        if len(coordinates.columns) == 3:
            coordinates = self._normalize_to_unit_sphere(coordinates)

        if get_node_marker_matrix:
            # Added here to avoid circular imports
            from pixelator.graph.utils import create_node_markers_counts

            node_marker_counts = create_node_markers_counts(self)  # type: ignore
            df = pd.concat([coordinates, node_marker_counts], axis=1)
        else:
            df = coordinates

        if only_keep_a_pixels:
            a_node_idx = [v.index for v in self.vs.vertices() if v["pixel_type"] == "A"]
            df = df.iloc[list(a_node_idx),]

        return df

    def get_edge_dataframe(self):
        """Get the edges as a pandas DataFrame."""
        return nx.to_pandas_edgelist(self.raw)

    def get_vertex_dataframe(self):
        """Get all vertices as a pandas DataFrame."""
        return pd.DataFrame(list(self.raw.nodes()), columns=["name"])

    def add_edges(self, edges: Iterable[Tuple[int]]) -> None:  # noqa: DOC501
        """Add edges to the graph instance.

        :param edges: Add the following edges to the graph instance.
        """
        self.raw.add_edges_from(edges)

    def add_vertices(self, n_vertices: int, attrs: Dict[str, List]) -> None:
        """Add some number of vertices to the graph instance.

        :param n_vertices: the number of vertices to be added to the graph instance.
        :param attrs: dict of sequences, all of length equal to the number of vertices
                      to be added, containing the attributes of the new vertices. If
                      `n_vertices=1` then they have to be lists of length 1.
        :raises IndexError: if the number of graph vertices to add and lists of
                            attributes are of different lengths
        """
        raise NotImplementedError()

    def add_names_to_vertexes(self, vs_names: List[str]) -> None:
        """Rename the current vertices on the graph instance.

        :param vs_names: Add the following vertices to the graph instance.
        :raises ValueError: if the graph is empty
        :raises IndexError: if the number of graph vertices and list of names are
                            of different length
        """
        raise NotImplementedError()


class NetworkxBasedVertex(Vertex):
    """A Vertex instance that plays well with NetworkX."""

    def __init__(self, index: int, data: Dict, graph: Union[nx.Graph, nx.MultiGraph]):
        """Create a new NetworkxBasedVertex instance."""
        self._index = index
        self._data = data
        self._graph = graph

    @property
    def index(self):
        """Get the index of the vertex."""
        return self._index

    @property
    def data(self) -> Dict:
        """Get the data of the vertex as a dict."""
        return self._data

    def __getitem__(self, attr: str) -> Any:
        """Get the attr of the provided vertex."""
        return self._data[attr]

    def __setitem__(self, attr: str, value: Any) -> None:
        """Set the attr of the vertex."""
        self._data[attr] = value

    def neighbors(self) -> VertexSequence:
        """Get the neighbors of the vertex."""
        neighbor_vertices = set(self._graph.neighbors(self._index))

        def generate_neighbors():
            for node_idx, data in self._graph.nodes(data=True):
                if node_idx in neighbor_vertices:
                    yield NetworkxBasedVertex(node_idx, data, self._graph)

        return NetworkxBasedVertexSequence(generate_neighbors())


class NetworkxBasedEdge(Edge):
    """An Edge instance backed by a Networkx Edge."""

    def __init__(
        self, edge_tuple: Tuple[int, int, Any], graph: Union[nx.Graph, nx.MultiGraph]
    ):
        """Create a NetworkxBasedEdge instance."""
        self.edge_tuple = edge_tuple
        self._graph = graph

    def __eq__(self, other: object) -> bool:
        """Determine equality of edge and `other`."""
        if isinstance(other, NetworkxBasedEdge):
            return self.index == other.index
        return False

    def __hash__(self) -> int:
        """Compute the hash of the edge."""
        return hash(self.index)

    @property
    def index(self) -> int:
        """The index of the edge."""
        return self.edge_tuple[2]["index"]

    @property
    def vertex_tuple(self) -> Tuple[Vertex, Vertex]:
        """Return the vertices the edge connects as a tuple."""
        v1_idx, v2_idx, _ = self.edge_tuple
        node_1_data = self._graph.nodes[v1_idx]
        node_2_data = self._graph.nodes[v2_idx]
        return (
            NetworkxBasedVertex(v1_idx, node_1_data, graph=self._graph),
            NetworkxBasedVertex(v2_idx, node_2_data, graph=self._graph),
        )


class NetworkxBasedVertexSequence(VertexSequence):
    """Proxy for a networkx based vertex sequence."""

    def __init__(self, vertices: Iterable[NetworkxBasedVertex]) -> None:
        """Instantiate a new NetworkxBasedVertexSequence."""
        self._vertices: Dict[int, NetworkxBasedVertex] = {v.index: v for v in vertices}

    def __len__(self) -> int:
        """Get the number of vertexes."""
        return len(self._vertices.keys())

    def vertices(self) -> Iterable[Vertex]:
        """Get an iterable of vertices."""
        return self._vertices.values()

    def __iter__(self) -> Iterator[Vertex]:
        """Get an iterator over the vertices in the sequence."""
        return iter(self._vertices.values())

    def attributes(self) -> Set[str]:
        """Get all attributes associated with the vertices."""

        def all_attributes():
            for node in self._vertices.values():
                for key in node.data.keys():
                    yield key

        return set(all_attributes())

    def get_vertex(self, vertex_id: int) -> Vertex:
        """Get the Vertex corresponding to the vertex id."""
        return self._vertices[vertex_id]

    def get_attribute(self, attr: str) -> Iterable[Any]:
        """Get the values of the attribute."""
        for node in self._vertices.values():
            yield node.data[attr]

    def select_where(self, key, value) -> VertexSequence:
        """Select a subset of vertices."""
        return NetworkxBasedVertexSequence(
            [
                NetworkxBasedVertex(idx, vertex.data, vertex._graph)
                for idx, vertex in self._vertices.items()
                if vertex.data[key] == value
            ],
        )


class NetworkxBasedEdgeSequence(EdgeSequence):
    """Proxy for a networkx based edge sequence."""

    def __init__(
        self, graph: Union[nx.Graph, nx.MultiGraph], edges: Iterable[NetworkxBasedEdge]
    ) -> None:
        """Instantiate a new NetworkxBasedEdgeSequence."""
        self._graph = graph
        self._edges = edges

    def __len__(self) -> int:
        """Get the number of edges."""
        return len(list(iter(self)))

    def __iter__(self) -> Iterator[Edge]:
        """Get an iterator over the edges."""
        for edge in self._edges:
            yield edge

    def select_where(self, key, value) -> EdgeSequence:
        """Select a subset of edges."""
        return NetworkxBasedEdgeSequence(
            self._graph,
            [
                NetworkxBasedEdge((v1, v2, data), self._graph)
                for v1, v2, data in self._graph.edges(data=True)
                if data[key] == value
            ],
        )

    def select_within(self, component) -> EdgeSequence:
        """Select a subset of edges."""
        return NetworkxBasedEdgeSequence(
            self._graph,
            [
                NetworkxBasedEdge(edge, self._graph)
                for edge in self._graph.subgraph(component).edges(data=True)
            ],
        )


class NetworkxBasedVertexClustering(VertexClustering):
    """Wrapper class for a cluster of vertexes."""

    def __init__(
        self, graph: Union[nx.Graph, nx.MultiGraph], clustering: Iterable[Set]
    ) -> None:
        """Instantiate a NetworkxBasedVertexClustering."""
        self._graph = graph
        self._clustering = list(clustering)

    def __len__(self) -> int:
        """Get the number of clusters."""
        return len(self._clustering)

    def __iter__(self) -> Iterator[NetworkxBasedVertexSequence]:
        """Provide an iterator over the clusters."""
        for cluster in self._clustering:
            yield NetworkxBasedVertexSequence(
                [
                    NetworkxBasedVertex(v[0], v[1], self._graph)
                    for v in self._graph.subgraph(cluster).nodes(data=True)
                ]
            )

    @property
    def modularity(self) -> float:
        """Get the modularity of the clusters."""
        return nx.community.modularity(self._graph, communities=self._clustering)

    def _crossing_edges(self):
        def find_crossing_edges():
            graph = self._graph
            for cluster in self._clustering:
                # Setting `nbunch2=None` means that we look for all edges
                # that go out of that cluster.
                for crossing_edge in nx.edge_boundary(
                    graph, nbunch1=cluster, nbunch2=None, data=True
                ):
                    yield NetworkxBasedEdge(crossing_edge, self._graph)

        return {e for e in find_crossing_edges()}

    def crossing(self) -> EdgeSequence:
        """Get any crossing edges."""
        return NetworkxBasedEdgeSequence(self._graph, self._crossing_edges())

    def giant(self) -> Graph:
        """Get the largest component."""
        from pixelator.graph import Graph

        return Graph(
            NetworkXGraphBackend(
                self._graph.subgraph(max(self._clustering, key=len).copy())
            )
        )

    def subgraphs(self) -> Iterable[Graph]:
        """Get subgraphs of each cluster."""
        from pixelator.graph import Graph

        return [
            Graph(NetworkXGraphBackend(self._graph.subgraph(cluster).copy()))
            for cluster in self._clustering
        ]
