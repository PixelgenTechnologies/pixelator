"""Implementation of the pixelator Graph protocol based on networkx.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import warnings
from timeit import default_timer as timer
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
    get_args,
)

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from networkx.algorithms import bipartite as nx_bipartite
from scipy.sparse import csr_matrix

from pixelator.common.graph.backends.protocol import (
    Edge,
    EdgeSequence,
    GraphBackend,
    SupportedLayoutAlgorithm,
    Vertex,
    VertexClustering,
    VertexSequence,
)

if TYPE_CHECKING:
    from pixelator.mpx.graph import Graph

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
        df: pl.DataFrame,
        create_using: Union[nx.Graph, nx.MultiGraph],
    ) -> Union[nx.Graph, nx.MultiGraph]:
        g = nx.empty_graph(0, create_using)

        # TODO Look at how to deal with setting project_pushdown=False
        # here. If it is needed or not seems to depend on the
        # exact call context, so it might be that we can actually
        # enable it again here and improve the memory usage.
        for idx, row in enumerate(df.iter_rows(named=False, buffer_size=1000)):
            g.add_edge(row[0], row[1], index=idx)
        return g

    @staticmethod
    def _build_graph_with_node_counts_from_edgelist(
        df: pl.DataFrame,
        create_using: Union[nx.Graph, nx.MultiGraph],
    ) -> Union[nx.Graph, nx.MultiGraph]:
        unique_markers = set(df.select("marker").unique()["marker"].to_list())
        initial_marker_dict = {marker: 0 for marker in unique_markers}

        g: nx.Graph = nx.empty_graph(0, create_using)

        for idx, row in enumerate(
            (
                df.select(["upia", "upib", "marker"]).iter_rows(
                    named=False, buffer_size=1000
                )
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
        edgelist_cl = edgelist.select(["upia", "upib", "marker"]).collect()
        graph = NetworkXGraphBackend._build_graph_with_node_counts_from_edgelist(
            edgelist_cl,
            create_using=nx.Graph if simplify else nx.MultiGraph,
        )
        a_nodes = set(edgelist_cl.select(["upia"]).unique()["upia"].to_list())
        NetworkXGraphBackend._add_node_attributes(graph, a_nodes)
        if use_full_bipartite:
            return graph
        return NetworkXGraphBackend._project_on_a_nodes(graph, a_nodes)

    @staticmethod
    def _build_plain_graph(
        edgelist: pl.LazyFrame, simplify: bool, use_full_bipartite: bool
    ) -> Union[nx.Graph, nx.MultiGraph]:
        edgelist_cl = edgelist.select(["upia", "upib"]).collect()
        graph = NetworkXGraphBackend._build_plain_graph_from_edgelist(
            edgelist_cl,
            create_using=nx.Graph if simplify else nx.MultiGraph,
        )
        a_nodes = set(edgelist_cl.select(["upia"]).unique()["upia"].to_list())
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
        convert_indices_to_integers: bool = True,
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
        :param convert_indices_to_integers: convert the indices to integers (this is the default)
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
        if convert_indices_to_integers:
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

    def get_adjacency_sparse(
        self, node_ordering: Iterable[Any] | None = None
    ) -> csr_matrix:
        """Get the sparse adjacency matrix.

        :param node_ordering: Control the node ordering in the adjacency matrix
        :return: a sparse adjacency matrix
        """
        return nx.to_scipy_sparse_array(self._raw, nodelist=node_ordering)

    def connected_components(self) -> NetworkxBasedVertexClustering:
        """Get the connected components in the Graph instance."""
        return NetworkxBasedVertexClustering(
            self._raw, nx.connected_components(self._raw)
        )

    def _layout_coordinates(
        self,
        layout_algorithm: SupportedLayoutAlgorithm = "wpmds_3d",
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if layout_algorithm not in get_args(SupportedLayoutAlgorithm):
            raise AssertionError(
                (
                    f"{layout_algorithm} not allowed `layout_algorithm` option. "
                    f"Options are: {'/'.join(get_args(SupportedLayoutAlgorithm))}"
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
        if layout_algorithm == "pmds":
            layout_inst = pmds_layout(raw, seed=random_seed, **kwargs)
        if layout_algorithm == "pmds_3d":
            layout_inst = pmds_layout(raw, dim=3, seed=random_seed, **kwargs)
        if layout_algorithm == "wpmds_3d":
            layout_inst = pmds_layout(
                raw, dim=3, weights="prob_dist", seed=random_seed, **kwargs
            )

        coordinates = pd.DataFrame.from_dict(
            layout_inst,
            orient="index",
            columns=["x", "y", "z"] if "3d" in layout_algorithm else ["x", "y"],
        )

        return coordinates

    @staticmethod
    def _normalize_to_unit_sphere(coordinates):
        coordinates[["x_norm", "y_norm", "z_norm"]] = (
            coordinates[["x", "y", "z"]]
            / (1 * np.linalg.norm(np.asarray(coordinates), axis=1))[:, None]
        )
        return coordinates

    def node_marker_counts(self) -> pd.DataFrame:
        """Get the marker counts of each node as a dataframe.

        :return: node markers as a dataframe
        :rtype: pd.DataFrame
        :raises: AssertionError if graph nodes don't include markers
        """
        if "markers" not in self.vs.attributes():
            raise AssertionError("Could not find 'markers' in vertex attributes")
        markers = list(sorted(next(iter(self.vs))["markers"].keys()))
        node_marker_counts = pd.DataFrame.from_records(
            list(self.vs.get_attribute("markers")),
            columns=markers,
            index=list(self.raw.nodes),
        )
        node_marker_counts = node_marker_counts.reindex(
            sorted(node_marker_counts.columns), axis=1
        )
        node_marker_counts.columns.name = "markers"
        node_marker_counts.columns = node_marker_counts.columns.astype(
            "string[pyarrow]"
        )
        node_marker_counts.index.name = "node"
        return node_marker_counts

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
          - pmds
          - pmds_3d
          - fruchterman_reingold
          - fruchterman_reingold_3d
          - kamada_kawai
          - kamada_kawai_3d
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
        start_time = timer()
        coordinates = self._layout_coordinates(
            layout_algorithm=layout_algorithm, random_seed=random_seed, **kwargs
        )

        # If we are doing a 3D layout we add the option of normalized
        # vectors where we scale the length of each point vector to be one, so that
        # we have the option of doing a spherical projection of the graph
        if len(coordinates.columns) == 3:
            coordinates = self._normalize_to_unit_sphere(coordinates)

        if get_node_marker_matrix:
            node_marker_counts = self.node_marker_counts()  # type: ignore
            df = pd.concat([coordinates, node_marker_counts], axis=1)
        else:
            df = coordinates
        pixel_name_and_type = pd.DataFrame.from_records(
            [(v.index, v["name"], v["pixel_type"]) for v in self.vs.vertices()],
            columns=["index", "name", "pixel_type"],
        )
        pixel_name_and_type = pixel_name_and_type.set_index("index", drop=False)
        df = pd.concat([pixel_name_and_type, df], axis=1)

        if only_keep_a_pixels:
            df = df[df["pixel_type"] == "A"]

        logger.debug(
            "Layout computation using %s took %.2f seconds",
            layout_algorithm,
            timer() - start_time,
        )
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

    def __init__(
        self, index: int | str, data: Dict, graph: Union[nx.Graph, nx.MultiGraph]
    ):
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
        self,
        edge_tuple: Tuple[int | str, int | str, Any],
        graph: Union[nx.Graph, nx.MultiGraph],
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
        from pixelator.mpx.graph import Graph

        return Graph(
            NetworkXGraphBackend(
                self._graph.subgraph(max(self._clustering, key=len).copy())
            )
        )

    def subgraphs(self) -> Iterable[Graph]:
        """Get subgraphs of each cluster."""
        from pixelator.mpx.graph import Graph

        return [
            Graph(NetworkXGraphBackend(self._graph.subgraph(cluster).copy()))
            for cluster in self._clustering
        ]


def pmds_layout(
    g: nx.Graph,
    pivots: int = 200,
    dim: int = 2,
    weights: Optional[Union[np.ndarray, str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate a pivot MDS layout for a graph as described in [1]_.

    The algorithm is similar to classical multidimensional scaling (MDS),
    but uses only a smalls set of random pivot nodes. The algorithm is
    considerably faster than MDS and therefore scales better to large graphs.
    The topology of resulting layouts are deterministic for a given seed,
    but may be mirrored across different systems due to variations in
    floating-point precision.

    .. [1] Brandes U, Pich C. Eigensolver Methods for Progressive
        Multidimensional Scaling of Large Data. International Symposium
        on Graph Drawing, 2007. Lecture Notes in Computer Science, vol
        4372. doi: 10.1007/978-3-540-70904-6_6.

    :param g: A networkx graph object
    :param pivots: The number of pivot nodes to use
    :param dim: The dimension of the layout
    :param weights: Edge weights to use for the layout computation.
    Options are:

    * an np.array with non-negative values (same number of elements as edges in g)
    * "prob_dist" to weight each edge (i, j) by -log(P)^3, where P is the probability
        of a random walker to go from i to j in 5 steps and then back again (j->i)
        in 5 steps. For this computation, self-loops are added to the graph to ensure
        that all transitions are possible.
    * None to use unweighted shortest path lengths
    :param seed: Set seed for reproducibility
    :return: A dataframe with layout coordinates
    :rtype: pd.DataFrame
    :raises ValueError: if conditions are not met
    """
    if not nx.is_connected(g):
        raise ValueError("Only connected graphs are supported.")

    n_nodes = len(g.nodes)
    if pivots >= n_nodes:
        total_nodes = n_nodes
        warnings.warn(
            f"'pivots' ({pivots}) should be less than the number of "
            f"nodes ({total_nodes}) in the graph. Using all nodes as 'pivots'."
        )
        pivots = total_nodes

    if dim not in [2, 3]:
        raise ValueError("'dim' must be either 2 or 3.")

    if pivots < dim:
        raise ValueError("'pivots' must be greater than or equal to dim.")

    if n_nodes <= dim:
        raise ValueError(
            f"Number of nodes in the graph ({n_nodes}) must be greater than or equal to 'dim' ({dim})."
        )

    pivot_lower_bound = np.min([np.floor(0.2 * len(g.nodes)), 50])
    if pivots < pivot_lower_bound:
        raise ValueError(
            f"'pivots' must be greater than or equal to {pivot_lower_bound}"
        )

    if isinstance(weights, str):
        if weights != "prob_dist":
            raise ValueError("If 'weights' is a string, it must be 'prob_dist'.")
        weights = -(np.log(_prob_edge_weights(g, k=5)) ** 3)
    elif isinstance(weights, np.ndarray):
        if len(weights) != len(g.edges):
            raise ValueError(
                "'weights' must have the same length as the number of edges in the graph."
            )
        if np.any(weights < 0):
            raise ValueError("All elements in 'weights' must be non-negative.")
    elif weights is not None:
        raise ValueError("'weights' must be a string or an array.")

    if weights is not None:
        edges = list(g.edges)
        edge_weight_dict = {edges[i]: weights[i] for i in range(len(edges))}
        nx.set_edge_attributes(g, edge_weight_dict, "weight")
        weight = "weight"
    else:
        weight = None

    if seed is not None:
        np.random.seed(seed)

    node_list = list(g.nodes)

    # Select random pivot nodes
    pivs = np.random.choice(node_list, pivots, replace=False)

    # Calculate the shortest path length from the pivots to all other nodes
    A = nx.to_scipy_sparse_array(g, weight=weight, nodelist=node_list, format="csr")

    # This is a workaround for what seems to be a bug in the type of
    # the indices of the sparse array created above
    A.indices = A.indices.astype(np.intc, copy=False)
    A.indptr = A.indptr.astype(np.intc, copy=False)
    D = sp.sparse.csgraph.shortest_path(
        A,
        directed=False,
        unweighted=weight is None,
        method="D",
        indices=np.where(np.isin(g.nodes, pivs))[0],
    ).T

    # Center values in rows and columns
    D2 = D**2
    cmean = np.mean(D2, axis=0)
    rmean = np.mean(D2, axis=1)
    D_pivs_centered = D2 - np.add.outer(rmean, cmean) + np.mean(D2)

    # Compute SVD and use distances to compute coordinates for all nodes
    # in an abstract cartesian space
    _, _, Vh = sp.sparse.linalg.svds(D_pivs_centered, k=dim, random_state=seed)

    coordinates = D_pivs_centered @ np.transpose(Vh)
    # Flip the coordinates here to make sure that we get the correct coordinate ordering
    # i.e. iqr(x) > iqr(y) > iqr(z)
    coordinates = np.flip(coordinates, axis=1)

    coordinates = {node_list[i]: coordinates[i, :] for i in range(coordinates.shape[0])}

    return coordinates


def _prob_edge_weights(
    g: nx.Graph,
    k: int = 5,
) -> np.ndarray:
    """Compute edge weights based on k-step transition probabilities.

    The transition probabilities are computed using powers of the
    stochastic matrix of the graph with self-loops allowed. Self-loops
    are necessary for bipartite graphs and without them many transitions
    will be impossible. For instance, if we have a bipartite graph with
    A nodes and B nodes where A can only be connected with B and vice versa,
    there is no possible 2-step path from an A node to a B node. However,
    if we allow self-loops, we can go from an A node to itself and then to
    a B node in two steps. By allowing self-loops, we make all transitions
    within the neighborhood k possible.

    The transition probabilities are computed for a k-step random walk
    by taking the k'th power of the stochastic matrix. Transition
    probabilities are not symmetric, i.e. it matters what node we start
    from. The probability of going from i to j in k steps
    is not the same as the probability of going from j to i in k steps.
    This is impractical for layout algorithms that require a single weight
    per edge. To make the transition probabilities symmetric, we multiply
    the transition probabilities in both directions (pk(i, j) * pk(j, i)).
    This way, we get the probability of going from i to j in k steps and then
    back again in k steps, so it no longer matters if we start in i or j.

    Once we have computed the transition probabilities for all k-step
    paths, we then extract the probabilities for the edges in the original
    graph.

    If we consider an edge (u, v) in the original graph, we now have
    the probability of a random walker going from u to v in k steps and then
    back again in k steps with self-loops allowed. If u anv v are well
    connected (if there are many possible k-step paths between them), this
    probability should be high.

    :param g: A networkx graph object
    :param k: The number of steps in the random walk
    :return: An array of edge weights
    :rtype: np.array
    :raises ValueError: if conditions are not met
    """
    # Check that k is an integer between 1 and 10
    if not isinstance(k, int) and k < 1 or k > 10:
        raise ValueError("'k' must be an integer between 1 and 10.")

    # Get the adjacency matrix. By default it is order by the nodes
    A = nx.to_scipy_sparse_array(g, weight=None, format="csr")

    # Add 1 to the diagonal to allow self-loops
    A = A + sp.sparse.diags_array([1] * A.shape[0], format="csr")

    # Divide by row sum to get the stochastic matrix
    D = sp.sparse.diags_array(1 / A.sum(axis=1), format="csr")
    P = D @ A

    # Compute the transition probabilities for a k-step walk
    min_weight = 0.001  # To avoid having the sparse matrix grow too dense
    P_step = _mat_pow(P, k, prune_threshold=min_weight)
    P_step = (P_step + min_weight * P) / (
        1 + min_weight
    )  # Ensure that the original values are not pruned

    # Keep edges from original graph
    P_step = P_step.multiply(A)

    # Compute bi-directional transition probabilities which are symmetric.
    # Now we get the probability of going from i to j and back again in k
    # steps, so it no longer matters if we start in i or j.
    P_step_bidirectional = P_step.multiply(P_step.T)

    # Extract the transition probabilities for edges in g
    edges = np.array(g.edges)
    nodes = np.array(g.nodes)

    # Get end node IDs for the edges
    node_from = edges[:, 0]
    node_to = edges[:, 1]

    # Find the correct positions to extract from the probability matrix
    index_dict = {val: idx for idx, val in enumerate(nodes)}
    vectorized_index = np.vectorize(lambda x: index_dict.get(x, -1))
    row_indices = vectorized_index(node_from)
    col_indices = vectorized_index(node_to)

    # Extract the probabilities
    edge_probs = np.asarray(P_step_bidirectional[row_indices, col_indices])

    return edge_probs


def _mat_pow(mat, power, prune_threshold: float | None = None):
    mat_power = mat.copy()
    for _ in range(power - 1):
        if prune_threshold:
            mat_power.data[np.abs(mat_power.data) < prune_threshold] = 0
            mat_power.eliminate_zeros()
        mat_power = mat @ mat_power
    return mat_power
