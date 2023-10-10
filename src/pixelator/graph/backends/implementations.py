"""Concrete implementations of graph backends used by pixelator.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import igraph
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pixelator.graph.backends.protocol import _GraphBackend

logger = logging.getLogger(__name__)


class IgraphBasedVertexSequence:
    """Proxy for a igraph.VertexSeq."""

    def __init__(self, raw) -> None:
        """Instantiate a new IgraphBasedVertexSequence."""
        self._raw = raw

    def select(self, **kwargs):
        """Select a subset of vertices.

        See https://python.igraph.org/en/stable/api/igraph.VertexSeq.html#select
        """
        return self._raw.select(**kwargs)

    @property
    def attributes(self):
        """Get all attributes associated with the vertices."""
        return self._raw.attributes

    def __getitem__(self, vertex):
        """Get the provide vertex."""
        return self._raw[vertex]

    def __setitem__(self, attribute, attribute_vector):
        """Set the given vertex attribute to the values in the attribute vector."""
        self._raw[attribute] = attribute_vector


class IgraphBasedEdgeSequence:
    """Proxy for a igraph.EdgeSeq."""

    def __init__(self, raw) -> None:
        """Instantiate a new IgraphBasedEdgeSequence."""
        self._raw = raw

    def select(self, **kwargs):
        """Select a subset of vertices.

        See https://python.igraph.org/en/stable/api/igraph.EdgeSeq.html#select
        """
        return self._raw.select(**kwargs)

    def __getitem__(self, edge):
        """Get the provided edge."""
        return self._raw[edge]

    def __setitem__(self, key, newvalue):
        """Set the given edge attribute to the values in the attribute vector."""
        self._raw[key] = newvalue


class IgraphBasedVertexClustering:
    """Wrapper class for a cluster of vertexes."""

    def __init__(self, raw) -> None:
        """Instantiate a VertexClustering."""
        self._raw = raw

    def __len__(self):
        """Get the number of clusters."""
        return len(self._raw)

    def __iter__(self):
        """Provide an iterator over the clusters."""
        for cluster in self._raw:
            yield cluster

    @property
    def modularity(self):
        """Get the modularity of the clusters."""
        return self._raw.modularity

    def crossing(self):
        """Get any crossing edges."""
        return self._raw.crossing()

    def giant(self):
        """Get the largest component."""
        return self._raw.giant()

    def subgraphs(self):
        """Get subgraphs of each cluster."""
        # Avoid a circular import at init time, for now...
        from pixelator.graph import Graph

        return [Graph(g) for g in self._raw.subgraphs()]


class IgraphGraphBackend(_GraphBackend):
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
        edgelist: pd.DataFrame,
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

    def connected_components(self):
        """Get the connected components in the Graph instance."""
        return IgraphBasedVertexClustering(self._raw.connected_components())

    def community_leiden(self, **kwargs):
        """Run community detection using the Leiden algorithm."""
        return IgraphBasedVertexClustering(self._raw.community_leiden(**kwargs))

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
        )

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

            node_marker_counts = create_node_markers_counts(raw)
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


class NetworkXGraphBackend(_GraphBackend):
    """`IGraphGraphBackend` represents a graph, using networkx."""

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
    def from_edgelist(
        edgelist: pd.DataFrame,
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
        raise NotImplementedError()

    @staticmethod
    def from_raw(graph: nx.Graph) -> "NetworkXGraphBackend":
        """Generate a Graph from an networkx.Graph object.

        :param graph: input igraph to use
        :return: A pixelator Graph object
        :rtype: NetworkXGraphBackend
        """
        raise NotImplementedError()

    @property
    def raw(self):
        """Get the raw underlying graph representation."""
        return self._raw

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        raise NotImplementedError()

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        raise NotImplementedError()

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        raise NotImplementedError()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        raise NotImplementedError()

    def get_adjacency_sparse(self):
        """Get the sparse adjacency matrix."""
        raise NotImplementedError()

    def connected_components(self):
        """Get the connected components in the Graph instance."""
        raise NotImplementedError()

    def community_leiden(self, **kwargs):
        """Run community detection using the Leiden algorithm."""
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_edge_dataframe(self):
        """Get the edges as a pandas DataFrame."""
        raise NotImplementedError()

    def get_vertex_dataframe(self):
        """Get all vertices as a pandas DataFrame."""
        raise NotImplementedError()

    def add_edges(self, edges: Iterable[Tuple[int]]) -> None:
        """Add edges to the graph instance.

        :param edges: Add the following edges to the graph instance.
        """
        raise NotImplementedError()

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
