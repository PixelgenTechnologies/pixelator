"""Functions related to the graph dataclass used in pixelator graph operations.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import logging
from typing import Dict, List, Optional, Tuple

import igraph
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Graph:
    """`Graph` represents a graph, i.e. a collection of vertices and edges."""

    def __init__(
        self,
        raw: Optional[igraph.Graph] = None,
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
        logger.debug(
            "Creating graph from edge list with %i edges",
            edgelist.shape[0],
        )

        vertices = pd.DataFrame(
            set(edgelist["upia"].unique()).union(set(edgelist["upib"].unique()))
        )
        graph = igraph.Graph.DataFrame(
            edgelist, vertices=vertices, directed=False, use_vids=False
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
        return Graph(graph.simplify() if use_full_bipartite and simplify else graph)

    @staticmethod
    def from_raw(graph: igraph.Graph) -> "Graph":
        """Generate a Graph from an igraph.Graph object.

        :param graph: input igraph to use
        :return: A pixelator Graph object
        :rtype: Graph
        """
        return Graph(graph)

    @classmethod
    def union(cls, graphs: List["Graph"]) -> "Graph":
        """Create union of graphs.

        Create a union of the provided graphs, merging any vertices
        which share the same name.

        :param graphs: the graphs to create the union from
        :return: a new graph that is the union of the input `graphs`
        :rtype: Graph
        """
        return cls.from_raw(igraph.union([g._raw for g in graphs]))

    @property
    def vs(self):
        """Get a sequence of the vertices in the Graph instance."""
        return self._raw.vs

    @property
    def es(self):
        """A sequence of the edges in the Graph instance."""
        return self._raw.es

    def vcount(self):
        """Get the total number of vertices in the Graph instance."""
        return self._raw.vcount()

    def ecount(self):
        """Get the total number of edges in the Graph instance."""
        return self._raw.ecount()

    def get_adjacency_sparse(self):
        """Get the sparse adjacency matrix."""
        return self._raw.get_adjacency_sparse()

    @property
    def connected_components(self):
        """Get the connected components in the Graph instance."""
        return self._raw.connected_components

    @property
    def components(self):
        """The components in the Graph instance."""
        return self._raw.components

    def community_leiden(self, **kwargs):
        """Run community detection using the Leiden algorithm."""
        return self._raw.community_leiden(**kwargs)

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

    def add_edges(self, edges: List[Tuple[int]]) -> None:
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
