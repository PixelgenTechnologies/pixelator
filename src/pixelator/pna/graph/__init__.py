"""Graph module.

Functions and classes relating to PNA graphs.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
from collections import defaultdict
from timeit import default_timer as timer
from typing import Iterable, Optional

import networkx as nx
import pandas as pd
import polars as pl
import pyarrow as pa

from pixelator.common.graph import Graph as BaseGraph
from pixelator.common.graph.backends.implementations._networkx import (
    NetworkXGraphBackend,
)
from pixelator.common.graph.backends.protocol import (
    SupportedLayoutAlgorithm,
    VertexClustering,
)

logger = logging.getLogger(__name__)


class PNAGraph(BaseGraph):
    """Graph class for PNA data."""

    def __init__(self, backend: NetworkXGraphBackend):
        """Create a new graph instance."""
        self._backend = backend
        self._connected_components_needs_recompute = False
        self._connected_components: VertexClustering | None = None

    def from_record_batches(edgelist: Iterable[pa.RecordBatch], **kwargs) -> "PNAGraph":
        """Create a graph from record batches."""
        return PNAGraph(PNAGraphBackend.from_record_batches(edgelist, **kwargs))

    def from_edgelist(edgelist: pl.LazyFrame, **kwargs):  # type: ignore
        """Create a graph from an edgelist."""
        return PNAGraph(PNAGraphBackend.from_edgelist(edgelist, **kwargs))

    @property
    def node_marker_counts(self):
        """Return the marker counts for each node."""
        return self._backend.node_marker_counts()

    def layout_coordinates(  # type: ignore
        self,
        layout_algorithm: SupportedLayoutAlgorithm = "wpmds_3d",
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


        The `pmds` options are much faster than the force-directed algorithms fruchterman_reingold
        and kamada_kawai. The `wpmds_3d` option is a weighted version of the `pmds_3d` algorithm.

        :param layout_algorithm: the layout algorithm to use to generate the coordinates
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
        return self._backend.layout_coordinates(
            layout_algorithm=layout_algorithm,
            get_node_marker_matrix=get_node_marker_matrix,
            random_seed=random_seed,
            **kwargs,
        )


class PNAGraphBackend(NetworkXGraphBackend):
    """Graph backend for PNA data that reuses most of the NetworkXGraphBackend."""

    @staticmethod
    def _build_graph_from_row_iterator(
        g: nx.Graph, row_iterator: Iterable[dict], **kwargs
    ):
        read_count_per_node: dict[str, int] = defaultdict(int)
        node_marker_type: dict[str, str] = defaultdict(str)
        node_type: dict[str, str] = defaultdict(str)

        def create_edges():
            for row in row_iterator:
                node1, node2 = row["umi1"], row["umi2"]
                read_count_per_node[node1] += row["read_count"]
                read_count_per_node[node2] += row["read_count"]
                node_marker_type[node1] = row["marker_1"]
                node_marker_type[node2] = row["marker_2"]
                node_type[node1] = "A"
                node_type[node2] = "B"
                yield node1, node2

        g.add_edges_from(create_edges())
        nx.set_node_attributes(g, read_count_per_node, "read_count")
        nx.set_node_attributes(g, node_marker_type, "marker")
        nx.set_node_attributes(g, node_type, "pixel_type")

        node_names = {node: node for node in g.nodes()}
        nx.set_node_attributes(g, node_names, "name")
        return g

    @staticmethod
    def _build_graph_from_lazy_frame(g: nx.Graph, edgelist: pl.LazyFrame, **kwargs):
        return PNAGraphBackend._build_graph_from_row_iterator(
            g, edgelist.collect().iter_rows(named=True), **kwargs
        )

    @staticmethod
    def _build_graph_from_pandas(g: nx.Graph, edgelist: pd.DataFrame, **kwargs):
        return PNAGraphBackend._build_graph_from_row_iterator(
            g, [edge for _, edge in edgelist.iterrows()], **kwargs
        )

    @staticmethod
    def from_edgelist(edgelist: pl.LazyFrame | pd.DataFrame, **kwargs):  # type: ignore
        """Create a graph from an edgelist."""
        g: nx.Graph = nx.empty_graph(0, nx.Graph)
        if isinstance(edgelist, pl.LazyFrame):
            g = PNAGraphBackend._build_graph_from_lazy_frame(g, edgelist, **kwargs)
        elif isinstance(edgelist, pd.DataFrame):
            g = PNAGraphBackend._build_graph_from_pandas(g, edgelist, **kwargs)

        return PNAGraphBackend(g)

    @staticmethod
    def from_record_batches(batches: Iterable[pa.RecordBatch], **kwargs):
        """Create a graph from an edgelist."""
        # TODO This is completely untested!
        g: nx.Graph = nx.empty_graph(0, nx.Graph)
        for batch in batches:
            edgelist = batch.to_pylist()
            g = PNAGraphBackend._build_graph_from_row_iterator(g, edgelist, **kwargs)

        return PNAGraphBackend(g)

    def node_marker_counts(self) -> pd.DataFrame:
        """Return the marker counts for each node."""
        markers_df = pd.DataFrame.from_dict(
            nx.get_node_attributes(self.raw, "marker"), orient="index"
        )
        markers_df = markers_df.reset_index(names="node")
        markers_df = markers_df.pivot_table(
            index="node", columns=0, aggfunc="size", fill_value=0
        )
        markers_df.columns.name = "marker"

        return markers_df

    def layout_coordinates(  # type: ignore
        self,
        layout_algorithm: SupportedLayoutAlgorithm = "wpmds_3d",
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

        if get_node_marker_matrix:
            node_marker_counts = self.node_marker_counts().astype(pd.UInt8Dtype())  # type: ignore
            df = pd.concat([coordinates, node_marker_counts], axis=1)
        else:
            df = coordinates
        pixel_name_and_type = pd.DataFrame.from_records(
            [(v.index, v["pixel_type"]) for v in self.vs.vertices()],
            columns=["index", "pixel_type"],
        )
        pixel_name_and_type = pixel_name_and_type.set_index("index", drop=False)
        df = pd.concat([pixel_name_and_type, df], axis=1)

        logger.debug(
            "Layout computation using %s took %.2f seconds",
            layout_algorithm,
            timer() - start_time,
        )
        return df
