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

from pixelator.mpx.graph import Graph as BaseGraph
from pixelator.mpx.graph.backends.implementations._networkx import NetworkXGraphBackend
from pixelator.mpx.graph.backends.protocol import (
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
    def _build_graph_from_lazy_frame(g: nx.Graph, edgelist: pl.LazyFrame, **kwargs):
        def create_nodes(umi_col, marker_col, type):
            if "marker1" in edgelist.collect_schema().keys():
                rows = (
                    edgelist.rename(
                        {"marker1": "marker_1", "marker2": "marker_2"}, strict=False
                    )
                    .select([pl.col(umi_col), pl.col(marker_col)])
                    .unique()
                    # When renaming and selecting we need to disable projection pushdown
                    .collect(streaming=True, projection_pushdown=False)
                    .iter_rows()
                )
            else:
                rows = (
                    edgelist.select([pl.col(umi_col), pl.col(marker_col)])
                    .unique()
                    .collect(streaming=True)
                    .iter_rows()
                )

            for row in rows:
                node = (row[0], {"marker": row[1], "pixel_type": type})
                yield node

        g.add_nodes_from(
            create_nodes(
                "umi1",
                "marker_1",
                "A",
            )
        )
        g.add_nodes_from(
            create_nodes(
                "umi2",
                "marker_2",
                "B",
            )
        )

        read_count_per_node: dict[str, int] = defaultdict(int)

        def create_edges():
            for row in (
                edgelist.select([pl.col("umi1"), pl.col("umi2"), pl.col("read_count")])
                .collect()
                .iter_rows()
            ):
                node1, node2, read_count = row[0], row[1], row[2]
                read_count_per_node[node1] += read_count
                read_count_per_node[node2] += read_count
                yield node1, node2

        g.add_edges_from(create_edges())

        nx.set_node_attributes(g, read_count_per_node, "read_count")

        node_names = {node: node for node in g.nodes()}
        nx.set_node_attributes(g, node_names, "name")
        return g

    @staticmethod
    def from_edgelist(edgelist: pl.LazyFrame, **kwargs):  # type: ignore
        """Create a graph from an edgelist."""
        g: nx.Graph = nx.empty_graph(0, nx.Graph)
        g = PNAGraphBackend._build_graph_from_lazy_frame(g, edgelist, **kwargs)
        return PNAGraphBackend(g)

    @staticmethod
    def from_record_batches(batches: Iterable[pa.RecordBatch], **kwargs):
        """Create a graph from an edgelist."""
        # TODO This is completely untested!
        g: nx.Graph = nx.empty_graph(0, nx.Graph)
        for batch in batches:
            edgelist = pl.from_arrow(batch).lazy()  # type: ignore
            g = PNAGraphBackend._build_graph_from_lazy_frame(g, edgelist, **kwargs)

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
            node_marker_counts = self.node_marker_counts()  # type: ignore
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
