"""Cycle analysis and edge filtering based on cycle participation.

In a graph, edges that do not participate in any cycles can be indicative of noise or erroneous
connections.
Given that an edge indicates spatial proximity between two markers, we expect that true connections
should often be part of cycles formed by other nearby markers. Edges that are not part of any cycles
are more likely to be spurious and can be removed to improve the quality of the graph. To prevent
removing
edges in sparse regions, we only evaluate edges that connect nodes in core-2 or higher layers of the
graph.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
import os
import tempfile
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix

from pixelator.pna.graph.constants import (
    DEFAULT_WORKING_DIR,
    MAX_CYCLE_SEARCH_STEPS,
    MAX_FRONTIER_SIZE_IN_CYCLE_SEARCH,
    MIN_PNA_COMPONENT_SIZE,
)

logger = logging.getLogger(__name__)


class ShortestPathState:
    """Class to maintain the state of the shortest path search."""

    def initialize_frontier(self, edgelist):
        """Initialize the frontier from the edgelist."""
        frontier_x = []
        frontier_y = []
        frontier_index = []
        cnt = 0
        for node1, node2_list in (
            edgelist.group_by("node1_id").agg("node2_id").iter_rows()
        ):
            for node2 in node2_list:
                frontier_index.append((node1, node2))
                for other_node2 in node2_list:
                    if node2 == other_node2:
                        continue
                    frontier_x.append(cnt)
                    frontier_y.append(other_node2)
                cnt += 1
        return frontier_x, frontier_y, frontier_index

    def __init__(self, edgelist):
        """Initialize the shortest path state with the given edgelist."""
        nnodes = max(edgelist["node1_id"].max(), edgelist["node2_id"].max()) + 1
        nedges = len(edgelist)
        frontier_x, frontier_y, frontier_index = self.initialize_frontier(edgelist)
        self.frontier = csc_matrix(
            ([True] * len(frontier_x), (frontier_x, frontier_y)), shape=(nedges, nnodes)
        )
        self.frontier_index = np.array(frontier_index)
        self.already_visited = csc_matrix(
            (
                [True] * len(frontier_index),
                (list(range(len(frontier_index))), [f[0] for f in frontier_index]),
            ),
            shape=(nedges, nnodes),
        )
        self.n_steps = 0


class ShortestPathFinder:
    """Class to find shortest paths in a graph using sparse matrix operations."""

    def __init__(self, edgelist, explore_limit=10000):
        """Initialize the shortest path finder with the given edgelist."""
        self.node_map = (
            pl.concat(
                (
                    edgelist.select(pl.col("node1").alias("node")),
                    edgelist.select(pl.col("node2").alias("node")),
                )
            )
            .unique()
            .sort("node")
            .with_row_index("node_id")
        )
        edgelist = edgelist.join(
            self.node_map.rename({"node": "node1", "node_id": "node1_id"}),
            on="node1",
            how="left",
        ).join(
            self.node_map.rename({"node": "node2", "node_id": "node2_id"}),
            on="node2",
            how="left",
        )
        mat_coords = edgelist.select(["node1_id", "node2_id"]).rows()
        data = [True] * 2 * len(mat_coords)
        self.adj_mat = csc_matrix(
            (
                data,
                (
                    [c[0] for c in mat_coords] + [c[1] for c in mat_coords],
                    [c[1] for c in mat_coords] + [c[0] for c in mat_coords],
                ),
            ),
            shape=(len(self.node_map), len(self.node_map)),
        )
        self.state = ShortestPathState(edgelist)
        self.passing_edges = []
        self.failed_edges = []

        self.explore_limit = explore_limit

    def resolve_frontier(self):
        """Resolve the current frontier to identify reached, exceeded, and failed edges."""
        index = self.state.frontier_index
        reached_target = self.state.already_visited[
            list(range(len(index))), index[:, 1]
        ].A1
        exceeded_frontier_size = (
            self.state.already_visited.sum(axis=1).A1 > self.explore_limit
        ) & (~reached_target)
        failed_to_reach = (self.state.frontier.sum(axis=1).A1 == 0) & (~reached_target)
        for i in np.nonzero(reached_target)[0]:
            edge = (index[i, 0], index[i, 1])
            self.passing_edges.append(
                {
                    "node1_id": edge[0],
                    "node2_id": edge[1],
                    "step": self.state.n_steps,
                }
            )
        for i in np.nonzero(exceeded_frontier_size)[0]:
            edge = (index[i, 0], index[i, 1])
            self.failed_edges.append(
                {
                    "node1_id": edge[0],
                    "node2_id": edge[1],
                    "step": self.state.n_steps,
                    "reason": "exceeded_frontier_size",
                }
            )
        for i in np.nonzero(failed_to_reach)[0]:
            edge = (index[i, 0], index[i, 1])
            self.failed_edges.append(
                {
                    "node1_id": edge[0],
                    "node2_id": edge[1],
                    "step": self.state.n_steps,
                    "reason": "failed_to_reach",
                }
            )

        to_keep = (~reached_target) & (~exceeded_frontier_size) & (~failed_to_reach)
        self.state.frontier = self.state.frontier[to_keep, :]
        self.state.frontier_index = self.state.frontier_index[to_keep]
        self.state.already_visited = self.state.already_visited[to_keep, :]
        return reached_target.sum()

    def step(self):
        """Perform a single step of frontier expansion."""
        self.state.frontier = (
            ((self.state.frontier @ self.adj_mat) > 0).astype(int)
            - self.state.already_visited
        ) > 0
        self.state.frontier.eliminate_zeros()
        self.state.already_visited = (
            self.state.already_visited + self.state.frontier
        ) > 0
        self.state.n_steps += 1
        n_reached = self.resolve_frontier()
        return n_reached

    def run(self, max_steps=20):
        """Run the shortest path finder for a maximum number of steps."""
        cycle_distribution = pd.Series(dtype=int)
        for step in range(max_steps):
            if self.state.frontier.shape[0] == 0:
                break
            n_reached = self.step()
            cycle_distribution.loc[step + 3] = (
                n_reached  # If we reach the target at step 0, it means that it is a neighbor of a neighbor, hence in a 3-cycle.
            )
        cycle_distribution = cycle_distribution.reset_index(name="n_edges").rename(
            columns={"index": "cycle_length"}
        )
        return cycle_distribution

    def get_passing_edges(self):
        """Get edges that participate in cycles."""
        return (
            pl.from_dicts(self.passing_edges)
            .join(
                self.node_map.rename({"node_id": "node1_id", "node": "node1"}),
                on="node1_id",
                how="left",
            )
            .join(
                self.node_map.rename({"node_id": "node2_id", "node": "node2"}),
                on="node2_id",
                how="left",
            )
            .drop(["node1_id", "node2_id"])
        )


def process_component(comp_name, edgelist_path, tmpdir):
    """Process a single component to remove edges not participating in cycles."""
    with duckdb.connect() as con:
        con.execute(
            """
            CREATE TEMP TABLE comp_edgelist AS
            SELECT * FROM read_parquet(?)
            WHERE component = ?
            """,
            [str(edgelist_path), comp_name],
        )
        comp_edgelist = con.execute("SELECT umi1, umi2 FROM comp_edgelist").pl()
        graph = nx.from_edgelist(comp_edgelist.rows())
        core_numbers = pd.Series(nx.core_number(graph))
        high_cores = core_numbers[core_numbers > 1].index
        edgelist_h = comp_edgelist.filter(
            pl.col("umi1").is_in(high_cores) & pl.col("umi2").is_in(high_cores)
        )
        if edgelist_h.height == 0:
            out_parquet = str(Path(tmpdir) / f"component_{comp_name}.parquet")
            con.execute(
                "COPY (SELECT * FROM comp_edgelist) TO ? (FORMAT PARQUET)",
                [out_parquet],
            )
            empty_dist = pd.DataFrame(
                {"cycle_length": pd.Series(dtype=int), "n_edges": pd.Series(dtype=int)}
            )
            return 0, empty_dist

        edgelist_h = edgelist_h.select(
            [pl.col("umi1").alias("node1"), pl.col("umi2").alias("node2")]
        )

        finder = ShortestPathFinder(
            edgelist_h, explore_limit=MAX_FRONTIER_SIZE_IN_CYCLE_SEARCH
        )
        edge_cycle_length_dist = finder.run(max_steps=MAX_CYCLE_SEARCH_STEPS)

        passing_edges = finder.get_passing_edges()
        remaining_edges = pl.concat(
            (
                comp_edgelist.filter(
                    ~pl.col("umi1").is_in(high_cores)
                    | ~pl.col("umi2").is_in(high_cores)
                ).select(["umi1", "umi2"]),
                passing_edges.select(
                    [pl.col("node1").alias("umi1"), pl.col("node2").alias("umi2")]
                ),
            )
        )
        n_removed_edges = comp_edgelist.height - remaining_edges.height
        graph = nx.from_edgelist(remaining_edges.rows())
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        for i, comp in enumerate(components):
            if len(comp) < MIN_PNA_COMPONENT_SIZE:
                # Only process contiguous large components; once a small component is found,
                # all subsequent components are assumed to be too small and are discarded.
                # This is intentional to avoid saving fragmented or noisy subgraphs.
                break

            nodes_df = pl.DataFrame({"umi": list(comp)})
            out_parquet = str(Path(tmpdir) / f"component_{comp_name}_{i}.parquet")
            con.execute(
                """
                COPY (
                    SELECT * FROM comp_edgelist
                    WHERE umi1 IN (SELECT umi FROM nodes_df)
                    AND umi2 IN (SELECT umi FROM nodes_df)
                ) TO ? (FORMAT PARQUET)
                """,
                [out_parquet],
            )

        return n_removed_edges, edge_cycle_length_dist


def remove_no_cycle_edges(
    input_edgelist_path: Path,
    n_threads: int = 1,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[int, pd.DataFrame, Path]:
    """Remove edges that do not participate in any cycles from the edgelist.

    Args:
        input_edgelist_path: Path to the input edgelist Parquet file (hive layout supported).
        n_threads: Number of parallel threads to use.
        working_dir: Directory for the merged output (``working_edgelist_with_cycle_verification``);
            defaults to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Total number of edges removed, the distribution of cycle lengths for remaining edges,
        and the path to the written output (hive-style Parquet under
        ``working_edgelist_with_cycle_verification``).
    """
    output_path = working_dir / "working_edgelist_with_cycle_verification"
    logger.info("Starting removal of no-cycle edges")
    with tempfile.TemporaryDirectory() as tmpdir:
        with duckdb.connect(tmpdir + "/temp_duckdb.db") as con:
            con.execute(
                """
                CREATE TEMP TABLE graph_edgelist AS
                SELECT * FROM read_parquet(?)
                """,
                [str(input_edgelist_path)],
            )
            con.execute(
                """
                CREATE TEMP TABLE tiny_components AS
                SELECT component, COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2) AS n_nodes
                FROM graph_edgelist
                GROUP BY component
                HAVING (COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)) < ?
                """,
                [MIN_PNA_COMPONENT_SIZE],
            )

            tiny_parquet_path = str(Path(tmpdir) / "tiny_components.parquet")
            con.execute(
                """
                COPY (
                    SELECT * FROM graph_edgelist
                    WHERE component IN (SELECT component FROM tiny_components)
                ) TO ? (FORMAT PARQUET)
                """,
                [tiny_parquet_path],
            )
            con.execute("""
                CREATE VIEW filtered_graph_edgelist AS
                SELECT * FROM graph_edgelist
                WHERE component NOT IN (SELECT component FROM tiny_components)
            """)
            components = [
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT component FROM filtered_graph_edgelist"
                ).fetchall()
            ]
            logger.info(
                f"Processing {len(components)} components in parallel using {n_threads} threads."
            )
            n_removed_edges_list = Parallel(n_jobs=n_threads)(
                delayed(process_component)(comp_name, input_edgelist_path, tmpdir)
                for comp_name in components
            )
            if len(n_removed_edges_list) == 0:
                total_removed_edges = 0
                edge_cycle_length_dist = pd.DataFrame(
                    {
                        "cycle_length": pd.Series(dtype=int),
                        "n_edges": pd.Series(dtype=int),
                    }
                )
            else:
                total_removed_edges = sum(
                    n_removed_edges for n_removed_edges, _ in n_removed_edges_list
                )
                edge_cycle_length_dist = (
                    pd.concat([dist for _, dist in n_removed_edges_list])
                    .groupby("cycle_length")
                    .sum()
                )

            logger.info(
                f"Total removed edges in cycle verification: {total_removed_edges}"
            )
            out_paths = [
                str(Path(tmpdir) / p)
                for p in os.listdir(tmpdir)
                if p.endswith(".parquet") and p.startswith("component_")
            ] + [str(Path(tmpdir) / "tiny_components.parquet")]
            con.execute(
                "CREATE TEMP TABLE merged_edgelist AS SELECT * FROM read_parquet(?)",
                [out_paths],
            )
            con.execute(
                "COPY merged_edgelist TO ? (FORMAT PARQUET, PARTITION_BY (component))",
                [str(output_path)],
            )
            return total_removed_edges, edge_cycle_length_dist, output_path
