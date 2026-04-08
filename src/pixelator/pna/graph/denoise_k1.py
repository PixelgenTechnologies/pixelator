"""Denoise edgelist using core-1 filtering.

Provide functions to denoise edgelist by removing core-1 connected components
that contain nodes in crossing edges. Crossing edges that connect nodes from
core-1 layer are hard to distinguish from actual edges, so we remove the entire
core-1 connected components that are connected to such edges. This helps to
reduce marker bleedover while preserving the more reliable core-2 and higher
core connections.

Copyright © 2026 Pixelgen Technologies AB.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import duckdb
import networkx as nx
import polars as pl
from joblib import Parallel, delayed

from .constants import MIN_PNA_COMPONENT_SIZE

logger = logging.getLogger(__name__)


def process_component(
    component_id: str,
    nodes_in_crossing_edges: pl.DataFrame,
    graph_edgelist_path: Path,
    tmpdir: Path,
) -> int:
    """Process a single component to remove core-1 connected components with crossing edges.

    Args:
        component_id (str): ID of the component to process.
        nodes_in_crossing_edges (pl.DataFrame): DataFrame of nodes in crossing edges with their components.
        graph_edgelist_path (Path): Path to the graph edgelist parquet file.
        tmpdir (Path): Temporary directory to save intermediate files.

    Returns:
        int: Number of nodes removed from the component.

    """
    with duckdb.connect() as con:
        component_nodes_in_crossing = nodes_in_crossing_edges.filter(
            pl.col("component") == component_id
        ).select("umi")
        con.execute(
            """
            CREATE TEMP TABLE component_edge_list AS
            SELECT * FROM read_parquet(?)
            WHERE component = ?
            """,
            [str(graph_edgelist_path), component_id],
        )
        component_edges = con.execute(
            "SELECT umi1, umi2 FROM component_edge_list"
        ).pl()
        # Build graph and compute core numbers
        graph = nx.from_edgelist(component_edges.rows())
        core_numbers = nx.core_number(graph)
        # Add core numbers to edges
        component_edges = component_edges.with_columns(
            [
                pl.col("umi1").replace_strict(core_numbers).alias("n1_core"),
                pl.col("umi2").replace_strict(core_numbers).alias("n2_core"),
            ]
        )
        # Keep only core-1 edges
        core1_edges = component_edges.filter(
            (pl.col("n1_core") == 1) & (pl.col("n2_core") == 1)
        )
        core1_graph = nx.from_edgelist(core1_edges.select(["umi1", "umi2"]).rows())
        core1_cc = list(nx.connected_components(core1_graph))
        if len(core1_cc) == 0:
            nodes_to_remove = pl.DataFrame({"umi": []}, schema={"umi": pl.UInt32})
        else:
            # Build DataFrame of core1 connected components
            def generate_core1_cc_dfs():
                for i, c in enumerate(core1_cc):
                    yield pl.DataFrame({"umi": list(c), "id": i})

            core1_cc_df = pl.concat(generate_core1_cc_dfs())
            crossing_components = (
                core1_cc_df.join(component_nodes_in_crossing, on="umi", how="inner")
                .select("id")
                .unique()
            )
            nodes_to_remove = core1_cc_df.join(
                crossing_components, on="id", how="inner"
            ).select("umi")

        out_parquet = str(tmpdir / f"component_{component_id}_full.parquet")
        con.execute(
            """
            COPY (
                SELECT * FROM component_edge_list
                WHERE umi1 NOT IN (SELECT umi FROM nodes_to_remove)
                AND umi2 NOT IN (SELECT umi FROM nodes_to_remove)
            ) TO ? (FORMAT PARQUET)
            """,
            [out_parquet],
        )

    return nodes_to_remove.height


def denoise_edgelist_core1(
    graph_edgelist_path: Path,
    original_edgelist_path: Path,
    output_path: Path,
    n_threads: int = 1,
):
    """Denoise edgelist by removing core-1 connected components with crossing edges.

    Args:
        graph_edgelist_path (Path): Path to the graph edgelist parquet file.
        original_edgelist_path (Path): Path to the original edgelist parquet file.
        output_path (Path): Path to save the denoised edgelist parquet file.
        n_threads (int): Number of threads to use. Default is 8.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with duckdb.connect(tmpdir + "/temp_duckdb.db") as con:
            con.execute(
                f"CREATE VIEW graph_edgelist AS SELECT * FROM read_parquet('{str(graph_edgelist_path)}')"
            )
            con.execute(
                f"CREATE VIEW original_edgelist AS SELECT * FROM read_parquet('{str(original_edgelist_path)}')"
            )
            con.execute(
                """
                CREATE TEMP TABLE tiny_components AS
                SELECT component
                FROM graph_edgelist
                GROUP BY component
                HAVING (COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)) < {min_size}
            """.format(min_size=MIN_PNA_COMPONENT_SIZE)
            )
            con.execute(
                """
                COPY (
                    SELECT * FROM graph_edgelist
                    WHERE component IN (SELECT component FROM tiny_components)
                ) TO '{output_path}' (FORMAT PARQUET)
            """.format(output_path=tmpdir + "/tiny_components.parquet")
            )
            con.execute("""
                CREATE VIEW filtered_graph_edgelist AS
                SELECT * FROM graph_edgelist
                WHERE component NOT IN (SELECT component FROM tiny_components)
            """)

            con.execute("""
                CREATE TABLE crossing_edges AS
                SELECT umi1, umi2
                FROM original_edgelist
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM filtered_graph_edgelist
                    WHERE filtered_graph_edgelist.umi1 = original_edgelist.umi1 AND filtered_graph_edgelist.umi2 = original_edgelist.umi2
                )
            """)
            logger.info("Denoising edgelist using core-1 filtering.")
            logger.info(
                f"Number of crossing edges: {con.execute('SELECT COUNT(*) FROM crossing_edges').fetchone()[0]}"  # type: ignore
            )
            logger.info(
                f"Number of nodes in crossing edges: {con.execute('SELECT COUNT(DISTINCT umi) FROM (SELECT umi1 AS umi FROM crossing_edges UNION ALL SELECT umi2 AS umi FROM crossing_edges)').fetchone()[0]}"  # type: ignore
            )

            crossing_node_component_map = con.execute("""
                SELECT DISTINCT umi, component FROM (
                SELECT umi1 AS umi, component FROM filtered_graph_edgelist WHERE umi1 IN (SELECT umi1 FROM crossing_edges)
                UNION ALL
                SELECT umi2 AS umi, component FROM filtered_graph_edgelist WHERE umi2 IN (SELECT umi2 FROM crossing_edges)
                )
            """).pl()

            components = [
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT component FROM filtered_graph_edgelist"
                ).fetchall()
            ]
            nodes_to_remove_counts = Parallel(n_jobs=n_threads)(
                delayed(process_component)(
                    comp_id, crossing_node_component_map, graph_edgelist_path, tmpdir
                )
                for comp_id in components
            )
            out_paths = [
                tmpdir + "/" + p
                for p in os.listdir(tmpdir)
                if p.endswith(".parquet") and p.startswith("component_")
            ] + [tmpdir + "/tiny_components.parquet"]
            all_files = ",".join([f"'{p}'" for p in out_paths])
            con.execute(
                f"COPY (SELECT * FROM read_parquet([{all_files}])) TO '{str(output_path)}' (FORMAT PARQUET)"
            )
            return sum(nodes_to_remove_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise edgelist using core-1 filtering."
    )
    parser.add_argument(
        "graph_edgelist", type=str, help="Path to the graph edgelist parquet file."
    )
    parser.add_argument(
        "original_edgelist",
        type=str,
        help="Path to the original edgelist parquet file.",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the denoised edgelist parquet file."
    )
    parser.add_argument(
        "--n_threads", type=int, default=8, help="Number of threads to use."
    )
    args = parser.parse_args()

    denoise_edgelist_core1(
        args.graph_edgelist, args.original_edgelist, args.output_path, args.n_threads
    )
