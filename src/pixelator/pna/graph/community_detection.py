"""Community detection and refinement functions for graph analysis.

Copyright (c) 2025 Pixelgen Technologies AB.
"""

import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import polars as pl
from graspologic_native import leiden
from pixelator_core import run_hybrid_community_detection

from pixelator.pna.cli.common import logger

from .component_recovery_utils import (
    create_working_edgelist,
    filter_connected_components_by_size,
    filter_edgelist_by_read_count,
    initialize_graph_statistics,
    remove_clashing_umis,
)
from .connected_components_common import (
    ConnectedComponentException,
    _name_components_with_umi_hashes,
)
from .constants import (
    LEIDEN_RANDOM_SEED,
    MIN_PNA_COMPONENT_SIZE,
)
from .cycle_analysis import remove_no_cycle_edges
from .denoise_k1 import denoise_edgelist_core1
from .report import GraphStatistics


def get_single_thread_duckdb_config(n_threads: int) -> dict:
    """Get a DuckDB configuration that limits memory usage for multi-threaded processing.

    Args:
        n_threads (int): Number of threads to be used in the multi-threaded processing.

    Returns:
        dict: DuckDB configuration dictionary with memory limit and single thread setting.

    """
    with duckdb.connect() as con:
        available_memory = float(
            con.execute("SELECT current_setting('memory_limit');")
            .fetchone()[0]  # type: ignore
            .split(" ")[0]
        )

    duckdb_single_config = {
        "memory_limit": str(int(available_memory / n_threads)) + "GB",
        "threads": "1",
    }

    return duckdb_single_config


@dataclass
class RefinementOptions:
    """Options for refining components."""

    max_edges_to_remove: int | None = 20
    max_edges_to_remove_relative: float | None = None
    leiden_resolution: float = 1.0
    min_component_size_to_prune: int = 1000


@dataclass
class StagedRefinementOptions:
    """Options for refining components at each stage."""

    initial_stage_options: RefinementOptions = field(
        default_factory=lambda: RefinementOptions(
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=1.0,
        )
    )
    refinement_stage_options: RefinementOptions = field(
        default_factory=lambda: RefinementOptions(
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=0.01,
            min_component_size_to_prune=1000,
        )
    )
    max_component_refinement_depth: int = 1


@dataclass(slots=True)
class MultipletRecoveryStats:
    """Statistics from multiplet recovery."""

    crossing_edges_removed: int = 0
    crossing_edges_removed_in_initial_stage: int = 0
    max_recursion_depth: int = 0


def calculate_post_recovery_component_statistics(
    edgelist_with_components_path: Path,
    component_stats: GraphStatistics,
) -> GraphStatistics:
    """Calculate and update graph statistics after multiplet recovery.

    Args:
        edgelist_with_components_path (Path): Path to the edgelist with components in Parquet format.
        component_stats (GraphStatistics): Graph statistics to be updated.

    Returns:
        GraphStatistics: Updated graph statistics.

    """
    edgelist = pl.scan_parquet(
        edgelist_with_components_path,
        hive_schema={"component": pl.String},
    )
    node_comp_stats = (
        edgelist.group_by("component").agg(
            (pl.n_unique("umi1") + pl.n_unique("umi2")).alias("n_umi"),
            pl.len().alias("n_edges"),
        )
    ).collect()

    component_stats.fraction_nodes_in_largest_component_post_recovery = (
        node_comp_stats.select(pl.col("n_umi")).max()[0]
        / node_comp_stats.select(pl.col("n_umi")).sum()[0]
    )[0, 0]
    component_stats.component_count_post_recovery = node_comp_stats.height
    component_stats.edge_count_post_recovery = node_comp_stats.select(
        pl.sum("n_edges")
    )[0, 0]
    component_stats.node_count_post_recovery = node_comp_stats.select(pl.sum("n_umi"))[
        0, 0
    ]

    return component_stats


def merge_communities_with_many_crossing_edges(
    edgelist: pl.DataFrame,
    node_community_dict: dict,
    max_edges_to_remove: int | None = None,
    max_edges_to_remove_relative: float | None = None,
) -> pd.Series:
    """Merge communities what have many crossing edges between them in an edge list.

    This function takes an edge list and a dictionary with the community mapping
    for each node. It then computes the number of edges between communities and
    if they are higher than the given a threshold. It assigns the same community
    id to the nodes in the highly connected communities. The
    threshold is determined by an absolute count and/or relative to the number
    of nodes in the smaller of the two communities, whichever is higher.
    If one of them is None, only the other one is considered and if they are
    both None, the split communities are not considered for merging.

    :param edgelist: The edge list to process
    :param node_community_dict: A dictionary mapping each node to a community
    :param n_edges: The threshold for the number of edges to be found between
    communities to merge or None to avoid merging
    :returns: The updated community mapping
    """
    community_serie = pd.Series(node_community_dict)
    if max_edges_to_remove is None and max_edges_to_remove_relative is None:
        return community_serie
    if community_serie.nunique() == 1:
        return community_serie

    community_sizes = community_serie.value_counts().to_dict()
    # This will be an edgelist where every node is a community
    # and there is an edge when there are more than maximum_edges to
    # remove between the communities.
    community_edgelist = (
        edgelist.with_columns(
            community1=pl.col("umi1").replace_strict(node_community_dict),
            community2=pl.col("umi2").replace_strict(node_community_dict),
        )
        .with_columns(
            community1=pl.min_horizontal(pl.col("community1"), pl.col("community2")),
            community2=pl.max_horizontal(pl.col("community1"), pl.col("community2")),
        )
        .filter(pl.col("community1") != pl.col("community2"))
        .group_by(["community1", "community2"])
        .len()
        .with_columns(
            community1_size=pl.col("community1").replace_strict(community_sizes),
            community2_size=pl.col("community2").replace_strict(community_sizes),
        )
        .with_columns(
            threshold=pl.max_horizontal(
                max_edges_to_remove,
                max_edges_to_remove_relative
                * pl.min_horizontal(
                    pl.col("community1_size"), pl.col("community2_size")
                ),
            ),
        )
        .filter(pl.col("len") >= pl.col("threshold"))
        .select(["community1", "community2"])
        .rows()
    )

    communities_graph = nx.from_edgelist(community_edgelist)
    for cc in nx.connected_components(communities_graph):
        new_tag = min(cc)
        community_serie[community_serie.isin(cc)] = new_tag

    return community_serie


def refine_component(
    component_id: str,
    component_edgelists_path: Path,
    refinement_options: RefinementOptions,
    duckdb_config: dict = {},
) -> tuple[pd.Series, int, pd.Series]:
    """Refine a component by running Leiden community detection and removing crossing edges.

    Args:
        component_id: ID of the component to refine.
        component_edgelists_path: Path to the component edgelists in Parquet (hive partitioned) format.
        refinement_options: Options for refinement during community detection.
        duckdb_config: Configuration for DuckDB connection.

    Returns:
        pd.Series: Sizes of new components after refinement.
        int: Number of crossing edges removed during refinement.
        pd.Series: Sizes of discarded components after refinement.

    """
    with duckdb.connect(config=duckdb_config) as con:
        edgelist = con.execute(f"""
            SELECT *
            FROM read_parquet('{str(component_edgelists_path)}/component={component_id}', hive_partitioning = true)
        """).pl()
        _, leiden_communities = leiden(
            edgelist.select(pl.col("umi1").cast(pl.Utf8), pl.col("umi2").cast(pl.Utf8))
            .with_columns(weight=pl.lit(1))
            .rows(),
            seed=LEIDEN_RANDOM_SEED,
            use_modularity=True,
            resolution=refinement_options.leiden_resolution,
            iterations=1,
            randomness=0.001,
            trials=1,
            starting_communities=None,
        )
        leiden_communities = {int(k): v for k, v in leiden_communities.items()}  # type: ignore
        community_serie = merge_communities_with_many_crossing_edges(
            edgelist,
            leiden_communities,
            refinement_options.max_edges_to_remove,
            refinement_options.max_edges_to_remove_relative,
        )

        if community_serie.nunique() == 1:  # No refinement happened
            return (
                pd.Series(dtype=int, name="count"),
                0,
                pd.Series(dtype=int, name="count"),
            )

        community_serie = component_id + "_" + community_serie.astype(str)
        pl_community_serie = pl.from_pandas(
            community_serie.reset_index(name="community")
        )

        edgelist = edgelist.with_columns(
            new_comp1=(
                pl.col("umi1").replace_strict(
                    old=pl_community_serie["index"], new=pl_community_serie["community"]
                )
            ),
            new_comp2=(
                pl.col("umi2").replace_strict(
                    old=pl_community_serie["index"], new=pl_community_serie["community"]
                )
            ),
        )
        n_removed_edges = edgelist.filter(
            pl.col("new_comp1") != pl.col("new_comp2")
        ).height
        all_comp_sizes = community_serie.value_counts()
        discard_sizes = all_comp_sizes[
            all_comp_sizes < refinement_options.min_component_size_to_prune
        ]
        comp_sizes = all_comp_sizes[
            all_comp_sizes >= refinement_options.min_component_size_to_prune
        ]

        edgelist = (
            edgelist.filter(pl.col("new_comp1") == pl.col("new_comp2"))
            .with_columns(component=pl.col("new_comp1"))
            .filter(pl.col("component").is_in(comp_sizes.index.to_list()))
            .drop(["new_comp2", "new_comp1"])
        )

        con.execute(
            f"COPY (SELECT * FROM edgelist ORDER BY component) TO '{str(component_edgelists_path)}' (FORMAT PARQUET, PARTITION_BY (component), OVERWRITE_OR_IGNORE)"
        )
        shutil.rmtree(f"{str(component_edgelists_path)}/component={component_id}")

    return comp_sizes, n_removed_edges, discard_sizes


def get_component_sizes(
    component_edgelists_path: Path,
) -> pd.Series:
    """Get sizes of components from edgelist with components."""
    with duckdb.connect() as con:
        component_sizes = con.execute(f"""
            SELECT component, COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2) AS n_umi
            FROM read_parquet(
                '{str(component_edgelists_path)}',
                hive_partitioning = true,
                hive_types = {{'component': VARCHAR}}
            )
            GROUP BY component
        """).pl()
    return component_sizes


def run_leiden_refinement(
    component_edgelists_path: Path,
    refinement_options: StagedRefinementOptions,
    component_stats: GraphStatistics,
    component_sizes: pl.DataFrame | None = None,
    max_workers: int = 10,
) -> tuple[GraphStatistics, pl.DataFrame]:
    """Recovery multiplets by leiden community detection, removing crossing edges between communities.

    Args:
        component_edgelists_path: Path to the component edgelists in Parquet (hive partitioned) format.
        refinement_options: Options for refinement during community detection.
        component_stats: Statistics about the components.
        component_sizes: Optional precomputed sizes of components.
        max_workers: Maximum number of worker processes to use.

    Returns:
        tuple[GraphStatistics, pl.DataFrame]: Updated component statistics and DataFrame of discarded component sizes.

    """
    if component_sizes is None:
        component_sizes = get_component_sizes(component_edgelists_path)

    min_size_to_prune = (
        refinement_options.refinement_stage_options.min_component_size_to_prune
    )
    to_be_refined = component_sizes.filter(pl.col("n_umi") > min_size_to_prune)[
        "component"
    ].to_list()
    duckdb_config = get_single_thread_duckdb_config(max_workers)

    worker_func = partial(
        refine_component,
        component_edgelists_path=component_edgelists_path,
        refinement_options=refinement_options.refinement_stage_options,
        duckdb_config=duckdb_config,
    )
    mp_context = multiprocessing.get_context("spawn")
    n_total_crossing_edges = 0
    all_discard_sizes = []
    for recursion_level in range(refinement_options.max_component_refinement_depth):
        if not to_be_refined:
            logger.debug(
                "No more components to refine, will break the refinement loop."
            )
            break

        all_new_component_sizes = []
        discard_sizes = []
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ) as executor:
            futures = []
            for component in to_be_refined:
                futures.append(executor.submit(worker_func, component_id=component))

            for future in as_completed(futures):
                try:
                    new_comp_sizes, n_crossing_edges, new_discard_sizes = (
                        future.result()
                    )
                    all_new_component_sizes.append(new_comp_sizes)
                    discard_sizes.append(new_discard_sizes)
                    n_total_crossing_edges += n_crossing_edges
                except Exception as e:
                    logger.error(
                        "Worker failed during component refinement: %s",
                        e,
                        exc_info=True,
                    )
                    raise e

        if len(all_new_component_sizes) > 0:
            results = pd.concat(all_new_component_sizes)
            to_be_refined = results[results > min_size_to_prune].index.tolist()
        else:
            to_be_refined = []
        if len(discard_sizes) > 0:
            iteration_discard_sizes_df = pl.from_pandas(
                pd.concat(discard_sizes).reset_index(name="n_umi")
            ).select(
                component=pl.col("index").cast(pl.Utf8),
                n_umi=pl.col("n_umi").cast(pl.UInt32),
            )
        else:
            iteration_discard_sizes_df = pl.DataFrame(
                {
                    "component": pl.Series(dtype=pl.Utf8),
                    "n_umi": pl.Series(dtype=pl.UInt32),
                }
            )
        all_discard_sizes.append(iteration_discard_sizes_df)

    if all_discard_sizes:
        discard_sizes_df = pl.concat(all_discard_sizes)
    else:
        discard_sizes_df = pl.DataFrame(
            {
                "component": pl.Series(dtype=pl.Utf8),
                "n_umi": pl.Series(dtype=pl.UInt32),
            }
        )
    component_stats.crossing_edges_removed = (
        component_stats.crossing_edges_removed_initial_stage + n_total_crossing_edges
    )
    component_stats.max_recursion_depth = recursion_level + 1
    return component_stats, discard_sizes_df


def find_components(
    input_edgelist_path: Path,
    working_dir: Path,
    multiplet_recovery: bool = True,
    edge_cycle_verification: bool = False,
    remove_k1_suspect_nodes: bool = False,
    min_read_count: int = 1,
    refinement_options: StagedRefinementOptions = StagedRefinementOptions(),
    component_size_threshold: bool | tuple[int, int] = (
        MIN_PNA_COMPONENT_SIZE,
        2**32 - 1,
    ),
    n_threads: int = 1,
) -> tuple[GraphStatistics, Path]:
    """Find components in the given edgelist.

    Args:
        input_edgelist_path: Path to the input edgelist in Parquet format.
        working_dir: Directory to use for temporary files and output.
        multiplet_recovery: Whether to perform multiplet recovery.
        edge_cycle_verification: Whether to perform edge cycle verification.
        remove_k1_suspect_nodes: Whether to remove K1 suspect nodes.
        min_read_count: Minimum read count threshold for an edge to be retained.
        refinement_options: Options for staged refinement during community detection.
        component_size_threshold: Minimum size threshold for components to be retained.
        n_threads: Number of threads to use for parallel processing.

    Returns:
        tuple[GraphStatistics, Path]: Component statistics and path to the edgelist with components.

    """
    logger.info("Starting component finding process.")
    component_stats = initialize_graph_statistics(
        collapsed_edgelist_path=input_edgelist_path
    )

    component_stats = remove_clashing_umis(
        edgelist_path=input_edgelist_path,
        target_path=working_dir / "no_clash_edgelist.parquet",
        component_stats=component_stats,
    )

    component_stats = filter_edgelist_by_read_count(
        edgelist_path=working_dir / "no_clash_edgelist.parquet",
        target_path=working_dir / "filtered_edgelist.parquet",
        min_read_count=min_read_count,
        component_stats=component_stats,
    )

    create_working_edgelist(
        input_edgelist_path=working_dir / "filtered_edgelist.parquet",
        node_map_path=working_dir / "node_map.parquet",
        working_edgelist_path=working_dir / "working_edgelist.parquet",
    )

    logger.info("Running FLP + Leiden native step")
    (
        partitioned_edgelist_path,
        pre_recovery_stats,
        post_flp_stats,
        post_recovery_stats,
    ) = run_hybrid_community_detection(
        parquet_file=str(working_dir / "working_edgelist.parquet"),
        resolution=refinement_options.initial_stage_options.leiden_resolution,
        output=str(working_dir / "partitioned_edgelist.parquet"),
        epochs=2,
        randomness=0.001,
        seed=LEIDEN_RANDOM_SEED,
        max_iteration=None,
        multiplet_recovery=multiplet_recovery,
    )

    component_stats.component_count_pre_recovery = (
        pre_recovery_stats.n_connected_components
    )
    component_stats.fraction_nodes_in_largest_component_pre_recovery = (
        pre_recovery_stats.fraction_in_largest_component
    )
    component_stats.node_count_pre_recovery = pre_recovery_stats.node_count
    component_stats.edge_count_pre_recovery = pre_recovery_stats.edge_weight_sum
    component_stats.stranded_nodes_pre_recovery = pre_recovery_stats.stranded_nodes

    n_crossing_edges = (
        pre_recovery_stats.edge_weight_sum - post_recovery_stats.edge_weight_sum
    )
    component_stats.crossing_edges_removed_initial_stage = n_crossing_edges
    component_stats.crossing_edges_removed = n_crossing_edges
    component_stats.post_flp_component_sizes = (
        post_flp_stats.component_size_distribution
    )

    max_post_recovery_component_size = max(
        post_recovery_stats.component_size_distribution.keys(),
        default=0,
    )
    if (
        max_post_recovery_component_size
        < refinement_options.initial_stage_options.min_component_size_to_prune
    ):
        msg = (
            "No connected components found in the graph. Likely they were all filtered away for being too small. "
            "This indicates some serious issue with the data. Will not continue with the rest of the computations."
        )
        raise ConnectedComponentException(msg)

    hive_partitioned_edgelist_path = working_dir / "hive_partitioned_edgelist.parquet"
    logger.debug("Filtering out small components from edge list")
    with duckdb.connect(working_dir / "temp_hivepartition.duckddb") as conn:
        conn.execute(f"""
            CREATE TEMP TABLE component_counts AS
                SELECT
                    component,
                    CAST(
                        COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)
                        AS UINT32
                    ) AS n_umi
                FROM parquet_scan('{str(partitioned_edgelist_path)}')
                GROUP BY component;

            CREATE TABLE discarded_components AS
                SELECT component, n_umi
                FROM component_counts
                WHERE n_umi < {refinement_options.initial_stage_options.min_component_size_to_prune};

            CREATE TABLE edgelist AS
                SELECT e.*
                FROM parquet_scan('{str(partitioned_edgelist_path)}') e
                JOIN component_counts c ON e.component = c.component
                WHERE c.n_umi >= {refinement_options.initial_stage_options.min_component_size_to_prune}
                ORDER BY e.component;

            COPY edgelist TO '{str(hive_partitioned_edgelist_path)}'
            (FORMAT PARQUET, PARTITION_BY (component), OVERWRITE_OR_IGNORE);
        """)
        discard_sizes = conn.execute("SELECT * FROM discarded_components").pl()

    latest_working_edgelist_path = hive_partitioned_edgelist_path

    if multiplet_recovery:
        logger.info("Starting multiplet recovery using Leiden algorithm.")

        if refinement_options.max_component_refinement_depth > 1:
            logger.info("Starting refinement stages on decoarsened edgelist.")
            time_start = time.time()
            refinement_options.max_component_refinement_depth -= 1
            component_stats, new_discarded_sizes = run_leiden_refinement(
                component_edgelists_path=latest_working_edgelist_path,
                refinement_options=refinement_options,
                component_stats=component_stats,
            )
            discard_sizes = pl.concat((discard_sizes, new_discarded_sizes))
            logger.info(
                f"Refinement stages completed in {time.time() - time_start:.2f} seconds."
            )

        component_stats = calculate_post_recovery_component_statistics(
            edgelist_with_components_path=latest_working_edgelist_path,
            component_stats=component_stats,
        )

    new_edgelist_path = working_dir / "unified_edge_list.parquet"
    pl.scan_parquet(
        latest_working_edgelist_path, hive_schema={"component": pl.String}
    ).sink_parquet(new_edgelist_path)
    latest_working_edgelist_path = new_edgelist_path

    if edge_cycle_verification:
        logger.info("Starting edge cycle verification.")
        time_start = time.time()
        n_edges_removed, edge_cycle_length_dist = remove_no_cycle_edges(
            edgelist_path=latest_working_edgelist_path,
            output_path=working_dir / "working_edgelist_with_cycle_verification",
            n_threads=n_threads,
        )
        latest_working_edgelist_path = (
            working_dir / "working_edgelist_with_cycle_verification"
        )
        component_stats.edges_removed_in_cycle_verification = n_edges_removed
        component_stats.edge_cycle_length_distribution = (
            edge_cycle_length_dist.to_dict()["n_edges"]
        )
        logger.info(
            f"Edge cycle verification completed in {time.time() - time_start:.2f} seconds."
        )

    if remove_k1_suspect_nodes:
        logger.info("Starting K1 suspect node removal.")
        time_start = time.time()
        n_nodes_removed = denoise_edgelist_core1(
            graph_edgelist_path=latest_working_edgelist_path,
            original_edgelist_path=working_dir / "working_edgelist.parquet",
            output_path=working_dir / "k1_denoised_working_edgelist",
            n_threads=n_threads,
        )
        latest_working_edgelist_path = working_dir / "k1_denoised_working_edgelist"
        component_stats.umis_removed_in_k1_denoising = n_nodes_removed
        logger.info(
            f"K1 suspect node removal completed in {time.time() - time_start:.2f} seconds."
        )

    latest_edgelist = pl.scan_parquet(
        latest_working_edgelist_path, hive_schema={"component": pl.String}
    )
    latest_edgelist_filtered, component_stats = filter_connected_components_by_size(
        edgelist=latest_edgelist,
        discard_sizes=discard_sizes,
        component_size_threshold=component_size_threshold,
        component_stats=component_stats,
    )
    umi_map = pl.read_parquet(working_dir / "node_map.parquet")
    final_edgelist_with_components = latest_edgelist_filtered.with_columns(
        umi1=pl.col("umi1").replace_strict(
            umi_map["working_name"], umi_map["original_name"]
        ),
        umi2=pl.col("umi2").replace_strict(
            umi_map["working_name"], umi_map["original_name"]
        ),
    )
    final_edgelist_with_components = _name_components_with_umi_hashes(
        final_edgelist_with_components
    )

    resolved_edgelist_path = working_dir / "edgelist_with_resolved_components.parquet"
    final_edgelist_with_components.sink_parquet(resolved_edgelist_path)

    return component_stats, resolved_edgelist_path
