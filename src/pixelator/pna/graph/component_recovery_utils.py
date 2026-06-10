"""Utility functions for the graph stage.

Includes shared connected-component helpers (hashing, UMI-based naming, size filtering)
used by community detection and legacy graph code.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import xxhash
from pixelator_core import PyGraphProperties

from pixelator.common.annotate.cell_calling import find_component_size_limits
from pixelator.common.exceptions import PixelatorBaseException
from pixelator.pna.graph.constants import DEFAULT_WORKING_DIR, MIN_PNA_COMPONENT_SIZE
from pixelator.pna.graph.report import GraphStatistics

logger = logging.getLogger(__name__)


class ConnectedComponentException(PixelatorBaseException):
    """Raised when connected-component computation or filtering fails."""


def populate_component_stats_from_hybrid_detection(
    component_stats: GraphStatistics,
    pre_recovery_stats: PyGraphProperties,
    post_flp_stats: PyGraphProperties,
    post_recovery_stats: PyGraphProperties,
) -> None:
    """Fill graph statistics from hybrid (FLP + Leiden) community-detection outputs.

    Args:
        component_stats: Statistics object to update with filtering information.
        pre_recovery_stats: Pre recovery stats.
        post_flp_stats: Post flp stats.
        post_recovery_stats: Post recovery stats.
    """
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
    component_stats.post_flp_community_sizes = (
        post_flp_stats.component_size_distribution
    )


def hash_component(component: set[int]) -> str:
    """Hash a component deterministically based on its nodes.

    Note: this preserves the historical hashing behavior used by the legacy pipeline.
    """
    hasher = xxhash.xxh3_64()
    for node in sorted(component):
        hasher.update(int(node).to_bytes(length=8, byteorder="little"))
    return hasher.hexdigest()


def name_components_with_umi_hashes(edgelist: pl.LazyFrame) -> pl.LazyFrame:
    """Map each ``component`` id to a stable hex hash from that component's UMIs.

    Groups rows by ``component``, takes the distinct ``umi1`` and ``umi2`` values per group,
    and replaces the ``component`` column with ``hash_component`` applied to their combined set.
    Two components with the same UMI multiset get the same name.

    Args:
        edgelist: Lazy edgelist with ``component``, ``umi1``, and ``umi2`` columns.

    Returns:
        The edgelist with ``component`` rewritten to UMI-derived hash strings.
    """
    comp_umis = (
        edgelist.group_by("component")
        .agg(pl.col("umi1").unique(), pl.col("umi2").unique())
        .collect()
    )
    comp_hashes: dict[object, str] = {}
    for comp, umi1, umi2 in comp_umis.rows():
        comp_hashes[comp] = hash_component(set(umi1 + umi2))

    return edgelist.with_columns(pl.col("component").replace_strict(comp_hashes))


def name_components_with_umi_hashes_from_parquet(
    input_edgelist_path: Path,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> Path:
    """Rewrite component ids to deterministic UMI-based hashes and write parquet."""
    output_path = working_dir / "edgelist_with_hashed_components.parquet"
    with duckdb.connect() as con:
        component_rows = con.execute(f"""
            SELECT
                component,
                LIST(DISTINCT umi1) AS umi1_values,
                LIST(DISTINCT umi2) AS umi2_values
            FROM parquet_scan('{str(input_edgelist_path)}')
            GROUP BY component
        """).fetchall()

        component_hashes = [
            (component, hash_component(set(umi1_values + umi2_values)))
            for component, umi1_values, umi2_values in component_rows
        ]
        con.register(
            "component_hashes",
            pl.DataFrame(
                component_hashes,
                schema=["component", "component_hash"],
                orient="row",
            ).to_arrow(),
        )
        con.execute(f"""
            COPY (
                SELECT
                    e.* EXCLUDE (component),
                    c.component_hash AS component
                FROM parquet_scan('{str(input_edgelist_path)}') e
                JOIN component_hashes c ON e.component = c.component
            ) TO '{str(output_path)}' (FORMAT PARQUET)
        """)
    return output_path


def initialize_graph_statistics(collapsed_edgelist_path: Path) -> GraphStatistics:
    """Initialize and return a GraphStatistics object with all fields set to zero."""
    component_stats = GraphStatistics()
    raw_stats = get_count_statistics(collapsed_edgelist_path)
    component_stats.molecules_input = raw_stats["n_molecules"]
    component_stats.reads_input = raw_stats["n_reads"]
    component_stats.umis_input = raw_stats["n_umi"]
    return component_stats


def get_count_statistics(edgelist_path: Path) -> dict:
    """Get count statistics from an edgelist Parquet file.

    This function reads an edgelist stored in a Parquet file and computes the total number of edges,
    as well as the total number of distinct UMIs present in the edgelist. It returns these
    statistics
    in a dictionary.

    Args:
        edgelist_path: Path to the input Parquet file containing the edgelist.

    Returns:
        dict: A dictionary containing the following keys: - 'n_edges': Total number of edges in the
        edgelist. - 'n_umi': Total number of distinct UMIs in the edgelist. - 'n_reads': Total
        number of reads in the edgelist. - 'n_molecules': Total number of molecules in the edgelist.
    """
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW edgelist AS SELECT * FROM parquet_scan('{str(edgelist_path)}')"
        )
        n_edges, n_reads, n_molecules, n_umi = con.execute("""
            SELECT
                COUNT(*) AS n_edges,
                SUM(read_count) AS n_reads,
                SUM(uei_count) AS n_molecules,
                (
                    SELECT COUNT(DISTINCT umi)
                    FROM (
                        SELECT umi1 AS umi FROM edgelist
                        UNION ALL
                        SELECT umi2 AS umi FROM edgelist
                    )
                ) AS n_umi
            FROM edgelist
        """).fetchone()  # type: ignore

    return {
        "n_edges": n_edges,
        "n_umi": n_umi,
        "n_reads": n_reads,
        "n_molecules": n_molecules,
    }


def write_hive_partitioned_edgelist_without_small_components(
    input_edgelist_path: Path,
    min_component_size_to_prune: int,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, pl.DataFrame]:
    """Remove components below a UMI score threshold and write a hive-partitioned edgelist.

    The per-component score matches hybrid community detection:
    ``COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)``. Rows from components that pass the
    threshold are written as Parquet with ``PARTITION_BY (component)``.

    Args:
        input_edgelist_path: Parquet file with component assignments (e.g. after hybrid detection).
        min_component_size_to_prune: Components with a score strictly below this are dropped.
        working_dir: Directory for a temporary DuckDB file and the hive-partition output. Defaults
            to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Path to the hive-partitioned output and a frame of discarded ``component`` / ``n_umi``.
    """
    hive_partitioned_edgelist_path = working_dir / "hive_partitioned_edgelist.parquet"
    min_sz = int(min_component_size_to_prune)
    logger.debug("Filtering out small components from edge list")
    with duckdb.connect(working_dir / "temp_hivepartition.duckdb") as conn:
        conn.execute(f"""
            CREATE TEMP TABLE component_counts AS
                SELECT
                    component,
                    CAST(
                        COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)
                        AS UINT32
                    ) AS n_umi
                FROM parquet_scan('{str(input_edgelist_path)}')
                GROUP BY component;

            CREATE TABLE discarded_components AS
                SELECT component, n_umi
                FROM component_counts
                WHERE n_umi < {min_sz};

            CREATE TABLE edgelist AS
                SELECT e.*
                FROM parquet_scan('{str(input_edgelist_path)}') e
                JOIN component_counts c ON e.component = c.component
                WHERE c.n_umi >= {min_sz}
                ORDER BY e.component;

            COPY edgelist TO '{str(hive_partitioned_edgelist_path)}'
            (FORMAT PARQUET, PARTITION_BY (component), OVERWRITE_OR_IGNORE);
        """)
        discard_sizes = conn.execute("SELECT * FROM discarded_components").pl()
    return hive_partitioned_edgelist_path, discard_sizes


def find_clashing_umis(
    input_file: Path, component_stats: GraphStatistics
) -> tuple[pl.Series, GraphStatistics]:
    """Identify and save clashing Unique Molecular Identifiers (UMIs) from an input Parquet file.

    This function processes an edgelist stored in a Parquet file, where each row contains
    `umi1`, `umi2`, `marker_1`, and `marker_2`. It identifies UMIs that:
    1. Appear as `umi1` associated with multiple `marker_1` values.
    2. Appear as `umi2` associated with multiple `marker_2` values.
    3. Are present in both `umi1` and `umi2`.

    Args:
        input_file: Path to the input Parquet file containing the edgelist.
        component_stats: Statistics object to update with clash information.

    Returns:
        (pl.Series, GraphStatistics): A tuple containing a Polars Series of clashing UMIs
        and the updated component statistics.
    """
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW edgelist AS SELECT * FROM parquet_scan('{str(input_file)}')"
        )

        umi1_clashes = con.execute("""
            SELECT umi1 AS umi
            FROM edgelist
            GROUP BY umi1
            HAVING COUNT(DISTINCT marker_1) > 1
        """).pl()

        umi2_clashes = con.execute("""
            SELECT umi2 AS umi
            FROM edgelist
            GROUP BY umi2
            HAVING COUNT(DISTINCT marker_2) > 1
        """).pl()

        umi1_umi2_clashes = con.execute("""
            SELECT DISTINCT umi
            FROM (
                SELECT umi1 AS umi FROM edgelist
                INTERSECT
                SELECT umi2 AS umi FROM edgelist
            )
        """).pl()

    component_stats.umi1_clashes = umi1_clashes.shape[0]
    component_stats.umi2_clashes = umi2_clashes.shape[0]
    component_stats.umi1_umi2_clashes = umi1_umi2_clashes.shape[0]
    umi_clashes = pl.concat([umi1_clashes, umi2_clashes, umi1_umi2_clashes]).unique()

    return umi_clashes["umi"], component_stats


def remove_umis(
    input_edgelist_path: Path,
    umis_to_remove: pl.Series,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> Path:
    """Remove specified UMIs from an edgelist and write the filtered edgelist under ``working_dir``.

    Args:
        input_edgelist_path: Input Parquet edgelist.
        umis_to_remove: UMIs whose edges should be dropped.
        working_dir: Output directory; defaults to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Path to the written Parquet file (``no_clash_edgelist.parquet`` in ``working_dir``).
    """
    target_path = working_dir / "no_clash_edgelist.parquet"
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW edgelist AS SELECT * FROM parquet_scan('{str(input_edgelist_path)}')"
        )
        umis_to_remove_df = umis_to_remove.to_frame()
        con.execute("""
            CREATE VIEW filtered_edgelist AS
            SELECT * FROM edgelist
            WHERE umi1 NOT IN (SELECT umi FROM umis_to_remove_df)
            AND umi2 NOT IN (SELECT umi FROM umis_to_remove_df)
        """)

        con.execute(
            f"COPY (SELECT * FROM filtered_edgelist) TO '{str(target_path)}' (FORMAT PARQUET)"
        )
    return target_path


def remove_clashing_umis(
    input_edgelist_path: Path,
    component_stats: GraphStatistics,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, GraphStatistics]:
    """Remove clashing UMIs from an edgelist and save the filtered edgelist.

    This function identifies clashing UMIs in the edgelist and removes any edges that contain
    these UMIs. The filtered edgelist is written under ``working_dir``.

    Args:
        input_edgelist_path: Path to the input Parquet file containing the edgelist.
        component_stats: Statistics object to update with clash information.
        working_dir: Output directory; defaults to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Path to the filtered edgelist and updated component statistics.
    """
    umis_to_remove, updated_stats = find_clashing_umis(
        input_file=input_edgelist_path, component_stats=component_stats
    )

    no_clash_edgelist_path = remove_umis(
        input_edgelist_path=input_edgelist_path,
        umis_to_remove=umis_to_remove,
        working_dir=working_dir,
    )
    no_collision_stats = get_count_statistics(no_clash_edgelist_path)

    updated_stats.molecules_post_umi_collision_removal = no_collision_stats[
        "n_molecules"
    ]
    updated_stats.reads_post_umi_collision_removal = no_collision_stats["n_reads"]

    return no_clash_edgelist_path, updated_stats


def create_working_edgelist(
    input_edgelist_path: Path,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, Path]:
    """Build a dense node-id edgelist and a map from original to working node names.

    Args:
        input_edgelist_path: Path to the input edgelist in Parquet format.
        working_dir: Directory for ``node_map.parquet`` and ``working_edgelist.parquet``; defaults
            to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        ``(working_edgelist_path, node_map_path)``.
    """
    node_map_path = working_dir / "node_map.parquet"
    working_edgelist_path = working_dir / "working_edgelist.parquet"
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW input_edgelist AS SELECT * FROM parquet_scan('{str(input_edgelist_path)}')"
        )

        con.execute("""
            CREATE VIEW node_map AS
            WITH all_umis AS (
                SELECT umi1 AS original_name FROM input_edgelist
                UNION
                SELECT umi2 AS original_name FROM input_edgelist
            )
            SELECT
                original_name,
                CAST(ROW_NUMBER() OVER (ORDER BY original_name) - 1 AS UINT64) AS working_name
            FROM all_umis
        """)

        con.execute("""
            CREATE VIEW working_edgelist AS
            SELECT
                nm1.working_name AS umi1,
                nm2.working_name AS umi2,
                ie.read_count,
                ie.uei_count,
                ie.marker_1,
                ie.marker_2
            FROM input_edgelist ie
            JOIN node_map nm1 ON ie.umi1 = nm1.original_name
            JOIN node_map nm2 ON ie.umi2 = nm2.original_name
        """)

        con.execute(
            f"COPY (SELECT * FROM node_map) TO '{str(node_map_path)}' (FORMAT PARQUET)"
        )
        con.execute(
            f"COPY (SELECT * FROM working_edgelist) TO '{str(working_edgelist_path)}' (FORMAT PARQUET)"
        )

    return working_edgelist_path, node_map_path


def filter_edgelist_by_read_count(
    input_edgelist_path: Path,
    min_read_count: int,
    component_stats: GraphStatistics,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, GraphStatistics]:
    """Filter edges in the edgelist by minimum read count.

    This function reads an edgelist from a Parquet file, filters out edges with read counts
    below a specified threshold, and saves the filtered edgelist under ``working_dir``.

    Args:
        input_edgelist_path: Path to the input Parquet file containing the edgelist.
        min_read_count: Minimum read count threshold for filtering edges.
        component_stats: Statistics object to update with filtering information.
        working_dir: Output directory; defaults to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Path to the filtered edgelist and updated graph statistics.
    """
    filtered_edgelist_path = working_dir / "filtered_edgelist.parquet"
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW edgelist AS SELECT * FROM parquet_scan('{str(input_edgelist_path)}')"
        )
        con.execute(
            f"CREATE VIEW filtered_edgelist AS SELECT * FROM edgelist WHERE read_count >= {min_read_count}"
        )
        con.execute(
            f"COPY (SELECT * FROM filtered_edgelist) TO '{str(filtered_edgelist_path)}' (FORMAT PARQUET)"
        )

    filtered_stats = get_count_statistics(filtered_edgelist_path)
    component_stats.molecules_post_read_count_filtering = filtered_stats["n_molecules"]
    component_stats.reads_post_read_count_filtering = filtered_stats["n_reads"]

    return filtered_edgelist_path, component_stats


def save_new_working_edgelist(
    input_working_edgelist_path: Path,
    new_assignments_path: Path,
    component_column_name: str,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, dict[str, int]]:
    """Save a new working edgelist with updated component assignments.

    This function reads a working edgelist and new component assignments from Parquet files,
    merges them to update the component assignments in the edgelist, and saves the resulting
    edgelist under ``working_dir``. It also computes statistics about crossing vs remaining edges.

    Args:
        input_working_edgelist_path: Path to the input working edgelist Parquet file.
        new_assignments_path: Path to the Parquet file containing new component assignments.
        component_column_name: Name of the component column in the assignments file.
        working_dir: Output directory; defaults to ``DEFAULT_WORKING_DIR`` (``/tmp``).

    Returns:
        Path to the updated edgelist and a dict with ``n_crossing_edges`` and
        ``n_remaining_edges``.
    """
    output_path = working_dir / "working_edgelist_with_new_assignments.parquet"
    w = str(input_working_edgelist_path)
    a = str(new_assignments_path)
    with duckdb.connect() as con:
        c1 = component_column_name + "_1"
        c2 = component_column_name + "_2"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE working_view AS
            SELECT
                e.*,
                a1.{component_column_name} AS {c1},
                a2.{component_column_name} AS {c2}
            FROM read_parquet('{w}') e
            JOIN read_parquet('{a}') a1 ON e.umi1 = a1.umi
            JOIN read_parquet('{a}') a2 ON e.umi2 = a2.umi
        """)

        stats = con.execute(f"""
        SELECT
            count(*) FILTER (WHERE {c1} != {c2}) as n_crossing,
            count(*) FILTER (WHERE {c1} = {c2}) as n_remaining
        FROM working_view
        """).fetchone()
        # Mypy cannot infer that `stats` is not None here
        stats = typing.cast(tuple[int, int], stats)
        n_crossing_edges = stats[0]
        n_remaining_edges = stats[1]

        con.execute(f"""
        COPY (
            SELECT
                * EXCLUDE ({c1}, {c2}),
                {c1} AS {component_column_name}
            FROM working_view
            WHERE {c1} = {c2}
        ) TO '{str(output_path)}' (FORMAT PARQUET)
        """)

    return output_path, {
        "n_crossing_edges": n_crossing_edges,
        "n_remaining_edges": n_remaining_edges,
    }


def create_component_size_data_frame(
    input_edgelist_path: Path,
) -> pl.DataFrame:
    """Find component sizes from a parquet edgelist.

    Add pre-filtering connected component size statistics to the component stats.

    Args:
        input_edgelist_path: Path to the input parquet edgelist with ``component``, ``umi1``, and
        ``umi2`` columns.

    Returns:
        A Polars DataFrame with columns ``component`` and ``n_umi``, where ``n_umi`` is the count
        of distinct UMIs in that component (counting both ``umi1`` and ``umi2``).
    """
    with duckdb.connect() as con:
        component_sizes = con.execute(f"""
            SELECT
                component,
                CAST(
                    COUNT(DISTINCT umi1) + COUNT(DISTINCT umi2)
                    AS UINT32
                ) AS n_umi
            FROM parquet_scan('{str(input_edgelist_path)}')
            GROUP BY component
        """).pl()
    return component_sizes


def filter_connected_components_by_size(
    input_edgelist_path: Path,
    component_size_threshold: bool | tuple[int, int],
    discard_sizes: pl.DataFrame,
    component_stats: GraphStatistics,
    dynamic_lowest_passable_bound=None,
    working_dir: Path = DEFAULT_WORKING_DIR,
) -> tuple[Path, GraphStatistics]:
    """Filter connected components by size and get statistics.

    This function filters connected components in an edgelist based on their sizes. It computes the
    sizes of each component,
    applies size thresholds (either dynamic or hard thresholds), and filters out components that do
    not meet the criteria.
    It also updates the component statistics with information about the filtering process.

    Args:
        input_edgelist_path: Path to the input parquet edgelist.
        component_size_threshold: Size threshold for filtering components. If True, dynamic
            thresholds are used. If False, no filtering is applied. If a tuple, it specifies
            (min_size, max_size) for hard thresholds.
        discard_sizes: DataFrame containing sizes of discarded components.
        component_stats: Statistics object to update with filtering information.
        dynamic_lowest_passable_bound: Lowest passable bound for dynamic thresholding. Defaults to
            None.
        working_dir: Directory for temporary parquet output.

    Returns:
        tuple[Path, GraphStatistics]: Path to the filtered parquet edgelist and updated component
        statistics.

    Raises:
        ConnectedComponentException: If no components remain after filtering.
    """
    component_sizes = create_component_size_data_frame(input_edgelist_path)
    component_sizes = pl.concat([component_sizes, discard_sizes], how="vertical")

    unique, counts = np.unique(
        component_sizes["n_umi"].cast(pl.Int32), return_counts=True
    )
    component_stats.pre_filtering_component_sizes = dict(zip(unique, counts))
    component_stats.component_count_pre_component_size_filtering = (
        component_sizes.height
    )
    if component_size_threshold is True:
        logger.debug("Filtering components by dynamic size thresholds")
        try:
            passing_components, min_size = filter_components_by_size_dynamic(
                component_sizes,
                lowest_passable_bound=dynamic_lowest_passable_bound,
            )
            component_stats.component_size_min_filtering_threshold = min_size
        except Exception as e:
            # This is a hack since the exception is not properly
            # by passed the spline interpolation package.
            if str(type(e)) == "<class 'dfitpack.error'>":
                msg = (
                    "Could not find component size filters, this probably means that no components (i.e. cells) were formed. "
                    "If you are running on the command line you can try to use --component-size-max-threshold and "
                    " --component-size-min-threshold to set hard thresholds, but most likely the problem is with the input data."
                )
                raise ConnectedComponentException(msg)
            else:
                raise e
    else:
        if component_size_threshold is False:
            min_size = 0
            max_size = np.iinfo(np.uint64).max
        else:
            min_size, max_size = component_size_threshold
        component_stats.component_size_max_filtering_threshold = max_size
        component_stats.component_size_min_filtering_threshold = min_size
        logger.debug(
            "Filtering components by hard size thresholds, min: %s, max: %s",
            min_size,
            max_size,
        )
        passing_components = filter_components_by_size_hard_thresholds(
            component_sizes, min_size, max_size
        )

    logger.info(f"{len(passing_components)} left after filtering.")
    if len(passing_components) == 0:
        msg = (
            "No connected components found in the graph. Likely they were all filtered away for being to small. "
            "This indicates some serious issue with the data. Will not continue with the rest of the computations."
        )
        raise ConnectedComponentException(msg)

    component_stats.component_count_post_component_size_filtering = len(
        passing_components
    )

    filtered_edgelist_path = working_dir / "component_size_filtered_edgelist.parquet"
    with duckdb.connect() as con:
        con.register(
            "passing_components",
            passing_components.to_frame().to_arrow(),
        )
        con.execute(f"""
            COPY (
                SELECT e.*
                FROM parquet_scan('{str(input_edgelist_path)}') e
                JOIN passing_components p
                ON e.component = p.component
            ) TO '{str(filtered_edgelist_path)}' (FORMAT PARQUET)
        """)

    return filtered_edgelist_path, component_stats


def filter_components_by_size_dynamic(
    component_sizes: pl.DataFrame,
    lowest_passable_bound: int | None = MIN_PNA_COMPONENT_SIZE,
) -> tuple[pl.Series, int | None]:
    """Filter components by size using dynamic thresholds.

    Args:
        component_sizes: DataFrame with columns `component` and `n_umi`.
        lowest_passable_bound: Lowest passable bound.

    Returns:
        Components that pass the filter, and the computed lower bound.
    """
    if lowest_passable_bound is None:
        lowest_passable_bound = MIN_PNA_COMPONENT_SIZE

    lower_bound = find_component_size_limits(
        component_sizes=component_sizes["n_umi"].to_numpy(), direction="lower"
    )
    if lower_bound is None or lower_bound < lowest_passable_bound:
        lower_bound = lowest_passable_bound
        logger.warning(
            "Could not find a lower bound for component size filtering, will "
            "set the lower bound to " + str(lowest_passable_bound),
        )
    return (
        component_sizes.filter(pl.col("n_umi") >= lower_bound)["component"],
        lower_bound,
    )


def filter_components_by_size_hard_thresholds(
    component_sizes: pl.DataFrame,
    lower_bound: int | None,
    higher_bound: int | None,
) -> pl.Series:
    """Filter components by size using hard thresholds.

    Args:
        component_sizes: DataFrame with columns `component` and `n_umi`.
        lower_bound: The lower bound for the component size.
        higher_bound: The higher bound for the component size.

    Returns:
        The `component` column for components that pass the filter.
    """
    if lower_bound is None:
        lower_bound = 0
    if higher_bound is None:
        higher_bound = np.iinfo(np.uint64).max
    return component_sizes.filter(
        (pl.col("n_umi") >= lower_bound) & (pl.col("n_umi") <= higher_bound)
    )["component"]
