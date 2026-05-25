"""Integration tests for the PNA component recovery pipeline.

Copyright © 2025 Pixelgen Technologies AB.
"""

import random
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pixelator.pna.graph.community_detection import (
    RefinementOptions,
    StagedRefinementOptions,
    find_components,
)
from pixelator.pna.graph.component_recovery_utils import ConnectedComponentException


@pytest.fixture
def random_graph_path():
    """Random graph path."""
    random.seed(0)

    n_clusters = 10
    n_nodes_per_cluster = 100
    n_edges_per_cluster = 800
    n_crossing_edges = int(0.004 * n_edges_per_cluster * n_clusters)

    edges = []
    for cluster in range(n_clusters):
        # To avoid clashing umis, we need to make sure there is no intersection between umi1 and umi2
        # This is done by always using even values for umi1 and odd values for umi2
        min_node_id = n_nodes_per_cluster * cluster
        max_node_id = n_nodes_per_cluster * (cluster + 1)
        edges.extend(
            [
                (
                    (umi1 := random.randrange(min_node_id // 2, max_node_id // 2) * 2),
                    (
                        umi2 := random.randrange(min_node_id // 2, max_node_id // 2) * 2
                        + 1
                    ),
                    str(cluster),
                    random.randrange(1_000, 10_000),
                    random.randrange(1_000, 10_000),
                    str(umi1),
                    str(umi2),
                )
                for _ in range(n_edges_per_cluster)
            ]
        )

    # Add an 11th cluster with a larger number of nodes (200)
    large_cluster = n_clusters
    min_node_id = n_nodes_per_cluster * n_clusters
    max_node_id = min_node_id + 200
    edges.extend(
        [
            (
                (umi1 := random.randrange(min_node_id // 2, max_node_id // 2) * 2),
                (umi2 := random.randrange(min_node_id // 2, max_node_id // 2) * 2 + 1),
                str(large_cluster),
                random.randrange(1_000, 10_000),
                random.randrange(1_000, 10_000),
                str(umi1),
                str(umi2),
            )
            for _ in range(2 * n_edges_per_cluster)
        ]
    )

    umi1s = list(set([edge[0] for edge in edges]))
    umi2s = list(set([edge[1] for edge in edges]))

    edges.extend(
        [
            (
                (umi1 := random.choice(umi1s)),
                (umi2 := random.choice(umi2s)),
                "crossing",
                random.randrange(1_000, 10_000),
                random.randrange(1_000, 10_000),
                str(umi1),
                str(umi2),
            )
            for _ in range(n_crossing_edges)
        ]
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet") as temp_file:
        pl.DataFrame(
            edges,
            schema={
                "umi1": pl.UInt64,
                "umi2": pl.UInt64,
                "component": pl.String,
                "read_count": pl.UInt32,
                "uei_count": pl.UInt16,
                "marker_1": pl.String,
                "marker_2": pl.String,
            },
            orient="row",
        ).write_parquet(temp_file.name)
        yield temp_file.name


@pytest.mark.parametrize("edge_cycle_verification", [False, True])
def test_find_components_small(
    random_graph_path,
    edge_cycle_verification,
):
    """Verify find components small.

    Args:
    random_graph_path: random graph path.
    edge_cycle_verification: edge cycle verification.

    """
    ground_truth_umi_map = (
        (
            pl.read_parquet(random_graph_path)
            .group_by("component")
            .agg(pl.col("umi1").append(pl.col("umi2")).unique().alias("umi"))
        )
        .explode("umi")
        .filter(pl.col("component") != "crossing")
    )
    true_sizes = ground_truth_umi_map.group_by("component").agg(
        pl.len().alias("true_size")
    )

    staged_refinement_options = StagedRefinementOptions(
        initial_stage_options=RefinementOptions(
            leiden_resolution=1.0,
            min_component_size_to_prune=50,
        ),
        refinement_stage_options=RefinementOptions(
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        component_stats, resolved_edgelist_path = find_components(
            input_edgelist_path=Path(random_graph_path),
            working_dir=working_dir,
            multiplet_recovery=True,
            edge_cycle_verification=edge_cycle_verification,
            min_read_count=1,
            component_size_threshold=(10, 1_000_000),
            n_threads=10,
            refinement_options=staged_refinement_options,
        )
        resolved_edgelist = pl.read_parquet(resolved_edgelist_path)
        umi_map = (
            resolved_edgelist.group_by("component")
            .agg(pl.col("umi1").append(pl.col("umi2")).unique().alias("umi"))
            .rename({"component": "component_recovered"})
        ).explode("umi")

        matched_map = ground_truth_umi_map.join(
            umi_map, on="umi", how="inner", suffix="_recovered"
        )
        best_matches = (
            matched_map.group_by(["component", "component_recovered"])
            .agg(pl.len().alias("intersection_size"))
            .filter(
                pl.col("intersection_size")
                == pl.col("intersection_size").max().over("component")
            )
        )
        best_matches = best_matches.join(true_sizes, on="component", how="left").join(
            umi_map.group_by("component_recovered").agg(
                pl.len().alias("recovered_size")
            ),
            on="component_recovered",
            how="left",
        )

        assert best_matches["component_recovered"].unique().shape == (11,)
        assert (
            (best_matches["true_size"] == best_matches["recovered_size"])
            & (best_matches["recovered_size"] == best_matches["intersection_size"])
        ).all()

        assert component_stats.component_count_pre_recovery == 1
        assert component_stats.component_count_post_recovery == 11

        assert component_stats.crossing_edges_removed_initial_stage == 29
        assert component_stats.crossing_edges_removed == 29
        assert component_stats.edge_count_pre_recovery == 9632
        assert component_stats.edge_count_post_recovery == 9603
        assert component_stats.node_count_pre_recovery == 1200
        assert component_stats.node_count_post_recovery == 1200
        assert (
            sum(
                size * count
                for size, count in component_stats.post_flp_community_sizes.items()
            )
            == 1200
        )


def test_find_no_components(random_graph_path):
    """Verify find no components.

    Args:
    random_graph_path: random graph path.

    """
    staged_refinement_options = StagedRefinementOptions(
        initial_stage_options=RefinementOptions(
            leiden_resolution=1.0,
            min_component_size_to_prune=300,
        ),
        refinement_stage_options=RefinementOptions(
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        with pytest.raises(ConnectedComponentException):
            component_stats, resolved_edgelist_path = find_components(
                input_edgelist_path=Path(random_graph_path),
                working_dir=working_dir,
                multiplet_recovery=True,
                edge_cycle_verification=False,
                min_read_count=1,
                component_size_threshold=(10, 1_000_000),
                n_threads=10,
                refinement_options=staged_refinement_options,
            )


def test_find_components_prunes_small_clusters_leaving_one_large(random_graph_path):
    """Verify find components prunes small clusters leaving one large.

    Args:
    random_graph_path: random graph path.

    """
    staged_refinement_options = StagedRefinementOptions(
        initial_stage_options=RefinementOptions(
            leiden_resolution=1.0,
            min_component_size_to_prune=150,
        ),
        refinement_stage_options=RefinementOptions(
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        component_stats, resolved_edgelist_path = find_components(
            input_edgelist_path=Path(random_graph_path),
            working_dir=working_dir,
            multiplet_recovery=True,
            edge_cycle_verification=False,
            min_read_count=1,
            component_size_threshold=(10, 1_000_000),
            n_threads=10,
            refinement_options=staged_refinement_options,
        )

    # With a pruning threshold of 150 nodes, only the 200-node cluster
    # should remain after component refinement.
    assert component_stats.component_count_post_recovery == 1


def test_find_components_empty_parquet_file():
    """Verify find components empty parquet file."""
    staged_refinement_options = StagedRefinementOptions(
        initial_stage_options=RefinementOptions(
            leiden_resolution=1.0,
            min_component_size_to_prune=50,
        ),
        refinement_stage_options=RefinementOptions(
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )
    empty_schema = {
        "umi1": pl.UInt64,
        "umi2": pl.UInt64,
        "component": pl.String,
        "read_count": pl.UInt32,
        "uei_count": pl.UInt16,
        "marker_1": pl.String,
        "marker_2": pl.String,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_path = Path(temp_dir) / "empty_edgelist.parquet"
        pl.DataFrame(schema=empty_schema).write_parquet(empty_path)
        working_dir = Path(temp_dir) / "work"
        working_dir.mkdir()
        with pytest.raises(ConnectedComponentException):
            find_components(
                input_edgelist_path=empty_path,
                working_dir=working_dir,
                multiplet_recovery=True,
                edge_cycle_verification=False,
                min_read_count=1,
                component_size_threshold=(10, 1_000_000),
                n_threads=1,
                refinement_options=staged_refinement_options,
            )


@pytest.mark.slow
@pytest.mark.parametrize("edge_cycle_verification", [False, True])
def test_find_components_big(
    testdata_3pc_crossing_parquet,
    testdata_0pc_crossing_parquet,
    edge_cycle_verification,
):
    """Verify find components big.

    Args:
    testdata_3pc_crossing_parquet: testdata 3pc crossing parquet.
    testdata_0pc_crossing_parquet: testdata 0pc crossing parquet.
    edge_cycle_verification: edge cycle verification.

    """
    ground_truth_umi_map = (
        pl.read_parquet(testdata_0pc_crossing_parquet)
        .group_by("component")
        .agg(pl.col("umi1").append(pl.col("umi2")).unique().alias("umi"))
    ).explode("umi")

    true_sizes = ground_truth_umi_map.group_by("component").agg(
        pl.len().alias("true_size")
    )

    staged_refinement_options = StagedRefinementOptions(
        initial_stage_options=RefinementOptions(
            leiden_resolution=0.25,
        ),
        refinement_stage_options=RefinementOptions(
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        component_stats, resolved_edgelist_path = find_components(
            input_edgelist_path=Path(testdata_3pc_crossing_parquet),
            working_dir=working_dir,
            multiplet_recovery=True,
            edge_cycle_verification=edge_cycle_verification,
            min_read_count=1,
            component_size_threshold=(10, 1_000_000_000),
            n_threads=10,
            refinement_options=staged_refinement_options,
        )

        resolved_edgelist = pl.read_parquet(resolved_edgelist_path)
    umi_map = (
        resolved_edgelist.group_by("component")
        .agg(pl.col("umi1").append(pl.col("umi2")).unique().alias("umi"))
        .rename({"component": "component_recovered"})
    ).explode("umi")

    matched_map = ground_truth_umi_map.join(
        umi_map, on="umi", how="inner", suffix="_recovered"
    )
    best_matches = (
        matched_map.group_by(["component", "component_recovered"])
        .agg(pl.len().alias("intersection_size"))
        .filter(
            pl.col("intersection_size")
            == pl.col("intersection_size").max().over("component")
        )
    )
    best_matches = best_matches.join(true_sizes, on="component", how="left").join(
        umi_map.group_by("component_recovered").agg(pl.len().alias("recovered_size")),
        on="component_recovered",
        how="left",
    )

    best_matches = best_matches.with_columns(
        match_rate=(
            pl.col("intersection_size")
            / pl.min_horizontal(pl.col("true_size"), pl.col("recovered_size"))
        ),
        recover_rate=pl.col("intersection_size") / pl.col("true_size"),
    )

    assert best_matches["component_recovered"].unique().shape == (40,)
    assert best_matches.select("match_rate").mean()[0, 0] > 0.9
    assert best_matches.select("recover_rate").mean()[0, 0] > 0.8
