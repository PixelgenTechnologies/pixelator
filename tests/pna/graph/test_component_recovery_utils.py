"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

import polars as pl
import pytest

from pixelator.pna.graph.component_recovery_utils import (
    absorb_core1_layer,
    create_component_size_data_frame,
    filter_connected_components_by_size,
    get_count_statistics,
    name_components_with_umi_hashes,
    name_components_with_umi_hashes_from_parquet,
    peel_core1_nodes,
    write_hive_partitioned_edgelist_without_out_of_size_bound_components,
)
from pixelator.pna.graph.report import GraphStatistics


def _complete_bipartite(umi1_values, umi2_values):
    return [(u1, u2) for u1 in umi1_values for u2 in umi2_values]


def _write_edgelist(path: Path, edges: list[tuple[int, int]]) -> Path:
    pl.DataFrame(
        {
            "umi1": [e[0] for e in edges],
            "umi2": [e[1] for e in edges],
            "read_count": [1] * len(edges),
        }
    ).write_parquet(path)
    return path


def _write_component_edgelist(
    path: Path, edges: list[tuple[int, int]], component: int
) -> Path:
    pl.DataFrame(
        {
            "umi1": [e[0] for e in edges],
            "umi2": [e[1] for e in edges],
            "read_count": [1] * len(edges),
            "component": [component] * len(edges),
        }
    ).write_parquet(path)
    return path


def test_get_count_statistics(tmp_path: Path) -> None:
    """Test correct edge, read, molecule, and distinct-UMI counts from a Parquet edgelist.

    Args:
        tmp_path: Tmp path.
    """
    path = tmp_path / "edgelist.parquet"
    pl.DataFrame(
        {
            "umi1": ["a", "b", "a"],
            "umi2": ["d", "c", "d"],
            "read_count": [10, 20, 30],
            "uei_count": [1, 2, 3],
        }
    ).write_parquet(path)

    stats = get_count_statistics(path)

    assert stats == {
        "n_edges": 3,
        "n_reads": 60,
        "n_molecules": 6,
        "n_umi": 4,
    }


def test_get_count_statistics_without_uei_count(tmp_path: Path) -> None:
    """When uei_count is absent, molecule count defaults to edge count."""
    path = tmp_path / "edgelist.parquet"
    pl.DataFrame(
        {
            "umi1": ["a", "b", "a"],
            "umi2": ["d", "c", "d"],
            "read_count": [10, 20, 30],
        }
    ).write_parquet(path)

    stats = get_count_statistics(path)

    assert stats == {
        "n_edges": 3,
        "n_reads": 60,
        "n_molecules": 3,
        "n_umi": 4,
    }


def test_write_hive_partitioned_edgelist_without_out_of_size_bound_components_prunes(
    tmp_path: Path,
) -> None:
    """Test components below the UMI score threshold are omitted and listed as discarded.

    Args:
        tmp_path: Tmp path.
    """
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["keep", "keep", "keep", "drop"],
            "umi1": ["a", "c", "e", "x"],
            "umi2": ["b", "d", "f", "y"],
        }
    ).write_parquet(partitioned)

    out_path, discarded = (
        write_hive_partitioned_edgelist_without_out_of_size_bound_components(
            input_edgelist_path=partitioned,
            min_component_size_to_prune=3,
            working_dir=tmp_path,
        )
    )

    assert out_path == tmp_path / "hive_partitioned_edgelist.parquet"
    kept = pl.scan_parquet(out_path, hive_schema={"component": pl.String}).collect()
    assert kept["component"].unique().to_list() == ["keep"]
    assert kept.height == 3

    discarded_sorted = discarded.sort("component")
    assert discarded_sorted["component"].to_list() == ["drop"]
    assert discarded_sorted["n_umi"].to_list() == [2]
    assert discarded_sorted.schema["n_umi"] == pl.UInt32
    assert discarded_sorted.schema["n_edges"] == pl.UInt32


def test_write_hive_partitioned_edgelist_without_out_of_size_bound_components_prunes_large(
    tmp_path: Path,
) -> None:
    """Components above the max UMI threshold are omitted and listed as discarded."""
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["at_max", "at_max", "over_max", "over_max"],
            "umi1": ["a", "a", "w", "y"],
            "umi2": ["b", "c", "x", "z"],
        }
    ).write_parquet(partitioned)

    out_path, discarded = (
        write_hive_partitioned_edgelist_without_out_of_size_bound_components(
            input_edgelist_path=partitioned,
            min_component_size_to_prune=0,
            max_component_size_to_prune=3,
            working_dir=tmp_path,
        )
    )

    kept = pl.scan_parquet(out_path, hive_schema={"component": pl.String}).collect()
    assert kept["component"].unique().to_list() == ["at_max"]
    assert kept.height == 2

    discarded_sorted = discarded.sort("component")
    assert discarded_sorted["component"].to_list() == ["over_max"]
    assert discarded_sorted["n_umi"].to_list() == [4]


def test_write_hive_partitioned_edgelist_without_out_of_size_bound_components_prunes_both_directions(
    tmp_path: Path,
) -> None:
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["too_small", "keep", "keep", "too_large", "too_large"],
            "umi1": ["a", "c", "c", "w", "y"],
            "umi2": ["b", "d", "e", "x", "z"],
        }
    ).write_parquet(partitioned)

    out_path, discarded = (
        write_hive_partitioned_edgelist_without_out_of_size_bound_components(
            input_edgelist_path=partitioned,
            min_component_size_to_prune=3,
            max_component_size_to_prune=3,
            working_dir=tmp_path,
        )
    )

    kept = pl.scan_parquet(out_path, hive_schema={"component": pl.String}).collect()
    assert kept["component"].unique().to_list() == ["keep"]
    assert kept.height == 2

    discarded_sorted = discarded.sort("component")
    assert discarded_sorted["component"].to_list() == ["too_large", "too_small"]
    assert discarded_sorted["n_umi"].to_list() == [4, 2]


def test_write_hive_partitioned_edgelist_without_out_of_size_bound_components_nothing_discarded(
    tmp_path: Path,
) -> None:
    """When every component meets the threshold, discarded frame is empty and all rows are kept.

    Args:
        tmp_path: Tmp path.
    """
    partitioned = tmp_path / "partitioned_edgelist.parquet"
    pl.DataFrame(
        {
            "component": ["a", "a", "b"],
            "umi1": ["u1", "u3", "w1"],
            "umi2": ["u2", "u4", "w2"],
        }
    ).write_parquet(partitioned)

    _, discarded = write_hive_partitioned_edgelist_without_out_of_size_bound_components(
        input_edgelist_path=partitioned,
        min_component_size_to_prune=2,
        working_dir=tmp_path,
    )

    assert discarded.height == 0
    kept = pl.scan_parquet(
        tmp_path / "hive_partitioned_edgelist.parquet",
        hive_schema={"component": pl.String},
    ).collect()
    assert kept.height == 3


def test_filter_connected_components_by_size_hard_thresholds(tmp_path: Path) -> None:
    """Hard thresholds keep only components within the configured UMI-size bounds."""
    input_path = tmp_path / "component_filter_input.parquet"
    input_frame = pl.DataFrame(
        {
            "component": ["a", "a", "b", "c", "c"],
            "umi1": ["u1", "u3", "v1", "w1", "w1"],
            "umi2": ["u2", "u4", "v2", "w2", "w3"],
        }
    )
    input_frame.write_parquet(input_path)
    discard_sizes = pl.DataFrame(
        schema={"component": pl.String, "n_umi": pl.UInt32},
    )
    component_stats = GraphStatistics()

    filtered_edgelist_path, stats = filter_connected_components_by_size(
        input_edgelist_path=input_path,
        component_size_threshold=(3, 4),
        discard_sizes=discard_sizes,
        component_stats=component_stats,
        working_dir=tmp_path,
    )

    filtered = pl.scan_parquet(
        filtered_edgelist_path, hive_schema={"component": pl.String}
    ).collect()

    assert set(filtered["component"].unique().to_list()) == {"a", "c"}
    assert filtered.height == 4
    assert stats.component_count_pre_component_size_filtering == 3
    assert stats.component_count_post_component_size_filtering == 2
    assert stats.component_size_min_filtering_threshold == 3
    assert stats.component_size_max_filtering_threshold == 4
    assert stats.pre_filtering_component_sizes == {2: 1, 3: 1, 4: 1}


def test_filter_connected_components_by_size_includes_early_discards_in_pre_filtering_stats(
    tmp_path: Path,
) -> None:
    """Components discarded early (e.g. for being too large) still count in the "pre filtering" stats."""
    input_path = tmp_path / "component_filter_input.parquet"
    pl.DataFrame(
        {
            "component": ["a", "a", "b", "c", "c"],
            "umi1": ["u1", "u3", "v1", "w1", "w1"],
            "umi2": ["u2", "u4", "v2", "w2", "w3"],
        }
    ).write_parquet(input_path)
    discard_sizes = pl.DataFrame(
        {
            "component": ["huge"],
            "n_umi": pl.Series([10], dtype=pl.UInt32),
        }
    )
    component_stats = GraphStatistics()

    filtered_edgelist_path, stats = filter_connected_components_by_size(
        input_edgelist_path=input_path,
        component_size_threshold=(3, 4),
        discard_sizes=discard_sizes,
        component_stats=component_stats,
        working_dir=tmp_path,
    )

    filtered = pl.scan_parquet(
        filtered_edgelist_path, hive_schema={"component": pl.String}
    ).collect()

    assert set(filtered["component"].unique().to_list()) == {"a", "c"}
    assert stats.component_count_pre_component_size_filtering == 4
    assert stats.component_count_post_component_size_filtering == 2
    assert stats.pre_filtering_component_sizes == {2: 1, 3: 1, 4: 1, 10: 1}
    fraction_of_discarded_components = (
        1
        - stats.component_count_post_component_size_filtering
        / stats.component_count_pre_component_size_filtering
    )
    assert fraction_of_discarded_components == pytest.approx(1 - 2 / 4)


def test_filter_connected_components_by_size_accepts_discard_sizes_with_n_edges(
    tmp_path: Path,
) -> None:
    """discard_sizes as returned by write_hive_partitioned_edgelist_without_out_of_size_bound_components carries an n_edges column; it must not break the concat with the (component, n_umi) sizes frame."""
    input_path = tmp_path / "component_filter_input.parquet"
    pl.DataFrame(
        {
            "component": ["a", "a", "b", "c", "c"],
            "umi1": ["u1", "u3", "v1", "w1", "w1"],
            "umi2": ["u2", "u4", "v2", "w2", "w3"],
        }
    ).write_parquet(input_path)
    discard_sizes = pl.DataFrame(
        {
            "component": ["huge"],
            "n_umi": pl.Series([10], dtype=pl.UInt32),
            "n_edges": pl.Series([20], dtype=pl.UInt32),
        }
    )
    component_stats = GraphStatistics()

    filtered_edgelist_path, stats = filter_connected_components_by_size(
        input_edgelist_path=input_path,
        component_size_threshold=(3, 4),
        discard_sizes=discard_sizes,
        component_stats=component_stats,
        working_dir=tmp_path,
    )

    filtered = pl.scan_parquet(
        filtered_edgelist_path, hive_schema={"component": pl.String}
    ).collect()

    assert set(filtered["component"].unique().to_list()) == {"a", "c"}
    assert stats.component_count_pre_component_size_filtering == 4
    assert stats.pre_filtering_component_sizes == {2: 1, 3: 1, 4: 1, 10: 1}


def test_create_component_size_data_frame_computes_sizes_per_component(
    tmp_path: Path,
) -> None:
    """Component sizes are computed from distinct umi1 + umi2 counts per component."""
    input_path = tmp_path / "component_sizes_input.parquet"
    pl.DataFrame(
        {
            "component": ["a", "a", "a", "b"],
            "umi1": ["u1", "u1", "u2", "v1"],
            "umi2": ["x1", "x2", "x2", "y1"],
        }
    ).write_parquet(input_path)

    combined = create_component_size_data_frame(input_path)
    combined_sorted = combined.sort("component")

    assert combined_sorted["component"].to_list() == ["a", "b"]
    assert combined_sorted["n_umi"].to_list() == [4, 2]


def test_create_component_size_data_frame_handles_empty_input(tmp_path: Path) -> None:
    """An empty edgelist produces an empty component-size dataframe."""
    input_path = tmp_path / "component_sizes_empty.parquet"
    pl.DataFrame(
        {
            "component": pl.Series([], dtype=pl.String),
            "umi1": pl.Series([], dtype=pl.String),
            "umi2": pl.Series([], dtype=pl.String),
        }
    ).write_parquet(input_path)

    combined = create_component_size_data_frame(input_path)

    assert combined.is_empty()
    assert combined.columns == ["component", "n_umi"]


def test_name_components_with_umi_hashes_same_umi_set_same_hash() -> None:
    """Components with the same UMI set get the expected deterministic hash."""
    edgelist = pl.DataFrame(
        {
            "component": ["c1", "c1", "c2", "c2", "c3"],
            "orig_component": ["c1", "c1", "c2", "c2", "c3"],
            "umi1": [1, 1, 2, 1, 9],
            "umi2": [2, 2, 1, 2, 10],
        }
    ).lazy()

    hashed = name_components_with_umi_hashes(edgelist).collect()

    c1_hash = (
        hashed.filter(pl.col("orig_component") == "c1")
        .select(pl.col("component").unique())
        .item()
    )
    c2_hash = (
        hashed.filter(pl.col("orig_component") == "c2")
        .select(pl.col("component").unique())
        .item()
    )
    c3_hash = (
        hashed.filter(pl.col("orig_component") == "c3")
        .select(pl.col("component").unique())
        .item()
    )

    assert c1_hash == "07ee86c281446bef"
    assert c2_hash == "07ee86c281446bef"
    assert c3_hash == "cc8f7e82d8a2a85e"


def test_name_components_with_umi_hashes_deterministic_across_row_order() -> None:
    """Hash assignment is stable even when row order changes."""
    base = pl.DataFrame(
        {
            "component": ["x", "x", "y"],
            "orig_component": ["x", "x", "y"],
            "umi1": [1, 3, 10],
            "umi2": [2, 4, 11],
        }
    )
    reversed_rows = base.reverse()

    hashed_base = name_components_with_umi_hashes(base.lazy()).collect()
    hashed_reversed = name_components_with_umi_hashes(reversed_rows.lazy()).collect()

    base_map = (
        hashed_base.group_by("orig_component")
        .agg(pl.col("component").first().alias("hash"))
        .sort("orig_component")
    )
    reversed_map = (
        hashed_reversed.group_by("orig_component")
        .agg(pl.col("component").first().alias("hash"))
        .sort("orig_component")
    )

    assert base_map["hash"].to_list() == reversed_map["hash"].to_list()


def test_name_components_with_umi_hashes_from_parquet_same_umi_set_same_hash(
    tmp_path: Path,
) -> None:
    """Parquet helper computes the same deterministic hashes as lazy-frame helper."""
    input_path = tmp_path / "input.parquet"
    pl.DataFrame(
        {
            "component": ["c1", "c1", "c2", "c2", "c3"],
            "orig_component": ["c1", "c1", "c2", "c2", "c3"],
            "umi1": [1, 1, 2, 1, 9],
            "umi2": [2, 2, 1, 2, 10],
        }
    ).write_parquet(input_path)

    output_path = name_components_with_umi_hashes_from_parquet(input_path, tmp_path)
    hashed = pl.read_parquet(output_path)

    c1_hash = (
        hashed.filter(pl.col("orig_component") == "c1")
        .select(pl.col("component").unique())
        .item()
    )
    c2_hash = (
        hashed.filter(pl.col("orig_component") == "c2")
        .select(pl.col("component").unique())
        .item()
    )
    c3_hash = (
        hashed.filter(pl.col("orig_component") == "c3")
        .select(pl.col("component").unique())
        .item()
    )

    assert c1_hash == "07ee86c281446bef"
    assert c2_hash == "07ee86c281446bef"
    assert c3_hash == "cc8f7e82d8a2a85e"


def test_name_components_with_umi_hashes_from_parquet_deterministic_across_row_order(
    tmp_path: Path,
) -> None:
    """Parquet helper hash assignment is stable across input row order."""
    base = pl.DataFrame(
        {
            "component": ["x", "x", "y"],
            "orig_component": ["x", "x", "y"],
            "umi1": [1, 3, 10],
            "umi2": [2, 4, 11],
        }
    )
    base_path = tmp_path / "base.parquet"
    reversed_path = tmp_path / "reversed.parquet"
    base.write_parquet(base_path)
    base.reverse().write_parquet(reversed_path)

    hashed_base = pl.read_parquet(
        name_components_with_umi_hashes_from_parquet(base_path, tmp_path)
    )
    hashed_reversed = pl.read_parquet(
        name_components_with_umi_hashes_from_parquet(reversed_path, tmp_path)
    )

    base_map = (
        hashed_base.group_by("orig_component")
        .agg(pl.col("component").first().alias("hash"))
        .sort("orig_component")
    )
    reversed_map = (
        hashed_reversed.group_by("orig_component")
        .agg(pl.col("component").first().alias("hash"))
        .sort("orig_component")
    )

    assert base_map["hash"].to_list() == reversed_map["hash"].to_list()


CLUSTER_UMI1 = list(range(1000, 1006))
CLUSTER_UMI2 = list(range(2000, 2006))


def test_peel_core1_nodes_keeps_fully_connected_graph_untouched(tmp_path: Path) -> None:
    """Every UMI in a complete 6x6 bipartite graph has degree 6 -- nothing to peel."""
    edges = _complete_bipartite(CLUSTER_UMI1, CLUSTER_UMI2)
    edgelist_path = _write_edgelist(tmp_path / "input.parquet", edges)

    stats = GraphStatistics()
    kept_path, discard_path, stats = peel_core1_nodes(
        input_edgelist_path=edgelist_path,
        working_dir=tmp_path,
        stats=stats,
    )

    assert stats.core1_peel_iterations_run == 0
    assert stats.core1_layer_edges == 0
    assert stats.edges_post_core1_peel == len(edges)
    assert pl.read_parquet(kept_path).height == len(edges)
    assert pl.read_parquet(discard_path).height == 0


def test_peel_core1_nodes_peels_a_single_leaf_in_one_round(tmp_path: Path) -> None:
    """umi1=9999 has exactly one edge -- a literal degree-1 UMI, peeled in the first round."""
    edges = _complete_bipartite(CLUSTER_UMI1, CLUSTER_UMI2) + [(9999, 2000)]
    edgelist_path = _write_edgelist(tmp_path / "input.parquet", edges)

    stats = GraphStatistics()
    kept_path, discard_path, stats = peel_core1_nodes(
        input_edgelist_path=edgelist_path,
        working_dir=tmp_path,
        stats=stats,
    )

    kept = pl.read_parquet(kept_path)
    discarded = pl.read_parquet(discard_path)
    assert stats.core1_peel_iterations_run == 1
    assert 9999 not in kept["umi1"].to_list()
    assert discarded.height == 1
    assert discarded["umi1"].to_list() == [9999]


def test_peel_core1_nodes_peels_a_chain_over_multiple_rounds(tmp_path: Path) -> None:
    """A length-5 chain hangs off the cluster: unrolling it takes 5 rounds, one hop at a time."""
    core_edges = _complete_bipartite(CLUSTER_UMI1, CLUSTER_UMI2)
    chain_edges = [(9003, 2000), (9003, 8002), (9002, 8002), (9002, 8001), (9001, 8001)]
    edgelist_path = _write_edgelist(
        tmp_path / "input.parquet", core_edges + chain_edges
    )

    stats = GraphStatistics()
    kept_path, discard_path, stats = peel_core1_nodes(
        input_edgelist_path=edgelist_path,
        working_dir=tmp_path,
        stats=stats,
    )

    kept = pl.read_parquet(kept_path)
    discarded = pl.read_parquet(discard_path)
    assert stats.core1_peel_iterations_run == 5
    assert discarded.height == len(chain_edges)
    assert kept.height == len(core_edges)
    for chain_umi in (9001, 9002, 9003):
        assert chain_umi not in kept["umi1"].to_list()


def test_peel_core1_nodes_respects_max_iterations(tmp_path: Path) -> None:
    """Capping ``max_iterations`` stops peeling before the full 2-core is reached."""
    core_edges = _complete_bipartite(CLUSTER_UMI1, CLUSTER_UMI2)
    chain_edges = [(9003, 2000), (9003, 8002), (9002, 8002), (9002, 8001), (9001, 8001)]
    edgelist_path = _write_edgelist(
        tmp_path / "input.parquet", core_edges + chain_edges
    )

    stats = GraphStatistics()
    kept_path, discard_path, stats = peel_core1_nodes(
        input_edgelist_path=edgelist_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=2,
    )

    kept = pl.read_parquet(kept_path)
    assert stats.core1_peel_iterations_run == 2
    assert stats.core1_layer_edges == 2
    assert stats.core1_layer_edges < len(chain_edges)
    # 9001 (round 1) and 8001 (round 2) are peeled, but 9002 -- which only becomes a leaf in
    # round 3 -- is not reached yet.
    assert 9001 not in kept["umi1"].to_list()
    assert 8001 not in kept["umi2"].to_list()
    assert 9002 in kept["umi1"].to_list()


def test_absorb_core1_layer_rescues_frontier_edge_and_discards_conflict(
    tmp_path: Path,
) -> None:
    """A frontier edge is rescued into its known component; a conflict edge is dropped."""
    base_edges = [(1, 101), (2, 102), (3, 103), (4, 104), (5, 105), (6, 106)]
    base_path = tmp_path / "base.parquet"
    pl.concat(
        [
            pl.DataFrame(
                {
                    "umi1": [e[0] for e in base_edges[:3]],
                    "umi2": [e[1] for e in base_edges[:3]],
                    "read_count": [1, 1, 1],
                    "component": [1, 1, 1],
                }
            ),
            pl.DataFrame(
                {
                    "umi1": [e[0] for e in base_edges[3:]],
                    "umi2": [e[1] for e in base_edges[3:]],
                    "read_count": [1, 1, 1],
                    "component": [2, 2, 2],
                }
            ),
        ]
    ).write_parquet(base_path)

    # (1, 900): frontier -- umi1=1 is known (component 1) -- must be rescued.
    # (4, 101): conflict -- umi1=4 is known (component 2), umi2=101 is known (component 1).
    discard_edges = [(1, 900), (4, 101)]
    discard_path = _write_edgelist(tmp_path / "discard.parquet", discard_edges)

    stats = GraphStatistics()
    final_path, stats = absorb_core1_layer(
        base_edgelist_path=base_path,
        core1_discard_path=discard_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=5,
    )

    final = pl.read_parquet(final_path)
    rescued = final.filter(pl.col("umi1") == 1, pl.col("umi2") == 900)
    assert rescued.height == 1
    assert rescued["component"].item() == 1
    assert final.filter(pl.col("umi1") == 4, pl.col("umi2") == 101).height == 0

    assert stats.core1_absorption_iterations_run == 1
    assert stats.core1_edges_reabsorbed == 1
    assert stats.core1_edges_discarded == 1


def test_absorb_core1_layer_treats_multiple_claims_as_fused(tmp_path: Path) -> None:
    """An unmapped UMI claimed by two different components in the same round is discarded."""
    base_path = tmp_path / "base.parquet"
    pl.concat(
        [
            _dataframe_for_component([(1, 101)], component=1),
            _dataframe_for_component([(2, 102)], component=2),
        ]
    ).write_parquet(base_path)

    # umi2=999 gets frontier edges from both component 1 (via umi1=1) and component 2 (via
    # umi1=2) in the same round -- ambiguous, so both edges are discarded as fused.
    discard_edges = [(1, 999), (2, 999)]
    discard_path = _write_edgelist(tmp_path / "discard.parquet", discard_edges)

    stats = GraphStatistics()
    final_path, stats = absorb_core1_layer(
        base_edgelist_path=base_path,
        core1_discard_path=discard_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=5,
    )

    final = pl.read_parquet(final_path)
    assert final.filter(pl.col("umi2") == 999).height == 0
    assert stats.core1_edges_reabsorbed == 0
    assert stats.core1_edges_discarded == 2


def test_absorb_core1_layer_propagates_evidence_across_iterations(
    tmp_path: Path,
) -> None:
    """A chain of core-1 edges is resolved one hop per iteration."""
    base_path = _write_component_edgelist(
        tmp_path / "base.parquet", [(1, 101)], component=1
    )

    # umi1=101 is known from round 0. Each subsequent edge only becomes a frontier edge once
    # its predecessor has been rescued, so a 3-edge chain needs 3 rounds to fully resolve.
    discard_edges = [(101, 900), (900, 901), (901, 902)]
    discard_path = _write_edgelist(tmp_path / "discard.parquet", discard_edges)

    stats = GraphStatistics()
    final_path, stats = absorb_core1_layer(
        base_edgelist_path=base_path,
        core1_discard_path=discard_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=5,
    )

    final = pl.read_parquet(final_path)
    assert final.height == 1 + len(discard_edges)
    assert (final["component"] == 1).all()
    assert stats.core1_absorption_iterations_run == 3
    assert stats.core1_edges_reabsorbed == len(discard_edges)
    assert stats.core1_edges_discarded == 0


def test_absorb_core1_layer_respects_max_iterations(tmp_path: Path) -> None:
    """Capping ``max_iterations`` leaves edges beyond the reachable hop count unresolved."""
    base_path = _write_component_edgelist(
        tmp_path / "base.parquet", [(1, 101)], component=1
    )
    discard_edges = [(101, 900), (900, 901), (901, 902)]
    discard_path = _write_edgelist(tmp_path / "discard.parquet", discard_edges)

    stats = GraphStatistics()
    final_path, stats = absorb_core1_layer(
        base_edgelist_path=base_path,
        core1_discard_path=discard_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=2,
    )

    final = pl.read_parquet(final_path)
    assert stats.core1_absorption_iterations_run == 2
    assert stats.core1_edges_reabsorbed == 2
    assert stats.core1_edges_discarded == 1
    assert final.filter(pl.col("umi1") == 901, pl.col("umi2") == 902).height == 0


def test_absorb_core1_layer_handles_empty_discard_pile(tmp_path: Path) -> None:
    """An empty core-1 discard pile leaves the base edgelist untouched."""
    base_path = _write_component_edgelist(
        tmp_path / "base.parquet", [(1, 101)], component=1
    )
    empty_discard_path = tmp_path / "discard.parquet"
    pl.DataFrame(
        {
            "umi1": pl.Series([], dtype=pl.Int64),
            "umi2": pl.Series([], dtype=pl.Int64),
            "read_count": pl.Series([], dtype=pl.Int64),
        }
    ).write_parquet(empty_discard_path)

    stats = GraphStatistics()
    final_path, stats = absorb_core1_layer(
        base_edgelist_path=base_path,
        core1_discard_path=empty_discard_path,
        working_dir=tmp_path,
        stats=stats,
        max_iterations=5,
    )

    final = pl.read_parquet(final_path)
    assert final.height == 1
    assert stats.core1_absorption_iterations_run == 0
    assert stats.core1_edges_reabsorbed == 0
    assert stats.core1_edges_discarded == 0


def _dataframe_for_component(
    edges: list[tuple[int, int]], component: int
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "umi1": [e[0] for e in edges],
            "umi2": [e[1] for e in edges],
            "read_count": [1] * len(edges),
            "component": [component] * len(edges),
        }
    )
