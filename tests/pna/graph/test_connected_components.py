"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import pytest
from polars.testing.asserts import assert_frame_equal
from scipy.spatial import distance_matrix

from pixelator.pna.graph.connected_components import (
    RefinementOptions,
    StagedRefinementOptions,
    build_pxl_file_with_components,
    filter_components_by_size_dynamic,
    filter_components_by_size_hard_thresholds,
    find_components,
    hash_component,
    make_edgelits_with_component_column,
    merge_communities_with_many_crossing_edges,
    recover_multiplets,
)
from pixelator.pna.pixeldataset import PNAPixelDataset


@pytest.mark.slow
def test_recover_multiplets():
    multiple_graph = nx.karate_club_graph()
    assert sum(1 for _ in nx.connected_components(multiple_graph)) == 1

    options = StagedRefinementOptions(
        inital_stage_options=RefinementOptions(
            min_component_size=10,
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
        ),
        max_component_refinement_depth=1,
    )
    edgelist = pl.LazyFrame(
        [{"umi1": e[0], "umi2": e[1]} for e in multiple_graph.edges()]
    )
    umi_component_map, stats = recover_multiplets(edgelist, refinement_options=options)
    edgelist = make_edgelits_with_component_column(edgelist, umi_component_map)
    split_graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).collect().rows())

    assert stats.crossing_edges_removed == 21
    assert stats.max_recursion_depth == 0

    # At this resolution the karate club graph has 4 communities, and after recovery these should
    # be their on connected components
    connected_components = list(nx.connected_components(split_graph))
    assert len(connected_components) == 4

    assert connected_components == [
        {0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21},
        {4, 5, 6, 10, 16},
        {32, 33, 8, 9, 14, 15, 18, 20, 22, 26, 29, 30},
        {23, 24, 25, 27, 28, 31},
    ]


def generate_cells(add_random_edges=True, add_small_pieces=False):
    nbr_of_cells = 20
    n_random_edges = 40
    rng = np.random.default_rng(0)

    def sample_spherical(n_points, n_dim=3):
        vec = rng.random((n_points, n_dim))
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    full_graph = nx.Graph()
    for _ in range(nbr_of_cells):
        points = sample_spherical(1000)
        distances = distance_matrix(points, points)
        graph = nx.from_numpy_array(
            distances < 0.01,
        )
        if add_small_pieces:
            points = sample_spherical(100)
            distances = distance_matrix(points, points)
            small_graph = nx.from_numpy_array(
                distances < 0.1,
            )
            graph = nx.disjoint_union(graph, small_graph)
            nodes = sorted(list(graph.nodes))
            # the disjoined graph labels the first graph first,
            # and the other graph second, hence we can pick the
            # nodes in this way.
            node_in_small_graph = nodes[-1]
            node_in_big_graph = nodes[0]
            graph.add_edge(node_in_big_graph, node_in_small_graph)

        full_graph = nx.disjoint_union(full_graph, graph)

    if add_random_edges:
        for _ in range(n_random_edges):
            node1 = rng.choice(full_graph.nodes())
            node2 = rng.choice(full_graph.nodes())
            full_graph.add_edge(node1, node2)

    return full_graph


@pytest.fixture(name="large_graph")
def large_graph_fixture():
    return generate_cells(add_random_edges=True, add_small_pieces=False)


@pytest.mark.slow
def test_recover_multiplets_large_graph(large_graph):
    # Check that we have a single "mega cluster" component before
    # starting
    assert sum(1 for _ in nx.connected_components(large_graph)) == 1

    options = StagedRefinementOptions(
        inital_stage_options=RefinementOptions(
            min_component_size=10,
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
        ),
        refinement_stage_options=RefinementOptions(
            min_component_size=10,
            max_edges_to_remove=5,
            max_edges_to_remove_relative=None,
        ),
        max_component_refinement_depth=4,
    )
    edgelist = pl.LazyFrame([{"umi1": e[0], "umi2": e[1]} for e in large_graph.edges()])

    umi_component_map, recovery_stats = recover_multiplets(
        edgelist, refinement_options=options
    )
    edgelist = make_edgelits_with_component_column(edgelist, umi_component_map)
    split_graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).collect().rows())
    assert recovery_stats.crossing_edges_removed == 36
    assert recovery_stats.crossing_edges_removed_in_initial_stage == 36
    assert recovery_stats.max_recursion_depth == 1

    # Check all components were resolved
    connected_components = list(nx.connected_components(split_graph))
    assert len(connected_components) == 20
    assert all(comp_size == 1000 for comp_size in map(len, connected_components))


@pytest.fixture(name="large_graph_with_smaller_attached")
def large_graph_with_smaller_things_added_on_fixture():
    return generate_cells(add_random_edges=True, add_small_pieces=True)


@pytest.mark.slow
def test_recover_multiplets_large_graph_with_small_stuff_added_on(
    large_graph_with_smaller_attached,
):
    large_graph = large_graph_with_smaller_attached
    # Check what we start from
    assert sum(1 for _ in nx.connected_components(large_graph)) == 2
    assert list(len(c) for c in nx.connected_components(large_graph)) == [21999, 1]
    options = StagedRefinementOptions(
        inital_stage_options=RefinementOptions(
            min_component_size=10,
            min_component_size_to_prune=50,
            max_edges_to_remove=None,
            # lowering the resolution to make sure something get
            # to process in the second stage
            leiden_resolution=0.01,
        ),
        refinement_stage_options=RefinementOptions(
            min_component_size=10,
            min_component_size_to_prune=50,
            max_edges_to_remove=5,
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=4,
    )

    edgelist = pl.LazyFrame(
        [{"umi1": e[0], "umi2": e[1]} for e in large_graph.edges() if e[0] != e[1]]
    )
    umi_component_map, recovery_stats = recover_multiplets(
        edgelist, refinement_options=options
    )
    edgelist = make_edgelits_with_component_column(edgelist, umi_component_map)
    split_graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).collect().rows())
    assert recovery_stats.crossing_edges_removed == 59
    assert recovery_stats.crossing_edges_removed_in_initial_stage == 58
    assert recovery_stats.max_recursion_depth == 2

    # Check all components were resolved
    connected_components = list(nx.connected_components(split_graph))
    assert len(connected_components) == 40


@pytest.mark.slow
def test_recover_multiplets_with_refinement_enabled():
    multiple_graph = nx.karate_club_graph()
    assert sum(1 for _ in nx.connected_components(multiple_graph)) == 1

    options = StagedRefinementOptions(
        inital_stage_options=RefinementOptions(
            min_component_size=3,
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=1.0,
        ),
        refinement_stage_options=RefinementOptions(
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=0.01,
        ),
        max_component_refinement_depth=3,
    )
    edgelist = pl.LazyFrame(
        [{"umi1": e[0], "umi2": e[1]} for e in multiple_graph.edges() if e[0] != e[1]]
    )

    umi_component_map, stats = recover_multiplets(edgelist, refinement_options=options)
    edgelist = make_edgelits_with_component_column(edgelist, umi_component_map)
    split_graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).collect().rows())
    assert stats.crossing_edges_removed == 21
    assert stats.max_recursion_depth == 1

    # At this resolution the karate club graph has 4 communities, and after recovery these should
    # be their on connected components
    connected_components = list(nx.connected_components(split_graph))
    assert len(connected_components) == 4

    assert connected_components == [
        {0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21},
        {4, 5, 6, 10, 16},
        {32, 33, 8, 9, 14, 15, 18, 20, 22, 26, 29, 30},
        {23, 24, 25, 27, 28, 31},
    ]


def test_filter_components_by_size_dynamic_if_no_limit_found_return_all():
    component_sizes = pl.DataFrame({"component": [0, 1, 2, 3], "n_umi": [3, 2, 4, 1]})
    filtered, threshold = filter_components_by_size_dynamic(
        component_sizes, lowest_passable_bound=-1
    )
    assert list(filtered) == component_sizes["component"].to_list()
    assert threshold == -1


def test_filter_components_by_size_dynamic():
    rng = np.random.default_rng(seed=0)
    cells_dist = rng.normal(10_000, 1000, 1000).astype(int)
    debris_dist = rng.normal(1000, 200, 1000).astype(int)
    debris_dist = debris_dist[debris_dist > 0]

    components = [set(range(cell_size)) for cell_size in cells_dist] + [
        set(range(debris_size)) for debris_size in debris_dist
    ]
    component_sizes = pl.DataFrame(
        {"component": range(len(components)), "n_umi": [len(c) for c in components]}
    )
    passing_components, threshold = filter_components_by_size_dynamic(
        component_sizes, lowest_passable_bound=0
    )
    # This number make sense, since it is close to the number of cells we generated
    assert len(passing_components) == 995
    assert threshold == 7071


def test_filter_components_by_size_hard_thresholds():
    components = [
        {1, 2, 3},
        {4, 5},
        {6, 7, 8, 9},
        {10},
    ]
    component_sizes = pl.DataFrame(
        {"component": range(len(components)), "n_umi": [len(c) for c in components]}
    )
    passing_components = filter_components_by_size_hard_thresholds(
        component_sizes, lower_bound=2, higher_bound=3
    )

    assert passing_components.to_list() == [0, 1]


def test_hash_component():
    component = {1, 2, 3}
    result = hash_component(component)
    assert result == "7c68b4906e7ea780"

    component_same_but_different_order = {3, 2, 1}
    result = hash_component(component_same_but_different_order)
    assert result == "7c68b4906e7ea780"


@pytest.fixture(name="lazy_edgelist_karate_graph")
def karate_edgelist(edgelist_karate_graph):
    edgelist = pl.DataFrame(edgelist_karate_graph).lazy()
    return edgelist


@pytest.mark.slow
def test_find_components(lazy_edgelist_karate_graph):
    edgelist = lazy_edgelist_karate_graph
    edgelist_with_components = find_components(
        edgelist,
        multiplet_recovery=True,
        min_read_count=1,
        component_size_threshold=(0, 100),
        refinement_options=StagedRefinementOptions(
            inital_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            refinement_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            max_component_refinement_depth=1,
        ),
    )

    edgelist = edgelist.collect()

    edgelist_with_components = edgelist_with_components.collect()
    # There are 10 crossing edges
    assert len(edgelist_with_components) == len(edgelist) - 10

    assert edgelist_with_components["component"].n_unique() == 4


def test_find_components_no_multiple_recovery(lazy_edgelist_karate_graph):
    edgelist = lazy_edgelist_karate_graph
    edgelist_with_components = find_components(
        edgelist,
        multiplet_recovery=False,
        min_read_count=1,
        component_size_threshold=(0, 100),
    )

    edgelist = edgelist.collect()

    edgelist_with_components = edgelist_with_components.collect()
    assert_frame_equal(edgelist, edgelist_with_components.drop("component"))

    assert edgelist_with_components["component"].n_unique() == 1


@pytest.mark.slow
def test_find_components_dynamic_size_filter(lazy_edgelist_karate_graph):
    edgelist = lazy_edgelist_karate_graph
    edgelist_with_components = find_components(
        edgelist,
        multiplet_recovery=True,
        min_read_count=1,
        component_size_threshold=True,
        refinement_options=StagedRefinementOptions(
            inital_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            refinement_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            max_component_refinement_depth=1,
        ),
        dynamic_lowest_passable_bound=0,
    )

    edgelist = edgelist.collect()

    edgelist_with_components = edgelist_with_components.collect()
    # There are 25 crossing edges
    assert len(edgelist_with_components) == len(edgelist) - 10
    assert edgelist_with_components["component"].n_unique() == 4


@pytest.mark.slow
def test_find_components_stats(lazy_edgelist_karate_graph):
    edgelist = lazy_edgelist_karate_graph
    _, stats = find_components(
        edgelist,
        multiplet_recovery=True,
        min_read_count=1,
        component_size_threshold=False,
        return_component_statistics=True,
        refinement_options=StagedRefinementOptions(
            inital_stage_options=RefinementOptions(
                min_component_size=1,
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            refinement_stage_options=RefinementOptions(
                min_component_size=1,
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            max_component_refinement_depth=1,
        ),
    )

    assert stats.node_count_pre_recovery == 51
    assert stats.edge_count_pre_recovery == 78

    assert stats.node_count_post_recovery == 51
    assert stats.edge_count_post_recovery == 68

    assert stats.crossing_edges_removed == 10
    assert stats.component_count_pre_recovery == 1
    assert stats.component_count_post_recovery == 4
    assert stats.fraction_nodes_in_largest_component_pre_recovery == 1.0
    assert (
        pytest.approx(stats.fraction_nodes_in_largest_component_post_recovery, abs=1e-3)
        == 0.373
    )
    assert stats.component_size_max_filtering_threshold == np.iinfo(np.uint64).max

    assert stats.reads_input == 780
    assert stats.reads_post_read_count_filtering == 780


@pytest.mark.slow
def test_find_components_fixed_thresholds(lazy_edgelist_karate_graph):
    edgelist = lazy_edgelist_karate_graph
    edgelist, stats = find_components(
        edgelist,
        multiplet_recovery=True,
        min_read_count=1,
        component_size_threshold=(3, 10),
        return_component_statistics=True,
        refinement_options=StagedRefinementOptions(
            inital_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            refinement_stage_options=RefinementOptions(
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            max_component_refinement_depth=1,
        ),
    )

    assert stats.node_count_pre_recovery == 51
    assert stats.edge_count_pre_recovery == 78

    assert stats.node_count_post_recovery == 51
    assert stats.edge_count_post_recovery == 68

    assert stats.crossing_edges_removed == 10
    assert stats.component_count_pre_recovery == 1
    assert stats.component_count_post_recovery == 4
    assert stats.fraction_nodes_in_largest_component_pre_recovery == 1.0
    assert (
        pytest.approx(stats.fraction_nodes_in_largest_component_post_recovery, abs=1e-3)
        == 0.373
    )


# TODO Add tests that loads data from file!


# TODO we need to check that we actually remove edges that are not assigned to a component
# This could happen in the lieden case if there are single node communities


@pytest.mark.slow
def test_build_pxl_file_with_components(lazy_edgelist_karate_graph, mock_panel, tmpdir):
    output = Path(tmpdir) / "output.pxl"

    _, stats = build_pxl_file_with_components(
        molecules_lazy_frame=lazy_edgelist_karate_graph,
        leiden_iterations=1,
        min_count=1,
        multiplet_recovery=True,
        component_size_threshold=False,
        panel=mock_panel,
        sample_name="test_sample",
        path_output_pxl_file=output,
        refinement_options=StagedRefinementOptions(
            inital_stage_options=RefinementOptions(
                min_component_size=1,
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            refinement_stage_options=RefinementOptions(
                min_component_size=1,
                max_edges_to_remove=None,
                max_edges_to_remove_relative=None,
            ),
            max_component_refinement_depth=1,
        ),
    )

    result = PNAPixelDataset.from_pxl_files(output)
    graph_edgelist = result.edgelist().to_df()
    assert set(graph_edgelist["component"].unique()) == {
        "dc467b2de758a377",
        "216e0ae82d88ed5c",
        "82d07c06fbe77d34",
        "051c05ea14e7a441",
    }

    assert stats.molecules_input == 156
    assert stats.reads_input == 780
    assert stats.reads_post_read_count_filtering == 780
    assert stats.molecules_post_read_count_filtering == 156
    assert stats.node_count_pre_recovery == 51
    assert stats.edge_count_pre_recovery == 78

    assert stats.node_count_post_recovery == 51
    assert stats.edge_count_post_recovery == 68

    assert stats.crossing_edges_removed == 10
    assert stats.component_count_pre_recovery == 1
    assert stats.component_count_post_recovery == 4
    assert stats.fraction_nodes_in_largest_component_pre_recovery == 1.0
    assert (
        pytest.approx(stats.fraction_nodes_in_largest_component_post_recovery, abs=1e-3)
        == 0.373
    )

    assert stats.molecules_output == 136
    assert stats.reads_output == 680


@pytest.mark.slow
def test_build_pxl_file_with_components_for_umi_collisions(
    edgelist_karate_graph, mock_panel, tmpdir
):
    edgelist_df = pl.DataFrame(edgelist_karate_graph)
    ## Make the graph bipartite
    edgelist_df = edgelist_df.with_columns(umi2=pl.col("umi2") + pl.col("umi1").max())
    ## Add UMI collisions
    last_umi = np.maximum(edgelist_df["umi1"].max(), edgelist_df["umi2"].max())
    umi1_collisions = pl.DataFrame(
        {
            "umi1": [last_umi + 1, last_umi + 1],  # same umi1
            "umi2": [last_umi + 2, last_umi + 3],
            "read_count": [1, 1],
            "uei_count": [1, 1],
            "marker_1": ["MarkerA", "MarkerB"],  # different marker_1
            "marker_2": ["MarkerB", "MarkerB"],
        }
    )
    umi1_umi2_collisions = pl.DataFrame(
        {
            "umi1": [last_umi + 4, last_umi + 5],
            "umi2": [last_umi + 5, last_umi + 6],
            # "last_umi + 5" appears both as a umi1 and a umi2
            "read_count": [1, 1],
            "uei_count": [1, 1],
            "marker_1": ["MarkerA", "MarkerB"],  # different marker_1
            "marker_2": ["MarkerB", "MarkerB"],
        }
    )
    edgelist_df_colliding = pl.concat(
        [edgelist_df, umi1_collisions, umi1_umi2_collisions]
    ).lazy()
    edgelist_df = edgelist_df.lazy()
    output = Path(tmpdir) / "output.pxl"
    output_colliding = Path(tmpdir) / "output_colliding.pxl"

    build_pxl_file_with_components(
        molecules_lazy_frame=edgelist_df,
        leiden_iterations=1,
        min_count=1,
        multiplet_recovery=True,
        panel=mock_panel,
        sample_name="test_sample",
        path_output_pxl_file=output,
        component_size_threshold=False,
    )
    build_pxl_file_with_components(
        molecules_lazy_frame=edgelist_df_colliding,
        leiden_iterations=1,
        min_count=1,
        multiplet_recovery=True,
        panel=mock_panel,
        sample_name="test_sample",
        path_output_pxl_file=output_colliding,
        component_size_threshold=False,
    )

    result = PNAPixelDataset.from_pxl_files(output)
    result_colliding = PNAPixelDataset.from_pxl_files(output_colliding)
    assert_frame_equal(
        result.edgelist().to_polars(), result_colliding.edgelist().to_polars()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "comp1_size,comp2_size,n_crossing_edges",
    [
        (1000, 2000, 10),
        (1000, 2000, 20),
        (21000, 22000, 20),
        (21000, 22000, 21),
    ],
)
def test_merge_communities_with_many_crossing_edges(
    comp1_size, comp2_size, n_crossing_edges
):
    max_edges_to_remove = 20
    max_edges_to_remove_relative = 0.001
    comp1_edges = pl.DataFrame(
        {"umi1": [0] * (comp1_size - 1), "umi2": range(1, comp1_size)}
    )
    comp2_edges = pl.DataFrame(
        {
            "umi1": [comp1_size] * (comp2_size - 1),
            "umi2": range(comp1_size + 1, comp1_size + comp2_size),
        }
    )

    crossing_edges_1_2 = pl.DataFrame(
        {
            "umi1": [0] * int(n_crossing_edges / 2),
            "umi2": range(comp1_size + 1, comp1_size + int(n_crossing_edges / 2) + 1),
        }
    )
    crossing_edges_2_1 = pl.DataFrame(
        {
            "umi1": [comp1_size] * (n_crossing_edges - int(n_crossing_edges / 2)),
            "umi2": range(1, (n_crossing_edges - int(n_crossing_edges / 2)) + 1),
        }
    )
    edgelist = pl.concat(
        [comp1_edges, comp2_edges, crossing_edges_1_2, crossing_edges_2_1]
    )
    comp_dict = dict.fromkeys(range(comp1_size), 0)
    comp_dict.update(dict.fromkeys(range(comp1_size, comp1_size + comp2_size), 1))
    comp_series = merge_communities_with_many_crossing_edges(
        edgelist,
        comp_dict,
        max_edges_to_remove=max_edges_to_remove,
        max_edges_to_remove_relative=max_edges_to_remove_relative,
    )
    if n_crossing_edges >= max(
        max_edges_to_remove,
        max_edges_to_remove_relative * min(comp1_size, comp2_size),
    ):
        assert len(comp_series.unique()) == 1
        assert len(comp_series[comp_series == 0]) == comp1_size + comp2_size
    else:
        assert len(comp_series.unique()) == 2
        assert len(comp_series[comp_series == 0]) == comp1_size
        assert len(comp_series[comp_series == 1]) == comp2_size
