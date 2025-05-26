"""Copyright © 2024 Pixelgen Technologies AB."""

from pixelator.pna.graph.report import GraphSampleReport, GraphStatistics


def test_can_convert_component_statistics_to_graph_sample_report():
    component_stats = GraphStatistics(
        component_count_pre_recovery=10,
        component_count_post_recovery=8,
        fraction_nodes_in_largest_component_pre_recovery=0.5,
        fraction_nodes_in_largest_component_post_recovery=0.7,
        crossing_edges_removed=5,
        component_size_min_filtering_threshold=100,
        component_size_max_filtering_threshold=500,
        node_count_pre_recovery=1000,
        edge_count_pre_recovery=2000,
        node_count_post_recovery=900,
        edge_count_post_recovery=1800,
        pre_filtering_component_sizes={
            1000: 1,
            900: 1,
            800: 1,
            700: 1,
            600: 1,
            500: 1,
            400: 1,
            300: 1,
            200: 1,
        },
    )

    report = GraphSampleReport(
        sample_id="test_sample",
        product_id="single-cell-pna",
        **component_stats.to_dict(),
    )

    assert report.component_count_pre_recovery == 10
    assert report.component_count_post_recovery == 8
    assert report.fraction_nodes_in_largest_component_pre_recovery == 0.5
    assert report.fraction_nodes_in_largest_component_post_recovery == 0.7
    assert report.crossing_edges_removed == 5
    assert report.component_size_min_filtering_threshold == 100
    assert report.component_size_max_filtering_threshold == 500
    assert report.node_count_pre_recovery == 1000
    assert report.edge_count_pre_recovery == 2000
    assert report.node_count_post_recovery == 900
    assert report.edge_count_post_recovery == 1800
    assert report.pre_filtering_component_sizes == {
        1000: 1,
        900: 1,
        800: 1,
        700: 1,
        600: 1,
        500: 1,
        400: 1,
        300: 1,
        200: 1,
    }


def test_graph_sample_report():
    report = GraphSampleReport(
        sample_id="test_sample",
        product_id="single-cell-pna",
        molecules_input=1000,
        reads_input=2000,
        molecules_post_umi_collision_removal=950,
        reads_post_umi_collision_removal=1950,
        molecules_output=900,
        reads_output=1800,
        reads_post_read_count_filtering=1900,
        molecules_post_read_count_filtering=1900,
        component_count_pre_recovery=10,
        component_count_post_recovery=8,
        component_count_pre_component_size_filtering=8,
        component_count_post_component_size_filtering=6,
        fraction_nodes_in_largest_component_pre_recovery=0.5,
        fraction_nodes_in_largest_component_post_recovery=0.7,
        crossing_edges_removed=5,
        component_size_min_filtering_threshold=100,
        component_size_max_filtering_threshold=500,
        node_count_pre_recovery=1000,
        edge_count_pre_recovery=2000,
        node_count_post_recovery=900,
        edge_count_post_recovery=1800,
        crossing_edges_removed_initial_stage=10,
        pre_filtering_component_sizes={
            1000: 1,
            900: 1,
            800: 1,
            700: 1,
            600: 1,
            500: 1,
            400: 1,
            300: 1,
            200: 1,
        },
        median_reads_per_component=295,
        median_markers_per_component=12,
        aggregate_count=3,
        read_count_in_aggregates=123,
        edge_count_in_aggregates=456,
    )

    assert report.component_count_pre_recovery == 10
    assert report.component_count_post_recovery == 8
    assert report.fraction_nodes_in_largest_component_pre_recovery == 0.5
    assert report.fraction_nodes_in_largest_component_post_recovery == 0.7
    assert report.crossing_edges_removed == 5
    assert report.component_size_min_filtering_threshold == 100
    assert report.component_size_max_filtering_threshold == 500
    assert report.node_count_pre_recovery == 1000
    assert report.edge_count_pre_recovery == 2000
    assert report.node_count_post_recovery == 900
    assert report.edge_count_post_recovery == 1800
    assert report.crossing_edges_removed_initial_stage == 10
    assert report.pre_filtering_component_sizes == {
        1000: 1,
        900: 1,
        800: 1,
        700: 1,
        600: 1,
        500: 1,
        400: 1,
        300: 1,
        200: 1,
    }
    assert report.median_reads_per_component == 295
    assert report.median_markers_per_component == 12
    assert report.aggregate_count == 3
    assert report.read_count_in_aggregates == 123
    assert report.edge_count_in_aggregates == 456
    assert report.fraction_of_aggregate_components == 0.5
    assert report.molecules_post_read_count_filtering == 1900
    assert report.reads_post_read_count_filtering == 1900
    assert report.reads_input == 2000
    assert report.reads_output == 1800
    assert report.reads_post_umi_collision_removal == 1950
    assert report.molecules_post_umi_collision_removal == 950


def test_graph_sample_report_to_json(snapshot):
    report = GraphSampleReport(
        sample_id="test_sample",
        product_id="single-cell-pna",
        molecules_input=1000,
        molecules_output=900,
        molecules_post_umi_collision_removal=950,
        reads_post_umi_collision_removal=1950,
        reads_input=2000,
        reads_output=1800,
        molecules_post_read_count_filtering=1000,
        reads_post_read_count_filtering=2000,
        component_count_pre_recovery=10,
        component_count_post_recovery=8,
        component_count_pre_component_size_filtering=8,
        component_count_post_component_size_filtering=6,
        fraction_nodes_in_largest_component_pre_recovery=0.5,
        fraction_nodes_in_largest_component_post_recovery=0.7,
        crossing_edges_removed=5,
        component_size_min_filtering_threshold=100,
        component_size_max_filtering_threshold=500,
        node_count_pre_recovery=1000,
        edge_count_pre_recovery=2000,
        node_count_post_recovery=900,
        edge_count_post_recovery=1800,
        crossing_edges_removed_initial_stage=10,
        pre_filtering_component_sizes={
            1000: 1,
            900: 1,
            800: 1,
            700: 1,
            600: 1,
            500: 1,
            400: 1,
            300: 1,
            200: 1,
        },
        median_reads_per_component=295,
        median_markers_per_component=12,
        aggregate_count=100,
        read_count_in_aggregates=123,
        edge_count_in_aggregates=456,
    )
    snapshot.assert_match(report.to_json(indent=4), "graph_sample_report.json")
