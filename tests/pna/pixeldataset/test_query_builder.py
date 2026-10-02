"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.pixeldataset.io import QueryBuilder


def test_edgelist_query_for_single_component_uses_equality():
    """Verify edgelist query for single component uses equality."""
    query = QueryBuilder().edgelist_query(["c1"])
    assert "component = $components" in query.sql
    assert query.params == {"components": "c1"}


def test_edgelist_query_for_multiple_components_uses_in_clause():
    """Verify edgelist query for multiple components uses in clause."""
    query = QueryBuilder().edgelist_query(["c1", "c2"])
    assert "component IN $components" in query.sql
    assert query.params == {"components": ["c1", "c2"]}


def test_proximity_query_contains_marker_filter_when_markers_provided():
    """Verify proximity query contains marker filter when markers provided."""
    query = QueryBuilder().proximity_query(["c1"], ["M1", "M2"])
    assert "(marker_1 IN $markers AND marker_2 IN $markers)" in query.sql
    assert query.params == {"components": "c1", "markers": ["M1", "M2"]}


def test_proximity_query_filters_each_sample_by_its_stored_ids():
    """Verify each sample gets its own marker filter."""
    query = QueryBuilder().proximity_query(
        None,
        {"sample_old": ["MarkerA"], "sample_new": ["MarkerANew"]},
    )
    assert "sample = $marker_sample_0" in query.sql
    assert "marker_1 IN $markers_0 AND marker_2 IN $markers_0" in query.sql
    assert "sample = $marker_sample_1" in query.sql
    assert query.params["marker_sample_0"] == "sample_old"
    assert query.params["markers_0"] == ["MarkerA"]
    assert query.params["markers_1"] == ["MarkerANew"]


def test_edgelist_proximity_qualifies_sample_in_the_expected_join():
    """A per-sample marker filter must name which sample column it uses.

    Proximity calculated from the edgelist joins stats_m1, stats_m2, and
    group_edges, and each of those has a sample column. The expected-count
    filter therefore uses t1.sample. The observed count reads the edgelist
    alone, so that filter still says sample.
    """
    query = QueryBuilder().proximity_query(
        None,
        {"sample_old": ["MarkerA"], "sample_new": ["MarkerANew"]},
        calculate_from_edgelist=True,
    )
    assert "t1.sample = $marker_sample_0" in query.sql
    assert "t1.marker_1 IN $markers_0 AND t2.marker_2 IN $markers_0" in query.sql
    assert "(sample = $marker_sample_0 AND marker_1 IN $markers_0" in query.sql


def test_proximity_query_without_markers_uses_true_guard():
    """Verify proximity query without markers uses true guard."""
    query = QueryBuilder().proximity_query(["c1"], None)
    assert "(marker_1 IN $markers AND marker_2 IN $markers)" not in query.sql
    assert query.params == {"components": "c1"}
