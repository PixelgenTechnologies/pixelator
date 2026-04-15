"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.pixeldataset.io import QueryBuilder


def test_edgelist_query_for_single_component_uses_equality():
    query = QueryBuilder().edgelist_query(["c1"])
    assert "component = $components" in query.sql
    assert query.params == {"components": "c1"}


def test_edgelist_query_for_multiple_components_uses_in_clause():
    query = QueryBuilder().edgelist_query(["c1", "c2"])
    assert "component IN $components" in query.sql
    assert query.params == {"components": ["c1", "c2"]}


def test_proximity_query_contains_marker_filter_when_markers_provided():
    query = QueryBuilder().proximity_query(["c1"], ["M1", "M2"])
    assert "(marker_1 IN $markers AND marker_2 IN $markers)" in query.sql
    assert query.params == {"components": "c1", "markers": ["M1", "M2"]}


def test_proximity_query_without_markers_uses_true_guard():
    query = QueryBuilder().proximity_query(["c1"], None)
    assert "(marker_1 IN $markers AND marker_2 IN $markers)" not in query.sql
    assert query.params == {"components": "c1"}
