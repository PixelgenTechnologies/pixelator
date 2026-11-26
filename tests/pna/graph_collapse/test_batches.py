"""Tests for how graph-collapse groups components into batches.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pixelator.pna.graph_collapse.batches import plan_component_batches


def test_batches_are_filled_to_the_edge_count_in_order():
    """A batch closes once it reaches the edge count, so c and e stand alone."""
    sizes = [("a", 3), ("b", 4), ("c", 20), ("d", 2), ("e", 9), ("f", 1)]

    assert plan_component_batches(sizes, 7) == [["a", "b"], ["c"], ["d", "e"], ["f"]]
    assert plan_component_batches(sizes, 1000) == [["a", "b", "c", "d", "e", "f"]]
    assert plan_component_batches([], 7) == []
