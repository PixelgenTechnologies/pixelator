"""Tests for Graph.layout_coordinates.

Copyright © 2026 Pixelgen Technologies AB.
"""

from typing import get_args

import networkx as nx
import pandas as pd
import pytest

from pixelator.common.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.common.graph.graph import Graph


def _attributed_cycle(n: int = 40) -> Graph:
    graph = nx.cycle_graph(n)
    nx.set_node_attributes(graph, {i: str(i) for i in graph.nodes()}, "name")
    nx.set_node_attributes(
        graph, {i: "A" if i % 2 == 0 else "B" for i in graph.nodes()}, "pixel_type"
    )
    return Graph.from_raw(graph)


def test_layout_coordinates_seed_kwarg_seeds_the_default_algorithm():
    """``seed=`` is the algorithm parameter and must seed, not TypeError."""
    graph = _attributed_cycle()
    common = dict(get_node_marker_matrix=False, only_keep_a_pixels=False)
    via_seed = graph.layout_coordinates(**common, seed=42)
    via_random_seed = graph.layout_coordinates(**common, random_seed=42)
    pd.testing.assert_frame_equal(via_seed, via_random_seed)


@pytest.mark.parametrize(
    "algorithm",
    [
        "fruchterman_reingold",
        "fruchterman_reingold_3d",
        "kamada_kawai",
        "kamada_kawai_3d",
    ],
)
def test_removed_force_directed_layout_algorithms_are_rejected(algorithm):
    assert algorithm not in get_args(SupportedLayoutAlgorithm)
    graph = _attributed_cycle()
    with pytest.raises(AssertionError, match="not allowed"):
        graph.layout_coordinates(
            layout_algorithm=algorithm,  # type: ignore[arg-type]
            get_node_marker_matrix=False,
            only_keep_a_pixels=False,
        )
