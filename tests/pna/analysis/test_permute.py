"""
Tests for the colocalization modules

Copyright © 2024 Pixelgen Technologies AB.
"""

import polars as pl
from numpy.random import default_rng
from polars.testing import assert_frame_equal

from pixelator.pna.analysis.permute import edgelist_permutations, permute_edgelist

random_number_generator = default_rng(seed=747)


def test_permute_edgelist():
    n_markers = 4
    n_nodes = 50
    n_edges = 500
    umi1_marker_map = pl.DataFrame(
        [
            [i for i in range(n_nodes)],
            random_number_generator.integers(0, n_markers, size=(n_nodes)),
        ],
        schema=["umi1", "marker_1"],
    )
    umi2_marker_map = pl.DataFrame(
        [
            [i for i in range(n_nodes)],
            random_number_generator.integers(0, n_markers, size=(n_nodes)),
        ],
        schema=["umi2", "marker_2"],
    )

    edgelist = pl.DataFrame(
        random_number_generator.integers(0, n_nodes, size=(n_edges, 2)),
        schema=["umi1", "umi2"],
    ).with_columns(
        marker_1=pl.col("umi1").replace_strict(
            umi1_marker_map["umi1"], umi1_marker_map["marker_1"]
        ),
        marker_2=pl.col("umi2").replace_strict(
            umi2_marker_map["umi2"], umi2_marker_map["marker_2"]
        ),
    )

    result = permute_edgelist(edgelist)

    assert edgelist.shape == result.shape
    # We want to preserve the number counts in each region
    m1_counts = umi1_marker_map.group_by("marker_1").len().sort("marker_1")
    m2_counts = umi2_marker_map.group_by("marker_2").len().sort("marker_2")

    m1_counts_perm = (
        result.select(["umi1", "marker_1"])
        .unique()
        .group_by("marker_1")
        .len()
        .sort("marker_1")
    )
    m2_counts_perm = (
        result.select(["umi2", "marker_2"])
        .unique()
        .group_by("marker_2")
        .len()
        .sort("marker_2")
    )

    assert_frame_equal(m1_counts, m1_counts_perm)
    assert_frame_equal(m2_counts, m2_counts_perm)


def test_permutations():
    df = pl.DataFrame(
        random_number_generator.integers(0, 100, size=(200, 4)),
        schema=["umi1", "umi2", "marker_1", "marker_2"],
    )
    result = edgelist_permutations(df, n=100)
    assert len(list(result)) == 100
