"""Tests for marker and umi population.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from pixelator.common.utils.test_data_generator.molecules import (
    _assign_markers,
    _assign_umis,
    _correlate_neighbors,
    _marker_probabilities,
)


def test_assign_umis():
    """Verify node mapping, umi value range and schema of _assign_umis."""
    # node 2 appears in both columns and some rows are repeated, so the umi map
    # must collapse them to a single row per unique node
    edgelist = pl.DataFrame(
        {
            "node1": [1, 2, 1, 5],
            "node2": [2, 3, 2, 4],
        }
    )
    expected_nodes = {1, 2, 3, 4, 5}

    umi_map = _assign_umis(edgelist, np.random.default_rng(0))

    # exactly the schema downstream joins rely on
    assert umi_map.columns == ["node", "umi"]

    # one row per unique node, covering exactly the nodes in the edgelist
    assert umi_map.height == len(expected_nodes)
    assert umi_map["node"].n_unique() == umi_map.height
    assert set(umi_map["node"].to_list()) == expected_nodes

    # umis are 56-bit, i.e. within [0, 2**56)
    umis = umi_map["umi"]
    assert (umis >= 0).all()
    assert (umis < (1 << 56)).all()


def test_assign_umis_reproducible():
    """Verify _assign_umis is deterministic for a given rng."""
    edgelist = pl.DataFrame({"node1": [1, 3, 5], "node2": [0, 2, 4]})

    first = _assign_umis(edgelist, np.random.default_rng(0))

    # the same seed yields an identical umi map
    assert_frame_equal(first, _assign_umis(edgelist, np.random.default_rng(0)))

    # a different seed yields different umis
    assert not first.equals(_assign_umis(edgelist, np.random.default_rng(1)))


def test_marker_probabilities(marker_panel):
    """Verify abundance tiers, hashing exclusion and normalization."""
    markers, probs, is_hashing = _marker_probabilities(marker_panel)

    # parallel arrays aligned with the panel order
    assert list(markers) == [
        "Hash0",
        "MarkerA",
        "MarkerB",
        "MarkerC",
        "MarkerD",
        "MarkerE",
        "MarkerF",
        "Hash1",
    ]
    assert markers.shape == probs.shape == is_hashing.shape == (8,)

    # exactly the two hashing markers are flagged, aligned with marker order
    assert list(is_hashing) == [True, False, False, False, False, False, False, True]

    # six non-hashing markers split into tiers of 1 high / 2 medium / 3 low
    n_high, n_medium, n_low = 1, 2, 3
    raw = np.array(
        [
            0.1 / n_low,  # Hash0 (hashing -> low rate, excluded from tiers)
            0.5 / n_high,  # MarkerA (high)
            0.4 / n_medium,  # MarkerB (medium)
            0.4 / n_medium,  # MarkerC (medium)
            0.1 / n_low,  # MarkerD (low)
            0.1 / n_low,  # MarkerE (low)
            0.1 / n_low,  # MarkerF (low)
            0.1 / n_low,  # Hash1 (hashing -> low rate, excluded from tiers)
        ]
    )
    expected = raw / raw.sum()

    # probabilities match the expected per-tier values
    assert np.allclose(probs, expected)

    # normalization and basic sanity
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()

    # hashing markers are sampled at the low non-hashing per-marker rate
    assert np.allclose(probs[is_hashing], probs[4])


def test_assign_markers_proportions(marker_panel):
    """Verify sampled markers appear in the expected proportions.

    The ~5% hashing overwrite scales every marker by 0.95 and adds 0.05 to the
    single chosen hashing marker, so observed proportions are checked against
    ``0.95 * probs`` for regular markers and the hashing markers are checked as
    a group (invariant to which hashing marker is picked).
    """
    n = 20_000
    umi_map = pl.DataFrame({"node": range(n)})

    result = _assign_markers(umi_map, marker_panel, np.random.default_rng(0))

    markers, probs, is_hashing = _marker_probabilities(marker_panel)
    counts = dict(
        result["marker"].value_counts().iter_rows()  # marker -> count
    )
    observed = {m: counts.get(m, 0) / n for m in markers}

    atol = 0.02

    # regular markers follow the abundance tiers, scaled by the 0.95 keep rate
    for marker, prob, hashing in zip(markers, probs, is_hashing):
        if not hashing:
            assert abs(observed[marker] - 0.95 * prob) < atol

    # the hashing markers as a group absorb the extra 0.05 overwrite
    observed_hashing = sum(observed[m] for m in markers[is_hashing])
    expected_hashing = 0.95 * probs[is_hashing].sum() + 0.05
    assert abs(observed_hashing - expected_hashing) < atol


def test_correlate_neighbors_membership():
    """Each node keeps its marker or copies one of its neighbours' markers."""
    edgelist = pl.DataFrame({"node1": [0, 0, 2], "node2": [1, 3, 3]})
    node_umi_map = pl.DataFrame(
        {
            "node": [0, 1, 2, 3],
            "umi": [10, 11, 12, 13],
            "marker": ["A", "B", "C", "D"],  # distinct per node
        }
    )

    result = _correlate_neighbors(node_umi_map, edgelist, np.random.default_rng(0))

    # schema, node set and ordering preserved, umi untouched
    assert result.columns == ["node", "umi", "marker"]
    assert result["node"].to_list() == [0, 1, 2, 3]
    assert result["umi"].to_list() == [10, 11, 12, 13]

    # distinct markers make each marker identify its source node, so membership
    # proves a copied marker came from an actual neighbour
    neighbours = {0: {1, 3}, 1: {0}, 2: {3}, 3: {0, 2}}
    original = dict(zip(node_umi_map["node"], node_umi_map["marker"]))
    for node, marker in zip(result["node"], result["marker"]):
        allowed = {original[node]} | {original[n] for n in neighbours[node]}
        assert marker in allowed


def test_correlate_neighbors_rate():
    """Verify roughly 10% of nodes copy a neighbour's marker."""
    n = 20_000
    # pair node 2k with 2k+1; a 2-colouring by parity means neighbours always
    # have a different marker, so any copy is observable as a flip
    node_umi_map = pl.DataFrame(
        {
            "node": range(n),
            "umi": range(n),
            "marker": ["X" if node % 2 == 0 else "Y" for node in range(n)],
        }
    )
    edgelist = pl.DataFrame(
        {
            "node1": range(0, n, 2),
            "node2": range(1, n, 2),
        }
    )

    result = _correlate_neighbors(node_umi_map, edgelist, np.random.default_rng(0))

    changed = sum(
        marker != ("X" if node % 2 == 0 else "Y")
        for node, marker in zip(result["node"], result["marker"])
    )
    assert abs(changed / n - 0.10) < 0.02
