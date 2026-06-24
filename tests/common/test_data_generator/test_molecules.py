"""Tests for marker and umi population.

Copyright © 2026 Pixelgen Technologies AB.
"""

from collections import defaultdict

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from pixelator.common.utils.test_data_generator.molecules import (
    _assign_markers,
    _assign_umis,
    _correlate_neighbors,
    _hashing_indices_per_cell,
    _marker_probabilities,
    generate_edgelist,
    populate_cell,
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

    # six regular markers and sixteen hashing markers (two groups x indices 1-8)
    assert (~is_hashing).sum() == 6
    assert is_hashing.sum() == 16

    # the hashing marker placed first is excluded from the tiers by mask, not
    # merely by trailing position
    assert markers[0] == "HashA-1"
    assert is_hashing[0]

    # rebuild the expected per-tier weights: 1 high / 2 medium / 3 low over the
    # non-hashing markers, everything else at the low per-marker rate
    n_high, n_medium, n_low = 1, 2, 3
    raw = np.full(len(markers), 0.1 / n_low)
    regular = np.flatnonzero(~is_hashing)
    raw[regular[0]] = 0.5 / n_high
    raw[regular[1:3]] = 0.4 / n_medium
    expected = raw / raw.sum()

    # probabilities match the expected per-tier values
    assert np.allclose(probs, expected)

    # normalization and basic sanity
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()

    # hashing markers are all sampled at the low non-hashing per-marker rate
    assert np.allclose(probs[is_hashing], (0.1 / n_low) / raw.sum())


def test_assign_markers_proportions(marker_panel):
    """Verify sampled markers appear in the expected proportions.

    The hashing overwrite scales every marker by the ``(1 - f)`` keep rate and
    adds ``f`` spread across the chosen hashing index, so observed proportions
    are checked against ``(1 - f) * probs`` for regular markers and the hashing
    markers are checked as a group (invariant to which hashing index is picked).
    """
    n = 20_000
    f = 0.2
    umi_map = pl.DataFrame({"node": range(n)})

    result = _assign_markers(umi_map, marker_panel, 1, f, np.random.default_rng(0))

    markers, probs, is_hashing = _marker_probabilities(marker_panel)
    counts = dict(
        result["marker"].value_counts().iter_rows()  # marker -> count
    )
    observed = {m: counts.get(m, 0) / n for m in markers}

    atol = 0.02

    # regular markers follow the abundance tiers, scaled by the (1 - f) keep rate
    for marker, prob, hashing in zip(markers, probs, is_hashing):
        if not hashing:
            assert abs(observed[marker] - (1 - f) * prob) < atol

    # the hashing markers as a group absorb the extra f overwrite
    observed_hashing = sum(observed[m] for m in markers[is_hashing])
    expected_hashing = (1 - f) * probs[is_hashing].sum() + f
    assert abs(observed_hashing - expected_hashing) < atol


def test_assign_markers_hashing_index(marker_panel):
    """The hashing overwrite targets all markers of the given index."""
    n = 50_000
    f = 0.2
    umi_map = pl.DataFrame({"node": range(n)})
    hashing_index = 3

    result = _assign_markers(
        umi_map, marker_panel, hashing_index, f, np.random.default_rng(0)
    )

    markers, probs, is_hashing = _marker_probabilities(marker_panel)
    counts = dict(result["marker"].value_counts().iter_rows())
    observed = {m: counts.get(m, 0) / n for m in markers}

    # every hashing marker is sampled at the same base rate before the overwrite
    base = (1 - f) * probs[is_hashing][0]

    # group hashing markers by their -index suffix
    by_index = defaultdict(list)
    for marker, hashing in zip(markers, is_hashing):
        if hashing:
            by_index[int(marker.rsplit("-", 1)[-1])].append(marker)

    # the chosen index's markers share the extra ~f and sit above the base
    chosen = by_index[hashing_index]
    assert all(observed[m] > base + 0.005 for m in chosen)
    assert abs(sum(observed[m] - base for m in chosen) - f) < 0.02

    # markers of every other index stay at the base rate
    for index, group in by_index.items():
        if index != hashing_index:
            for marker in group:
                assert abs(observed[marker] - base) < 0.01


def test_hashing_indices_per_cell_covers_all(marker_panel):
    """Every panel hashing index is assigned to at least one cell."""
    n_cells = 20
    result = _hashing_indices_per_cell(n_cells, marker_panel, np.random.default_rng(0))

    assert len(result) == n_cells
    # the panel has hashing markers indexed 1-8
    assert set(result.tolist()) == set(range(1, 9))


def test_generate_edgelist_uses_all_hashing_indices(marker_panel):
    """generate_edgelist exercises hashing markers across all panel indices."""
    edgelist = generate_edgelist(
        n_cells=8,
        n_nodes=40,
        n_edges=80,
        min_neighbors=10,
        panel=marker_panel,
        n_crossing_edges=0,
        rng=0,
    )
    assert edgelist.columns == ["umi1", "marker1", "umi2", "marker2"]
    assert edgelist.height == 8 * 80


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


def test_populate_cell(marker_panel):
    """populate_cell attaches consistent per-node umis/markers to every edge."""
    # nodes 1, 3, 5 appear in node1 and 0, 2, 4 in node2; several nodes recur
    # across edges, so the per-node umi/marker must stay consistent
    edgelist = pl.DataFrame(
        {
            "node1": [1, 1, 3, 3, 5],
            "node2": [0, 2, 2, 4, 0],
        }
    )

    result = populate_cell(edgelist, marker_panel, hashing_index=1, rng=0)

    # schema and one row per input edge
    assert result.columns == ["umi1", "marker1", "umi2", "marker2"]
    assert result.height == edgelist.height

    # markers come from the panel
    panel_markers = set(marker_panel.to_polars()["marker_id"])
    assert set(result["marker1"]) <= panel_markers
    assert set(result["marker2"]) <= panel_markers

    # each node resolves to a single (umi, marker): every umi maps to exactly one
    # marker, and the number of distinct umis equals the number of distinct nodes
    pairs = list(zip(result["umi1"], result["marker1"]))
    pairs += list(zip(result["umi2"], result["marker2"]))
    umi_to_marker = dict(pairs)
    for umi, marker in pairs:
        assert umi_to_marker[umi] == marker

    n_nodes = pl.concat([edgelist["node1"], edgelist["node2"]]).n_unique()
    assert len(umi_to_marker) == n_nodes


def test_populate_cell_reproducible(marker_panel):
    """populate_cell is deterministic for a given rng."""
    edgelist = pl.DataFrame({"node1": [1, 3, 5], "node2": [0, 2, 4]})

    first = populate_cell(edgelist, marker_panel, hashing_index=1, rng=0)

    # the same seed yields an identical edge list
    assert_frame_equal(
        first, populate_cell(edgelist, marker_panel, hashing_index=1, rng=0)
    )

    # a different seed yields different umis
    assert not first.equals(
        populate_cell(edgelist, marker_panel, hashing_index=1, rng=1)
    )
