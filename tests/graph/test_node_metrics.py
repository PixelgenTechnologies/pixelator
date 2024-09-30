"""Test transition probability and local G functions.

Copyright © 2024 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from pandas.testing import assert_frame_equal

from pixelator.graph.node_metrics import compute_transition_probabilities, local_g


def test_compute_transition_probabilities(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()

    # Compute transition probabilities
    W = compute_transition_probabilities(A, k=1, remove_self_loops=False)

    # Expected transition probabilities
    expected_W = np.array(
        [
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0, 0.5, 0.0],
        ]
    )

    # Compare the computed and expected transition probabilities
    assert sp.issparse(W)
    assert np.allclose(W.toarray(), expected_W)


def test_compute_transition_probabilities_remove_self_loops_true(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()

    # Compute transition probabilities
    W = compute_transition_probabilities(A, k=2, remove_self_loops=True)

    # Expected transition probabilities
    expected_W = np.array(
        [
            [0, 0, 0, 0.5, 0.5],
            [0, 0, 0.5, 0.5, 0],
            [0, 0.5, 0, 0, 0.5],
            [0.5, 0.5, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0],
        ]
    )

    # Compare the computed and expected transition probabilities
    assert sp.issparse(W)
    assert np.allclose(W.toarray(), expected_W)


def test_local_g(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()
    counts = pentagram_graph.node_marker_counts

    # Compute local g-scores
    gi_scores = local_g(
        A, counts, k=1, use_weights=True, normalize_counts=True, method="gi"
    )

    # Expected local g-scores
    expected_gi_scores = pd.DataFrame.from_dict(
        {
            0: {"A": 0.0, "B": -1.0, "C": 1.0, "D": 1.0, "E": -1.0},
            2: {"A": 1.0, "B": -1.0, "C": 0.0, "D": -1.0, "E": 1.0},
            3: {"A": 1.0, "B": 1.0, "C": -1.0, "D": 0.0, "E": -1.0},
            1: {"A": -1.0, "B": 0.0, "C": -1.0, "D": 1.0, "E": 1.0},
            4: {"A": -1.0, "B": 1.0, "C": 1.0, "D": -1.0, "E": 0.0},
        },
        orient="index",
    )
    expected_gi_scores.index.name = "node"
    expected_gi_scores.columns.name = "markers"

    # Compare the computed and expected local g-scores
    assert isinstance(gi_scores, pd.DataFrame)
    assert_frame_equal(
        gi_scores.sort_index(), expected_gi_scores.sort_index(), check_column_type=False
    )


def test_local_g_use_weights_false(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()
    # Create a 5x5 DataFrame of marker counts
    counts = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    counts.index.name = "node"
    counts.columns.name = "markers"

    # Compute local g-scores
    gi_scores = local_g(
        A, counts, k=1, use_weights=False, normalize_counts=True, method="gi"
    )

    # Expected local g-scores
    expected_gi_scores = pd.DataFrame.from_dict(
        {
            0: {
                "A": -1.4313800979143823,
                "B": -1.4313800979143463,
                "C": 0.0,
                "D": 1.4313800979143463,
                "E": 1.4313800979144704,
            },
            1: {
                "A": -0.8850404109843606,
                "B": -0.8850404109843573,
                "C": 0.0,
                "D": 0.8850404109843546,
                "E": 0.8850404109843633,
            },
            2: {
                "A": -0.8210543814390132,
                "B": -0.8210543814390153,
                "C": 0.0,
                "D": 0.8210543814390054,
                "E": 0.8210543814390159,
            },
            3: {
                "A": 1.2983285241175744,
                "B": 1.298328524117571,
                "C": 0.0,
                "D": -1.2983285241175513,
                "E": -1.2983285241175755,
            },
            4: {
                "A": 0.9035128891381983,
                "B": 0.9035128891381959,
                "C": 0.0,
                "D": -0.9035128891381841,
                "E": -0.9035128891382024,
            },
        },
        orient="index",
    )
    expected_gi_scores.index.name = "node"
    expected_gi_scores.columns.name = "markers"

    # Compare the computed and expected local g-scores
    assert isinstance(gi_scores, pd.DataFrame)
    assert_frame_equal(
        gi_scores.sort_index(), expected_gi_scores.sort_index(), check_column_type=False
    )


def test_local_g_normalize_counts_false(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()
    # Create a 5x5 DataFrame of marker counts
    counts = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    counts.index.name = "node"
    counts.columns.name = "markers"

    # Compute local g-scores
    gi_scores = local_g(
        A, counts, k=1, use_weights=False, normalize_counts=False, method="gi"
    )

    # Expected local g-scores
    expected_gi_scores = pd.DataFrame.from_dict(
        {
            0: {
                "A": -1.5491933384829668,
                "B": -1.5491933384829668,
                "C": -1.5491933384829668,
                "D": -1.5491933384829668,
                "E": -1.5491933384829668,
            },
            1: {
                "A": -0.29277002188455997,
                "B": -0.29277002188455997,
                "C": -0.29277002188455997,
                "D": -0.29277002188455997,
                "E": -0.29277002188455997,
            },
            2: {
                "A": -0.5477225575051661,
                "B": -0.5477225575051661,
                "C": -0.5477225575051661,
                "D": -0.5477225575051661,
                "E": -0.5477225575051661,
            },
            3: {
                "A": 1.4638501094228,
                "B": 1.4638501094228,
                "C": 1.4638501094228,
                "D": 1.4638501094228,
                "E": 1.4638501094228,
            },
            4: {
                "A": 0.7745966692414834,
                "B": 0.7745966692414834,
                "C": 0.7745966692414834,
                "D": 0.7745966692414834,
                "E": 0.7745966692414834,
            },
        },
        orient="index",
    )
    expected_gi_scores.index.name = "node"
    expected_gi_scores.columns.name = "markers"

    # Compare the computed and expected local g-scores
    assert isinstance(gi_scores, pd.DataFrame)
    assert_frame_equal(
        gi_scores.sort_index(), expected_gi_scores.sort_index(), check_column_type=False
    )


def test_local_g_normalize_method_gstari(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()
    # Create a 5x5 DataFrame of marker counts
    counts = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    counts.index.name = "node"
    counts.columns.name = "markers"

    # Compute local g-scores
    gi_scores = local_g(
        A, counts, k=1, use_weights=True, normalize_counts=True, method="gstari"
    )

    # Expected local g-scores
    expected_gi_scores = pd.DataFrame.from_dict(
        {
            0: {
                "A": -1.181174161915827,
                "B": -1.1811741619158245,
                "C": 0.0,
                "D": 1.1811741619158003,
                "E": 1.1811741619158258,
            },
            1: {
                "A": -0.9257851539340258,
                "B": -0.925785153934024,
                "C": 0.0,
                "D": 0.9257851539340036,
                "E": 0.9257851539340244,
            },
            2: {
                "A": -0.6508872633980606,
                "B": -0.6508872633980609,
                "C": 0.0,
                "D": 0.6508872633980447,
                "E": 0.6508872633980601,
            },
            3: {
                "A": 1.5624841391108724,
                "B": 1.5624841391108693,
                "C": 0.0,
                "D": -1.5624841391108488,
                "E": -1.5624841391108784,
            },
            4: {
                "A": 1.1953624401370362,
                "B": 1.1953624401370306,
                "C": 0.0,
                "D": -1.1953624401370189,
                "E": -1.1953624401370413,
            },
        },
        orient="index",
    )
    expected_gi_scores.index.name = "node"
    expected_gi_scores.columns.name = "markers"

    # Compare the computed and expected local g-scores
    assert isinstance(gi_scores, pd.DataFrame)
    assert_frame_equal(
        gi_scores.sort_index(), expected_gi_scores.sort_index(), check_column_type=False
    )


def test_local_g_k4(pentagram_graph):
    # Create a sparse adjacency matrix
    A = pentagram_graph.get_adjacency_sparse()
    # Create a 5x5 DataFrame of marker counts
    counts = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    counts.index.name = "node"
    counts.columns.name = "markers"

    # Compute local g-scores
    gi_scores = local_g(
        A, counts, k=4, use_weights=True, normalize_counts=True, method="gstari"
    )

    # Expected local g-scores
    expected_gi_scores = pd.DataFrame.from_dict(
        {
            0: {
                "A": -1.6919339037036998,
                "B": -1.6919339037037184,
                "C": 0.0,
                "D": 1.6919339037036492,
                "E": 1.691933903703703,
            },
            1: {
                "A": -0.6945173598360008,
                "B": -0.6945173598360104,
                "C": 0.0,
                "D": 0.6945173598359781,
                "E": 0.6945173598359913,
            },
            2: {
                "A": -0.30065880545156753,
                "B": -0.3006588054515778,
                "C": 0.0,
                "D": 0.30065880545157325,
                "E": 0.3006588054516006,
            },
            3: {
                "A": 1.4593826070876255,
                "B": 1.459382607087601,
                "C": 0.0,
                "D": -1.4593826070876004,
                "E": -1.4593826070876499,
            },
            4: {
                "A": 1.2277274619036243,
                "B": 1.2277274619036218,
                "C": 0.0,
                "D": -1.2277274619036032,
                "E": -1.2277274619036265,
            },
        },
        orient="index",
    )

    expected_gi_scores.index.name = "node"
    expected_gi_scores.columns.name = "markers"

    # Compare the computed and expected local g-scores
    assert isinstance(gi_scores, pd.DataFrame)
    assert_frame_equal(
        gi_scores.sort_index(), expected_gi_scores.sort_index(), check_column_type=False
    )


# Run the tests
if __name__ == "__main__":
    pytest.main()
