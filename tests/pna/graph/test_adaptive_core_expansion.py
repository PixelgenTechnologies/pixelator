"""Tests for adaptive_core_expansion.

Modified to match the structure and intent of the R test suite.

Copyright © 2025 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pixelator import read_pna as read
from pixelator.pna.graph.adaptive_core_expansion import (
    _build_p_step_matrix,
    _expand_k_core_layer_until_min_seed_pct,
    _find_largets_connect_component_of_k_core,
    _find_partitions,
    _get_k_cores,
    _PartitionCandidate,
    adaptive_core_expansion,
    validate_best_candidate,
)

PXL_FILE = Path(__file__).parents[2] / "pna" / "data" / "PNA055_Sample07_S7.layout.pxl"
COMPONENT_ID = "0a45497c6bfbfb22"


@pytest.fixture(scope="module")
def graph_from_pxl():
    """Load a single graph from the minimal PNA PBMC pxl file."""
    pg_data = read(str(PXL_FILE))
    g = pg_data.filter(components=COMPONENT_ID).edgelist().iterator().__next__().graph
    return g


@pytest.mark.slow
def test_adaptive_core_expansion_works_as_expected(graph_from_pxl):
    """Test that adaptive_core_expansion works for various valid inputs."""
    # Basic execution and exact count check
    res = adaptive_core_expansion(graph_from_pxl)
    partitions = list(nx.get_node_attributes(res.raw, "partition").values())
    partition_counts = {
        "high": partitions.count("high"),
        "low": partitions.count("low"),
    }
    # Expected counts for COMPONENT_ID in PNA055_Sample07_S7.layout.pxl
    assert partition_counts == {"high": 43144, "low": 399}

    # Test with different valid parameter combinations (expect no errors)
    adaptive_core_expansion(graph_from_pxl, k=2)
    adaptive_core_expansion(graph_from_pxl, max_iter=10)
    adaptive_core_expansion(graph_from_pxl, min_seed_pct=0.2)
    adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold=100)


def test_adaptive_core_expansion_fails_with_invalid_input(graph_from_pxl):
    """Test that adaptive_core_expansion raises errors for invalid inputs."""
    # Invalid graph input
    with pytest.raises((AttributeError, TypeError)):
        adaptive_core_expansion("Invalid")

    # Invalid types for numerical parameters
    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, k="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, max_iter="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, min_seed_pct="Invalid")

    with pytest.raises((AssertionError, TypeError)):
        adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold="Invalid")

    # Out of bounds numerical values
    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, max_iter=0)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, max_iter=1001)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, min_seed_pct=2.0)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, nodes_to_move_threshold=-1)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, min_allowed_nodes_pct=1.0)

    with pytest.raises(ValueError):
        adaptive_core_expansion(graph_from_pxl, min_allowed_nodes_pct=-0.1)


def test_adaptive_core_expansion_min_allowed_nodes_pct(graph_from_pxl, caplog):
    """Test that min_allowed_nodes_pct correctly influences partition selection."""
    # Run with a high min_allowed_nodes_pct that the best BC score partition
    # might not meet, forcing a different selection or fallback.
    # From previous tests, the default high count is ~43k out of ~43.5k (~99%).
    # We'll set a very high threshold to see if it responds.
    with caplog.at_level(logging.WARNING):
        res = adaptive_core_expansion(graph_from_pxl, min_allowed_nodes_pct=0.999)

    partitions = list(nx.get_node_attributes(res.raw, "partition").values())
    partition_counts = {
        "high": partitions.count("high"),
        "low": partitions.count("low"),
    }
    # If no partition meets 99.9%, it should set all to "high"
    assert partition_counts["low"] == 0
    assert partition_counts["high"] == len(res.raw.nodes())
    assert "Found no partition with at least 99.90%" in caplog.text


@pytest.mark.slow
def test_adaptive_core_expansion_isotype_reduction():
    """Test that the core partition has fewer isotype controls."""
    # Use the requested COMPONENT_ID
    comp_id = "c3c393e9a17c1981"
    pg_data = read(str(PXL_FILE))
    # Fetch the specific component requested
    g = next(pg_data.filter(components=comp_id).edgelist().iterator()).graph

    # Run ACE
    res = adaptive_core_expansion(g)

    # Get node metadata (marker counts) and partition
    node_data = res.node_marker_counts
    partitions = nx.get_node_attributes(res.raw, "partition")

    # Map partitions back to the node data rows
    node_data["partition"] = node_data.index.map(partitions)

    # Calculate isotype control sums
    isotypes = ["mIgG1", "mIgG2a", "mIgG2b"]
    # Check if all isotypes exist in the data to avoid KeyErrors
    available_isotypes = [c for c in isotypes if c in node_data.columns]

    # Sum of isotype counts for high vs low
    isotype_counts = (
        node_data.groupby("partition")[available_isotypes].sum().sum(axis=1)
    )

    high_isotype_sum = isotype_counts.get("high", 0)
    low_isotype_sum = isotype_counts.get("low", 0)
    total_isotype_sum = high_isotype_sum + low_isotype_sum

    # Check that high core isotype percentage is below 25%
    high_pct = high_isotype_sum / total_isotype_sum if total_isotype_sum > 0 else 0

    assert high_pct < 0.25


def test_expand_k_core_layer_until_min_seed_pct_reduces_max_k():
    """K-core expansion should lower max_k until seed size threshold is met."""
    k_cores = np.array([4, 4, 3, 3, 3])
    expanded, max_k = _expand_k_core_layer_until_min_seed_pct(
        k_cores=k_cores, max_k=4, min_seed_pct=0.5
    )
    assert max_k == 3
    assert np.array_equal(expanded, np.array([3, 3, 3, 3, 3]))


def test_expand_k_core_layer_until_min_seed_pct_raises_when_unreachable():
    """Raise if even k=1 cannot satisfy minimum seed fraction."""
    with pytest.raises(ValueError, match="min_seed_pct"):
        _expand_k_core_layer_until_min_seed_pct(
            k_cores=np.array([2, 1, 1, 1]), max_k=2, min_seed_pct=0.9
        )


def test_get_k_cores_caps_high_layers():
    """K-core levels above max_k_core should be capped."""
    graph = nx.complete_graph(5)
    node_list = list(graph.nodes())
    k_cores, max_k = _get_k_cores(
        graph=graph, node_list=node_list, max_k_core=3, min_seed_pct=0.1
    )
    assert max_k == 3
    assert np.all(k_cores <= 3)


def test_get_k_cores_raises_when_max_k_is_one():
    """Raise when graph has no k-core layer above 1."""
    graph = nx.path_graph(4)
    with pytest.raises(ValueError, match="k-core layers above 1"):
        _get_k_cores(
            graph=graph, node_list=list(graph.nodes()), max_k_core=4, min_seed_pct=0.1
        )


def test_build_p_step_matrix_has_zero_diagonal_and_normalized_rows(graph_from_pxl):
    """P-step matrix should have zero diagonal and row-normalized probabilities."""
    node_list = list(graph_from_pxl.raw.nodes())
    p_step = _build_p_step_matrix(g=graph_from_pxl, node_list=node_list, k=2)
    assert p_step.shape == (len(node_list), len(node_list))
    assert p_step.format == "csr"
    assert np.allclose(p_step.diagonal(), 0.0)
    row_sums = np.ravel(p_step.sum(axis=1))
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-8)


def test_find_largets_connect_component_of_k_core_downgrades_non_lcc_nodes():
    """Nodes in smaller seed components should be downgraded by one core level."""
    raw_graph = nx.Graph()
    raw_graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])  # largest seed CC
    raw_graph.add_edge("d", "e")  # smaller seed CC
    node_list = ["a", "b", "c", "d", "e"]
    k_cores = np.array([3, 3, 3, 3, 1])

    updated = _find_largets_connect_component_of_k_core(
        raw_graph=raw_graph, node_list=node_list, k_cores=k_cores.copy(), max_k=3
    )

    assert np.array_equal(updated, np.array([3, 3, 3, 2, 1]))


def test_find_partitions_sorts_thresholds_descending_and_builds_candidates():
    """Partition candidates should follow descending threshold order."""
    counts = pd.DataFrame(np.ones((3, 2)))
    k_cores = np.array([2, 1, 1])
    p_step = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.2, 0.2, 0.0],
            ]
        )
    )
    partitions = _find_partitions(
        counts=counts,
        k_cores=k_cores,
        max_k=2,
        binding_thresholds=[0.1, 0.5],
        P_step=p_step,
        max_iter=5,
        nodes_to_move_threshold=1,
    )

    assert [p.binding_threshold for p in partitions] == [0.5, 0.1]
    assert np.array_equal(partitions[0].partition, np.array([1, 1, 0]))
    assert np.array_equal(partitions[1].partition, np.array([1, 1, 1]))


def test_validate_best_candidate_prefers_valid_partition_with_highest_bc():
    """When needed, best candidate should be replaced by best valid partition."""
    initial = _PartitionCandidate(
        partition=np.array([1, 0, 0]),
        bc_score=0.9,
        nodes_pct=0.5,
        binding_threshold=0.3,
    )
    valid_low = _PartitionCandidate(
        partition=np.array([1, 1, 0]),
        bc_score=0.2,
        nodes_pct=0.8,
        binding_threshold=0.4,
    )
    valid_high = _PartitionCandidate(
        partition=np.array([1, 1, 1]),
        bc_score=0.4,
        nodes_pct=0.9,
        binding_threshold=0.5,
    )

    selected = validate_best_candidate(
        best_candidate=initial,
        partitions=[initial, valid_low, valid_high],
        min_allowed_nodes_pct=0.75,
    )
    assert selected == valid_high


def test_validate_best_candidate_returns_none_when_no_partition_is_valid(caplog):
    """Return None and log warning when no candidate meets node-percentage floor."""
    candidate = _PartitionCandidate(
        partition=np.array([1, 0, 0]),
        bc_score=0.9,
        nodes_pct=0.5,
        binding_threshold=0.3,
    )
    with caplog.at_level(logging.WARNING):
        selected = validate_best_candidate(
            best_candidate=candidate,
            partitions=[candidate],
            min_allowed_nodes_pct=0.8,
        )
    assert selected is None
    assert "Setting all nodes as 'high'" in caplog.text
