"""Tests for PLS-based graph denoising.

Copyright © 2026 Pixelgen Technologies AB.
"""

from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pytest

import pixelator.common.graph.node_pls as node_pls_mod
from pixelator.pna.analysis import denoise as denoise_mod
from pixelator.pna.analysis.denoise import (
    DenoiseGraph,
    _nodes_outside_largest_cc_after_pls_scores,
    _pixel_type_design_matrix,
    denoise_pls,
)
from pixelator.pna.graph import PNAGraph


def _dense_bipartite_edgelist() -> pl.LazyFrame:
    """Build a connected bipartite edgelist with enough markers for PLS."""
    rows = []
    umis_a = [f"a{i}" for i in range(10)]
    umis_b = [f"b{i}" for i in range(10)]
    markers = [f"M{k}" for k in range(8)]
    for i, a in enumerate(umis_a):
        for j, b in enumerate(umis_b):
            if (i + j) % 3 != 0:
                continue
            rows.append(
                {
                    "umi1": a,
                    "umi2": b,
                    "marker_1": markers[(i + j) % 8],
                    "marker_2": markers[(i + j + 2) % 8],
                    "read_count": 1,
                }
            )
    return pl.DataFrame(rows).lazy()


@pytest.fixture
def pls_sized_pna_graph() -> PNAGraph:
    return PNAGraph.from_edgelist(_dense_bipartite_edgelist())


def test_nodes_outside_largest_cc_after_pls_scores_path():
    raw = nx.path_graph(4)
    idx = pd.Index(list(raw.nodes))
    passing = np.array([True, True, False, False])
    removed = _nodes_outside_largest_cc_after_pls_scores(raw, idx, passing)
    assert set(removed) == {2, 3}


def test_nodes_outside_largest_cc_all_passing_returns_empty():
    raw = nx.cycle_graph(5)
    idx = pd.Index(list(raw.nodes))
    passing = np.ones(len(idx), dtype=bool)
    removed = _nodes_outside_largest_cc_after_pls_scores(raw, idx, passing)
    assert removed == []


def test_nodes_outside_largest_cc_no_passers_removes_all():
    raw = nx.star_graph(4)
    idx = pd.Index(list(raw.nodes))
    passing = np.zeros(len(idx), dtype=bool)
    removed = _nodes_outside_largest_cc_after_pls_scores(raw, idx, passing)
    assert set(removed) == set(raw.nodes)


def test_pixel_type_design_matrix_matches_node_order(pls_sized_pna_graph):
    idx = pls_sized_pna_graph.node_marker_counts.index
    mat = _pixel_type_design_matrix(pls_sized_pna_graph, idx)
    assert mat.shape == (len(idx), 2)
    assert np.all(mat[:, 0] == 1.0)
    pt = nx.get_node_attributes(pls_sized_pna_graph.raw, "pixel_type")
    for i, node_id in enumerate(idx):
        assert mat[i, 1] == (1.0 if pt.get(node_id) == "B" else 0.0)


def test_denoise_pls_no_removals_when_no_components_selected(pls_sized_pna_graph):
    with mock.patch.object(denoise_mod, "pearsonr", return_value=(0.0, 0.5)):
        out = denoise_pls(
            pls_sized_pna_graph,
            min_pls_coreness_correlation=0.0,
            pls_component_p_threshold=0.01,
        )
    assert out == []


def test_denoise_pls_create_matrix_uses_model_k_and_pred_k(pls_sized_pna_graph):
    ks: list[int] = []
    _create_real = node_pls_mod._create_node_neighborhood_abundance_matrix

    def _capture(cg, k, *args, **kwargs):
        ks.append(k)
        return _create_real(cg, k, *args, **kwargs)

    with (
        mock.patch.object(
            node_pls_mod,
            "_create_node_neighborhood_abundance_matrix",
            side_effect=_capture,
        ),
        mock.patch.object(
            denoise_mod,
            "_create_node_neighborhood_abundance_matrix",
            side_effect=_capture,
        ),
    ):
        denoise_pls(
            pls_sized_pna_graph,
            model_k=2,
            pred_k=1,
            pls_component_p_threshold=1e-9,
            min_pls_coreness_correlation=-1.0,
            pls_score_threshold=-1e9,
        )
    assert 2 in ks and 1 in ks
    assert ks.index(2) < ks.index(1)


def test_denoise_graph_accepts_pls_only():
    g = DenoiseGraph(run_one_core=False, run_ace=False, run_pls=True)
    assert g.run_pls and not g.run_one_core and not g.run_ace


def test_denoise_graph_requires_at_least_one_mode():
    with pytest.raises(ValueError, match="At least one"):
        DenoiseGraph(run_one_core=False, run_ace=False, run_pls=False)
