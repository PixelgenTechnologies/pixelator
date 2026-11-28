"""Tests for `pixelator.pna.analysis.segmentation.segment_cell`.

Copyright © 2026 Pixelgen Technologies AB.

Two fixtures:

* Synthetic two-community conjugate ``PNAGraph`` (path-like bipartite
  communities, exclusive markers, two crossing edges, a disconnected
  island). This is the conjugate-correctness test from PNA-3546 / DAT-189.
* The shared five-cell PBMC ``.pxl`` (``pna_pxl_dataset``), same cells,
  labels, and second component as pixelatorR
  ``tests/testthat/test-segment_cell.R``. Those tests check the R
  contract on a real graph; they are not a biological conjugate gold
  table, and they do not commit R label CSVs.

Boundary-node disagreements vs R are expected because NNLS projection
here has no RcppML L1=0.2 penalty.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from pixelator.pna.analysis.segmentation import cc_protein_weights, segment_cell
from pixelator.pna.analysis.segmentation.segment import (
    _expand_adjacency_matrix,
    _kmeans_midpoint,
    _marker_indicator,
    _node_marker_labels,
    _project_weights,
    _retained_component_nodes,
    _shared_markers,
)
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset

POP1 = "T"
POP2 = "B"
N_PER_SIDE = 8

T_A = [f"ta{i}" for i in range(N_PER_SIDE)]
T_B = [f"tb{i}" for i in range(N_PER_SIDE)]
B_A = [f"ba{i}" for i in range(N_PER_SIDE)]
B_B = [f"bb{i}" for i in range(N_PER_SIDE)]
ISLAND_A = "xa0"
ISLAND_B = "xb0"

CROSSING = [("ta7", "bb0"), ("ba0", "tb7")]
CORE_T = {"ta0", "ta1", "tb0", "tb1"}
CORE_B = {"ba6", "ba7", "bb6", "bb7"}
CROSSING_NODES = {"ta7", "bb0", "ba0", "tb7"}


def _path_community_rows(a_nodes, b_nodes, marker_a, marker_b):
    """Bipartite path ta0-tb0-ta1-tb1-... so hop distance from the interface is defined."""
    rows = []
    for a, b in zip(a_nodes, b_nodes):
        rows.append(
            {
                "umi1": a,
                "umi2": b,
                "marker_1": marker_a,
                "marker_2": marker_b,
                "read_count": 1,
            }
        )
    for a, b in zip(a_nodes[1:], b_nodes[:-1]):
        rows.append(
            {
                "umi1": a,
                "umi2": b,
                "marker_1": marker_a,
                "marker_2": marker_b,
                "read_count": 1,
            }
        )
    return rows


def _conjugate_edgelist() -> pd.DataFrame:
    rows = []
    rows.extend(_path_community_rows(T_A, T_B, "CD3e", "CD4"))
    rows.extend(_path_community_rows(B_A, B_B, "CD20", "CD19"))
    for umi1, umi2 in CROSSING:
        marker_1 = "CD3e" if umi1.startswith("ta") else "CD20"
        marker_2 = "CD19" if umi2.startswith("bb") else "CD4"
        rows.append(
            {
                "umi1": umi1,
                "umi2": umi2,
                "marker_1": marker_1,
                "marker_2": marker_2,
                "read_count": 1,
            }
        )
    rows.append(
        {
            "umi1": ISLAND_A,
            "umi2": ISLAND_B,
            "marker_1": "CD3e",
            "marker_2": "CD4",
            "read_count": 1,
        }
    )
    return pd.DataFrame(rows)


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            POP1: [1.0, 1.0, 0.0, 0.0],
            POP2: [0.0, 0.0, 1.0, 1.0],
        },
        index=pd.Index(["CD3e", "CD4", "CD20", "CD19"], name="marker"),
    )


@pytest.fixture
def graph() -> PNAGraph:
    return PNAGraph.from_edgelist(_conjugate_edgelist())


@pytest.fixture
def weights() -> pd.DataFrame:
    return _weights()


def _compartments(graph: PNAGraph) -> dict:
    return nx.get_node_attributes(graph.raw, "compartment")


def test_segment_cell_returns_expected_compartments(graph, weights):
    result = segment_cell(graph, w=weights, verbose=False)

    assert result is graph
    labels = _compartments(graph)
    assert set(labels) == set(graph.raw.nodes())
    assert set(labels.values()) <= {POP1, POP2, "interface", "other"}
    assert {POP1, POP2} <= set(labels.values())
    assert not any(value.startswith("intra_") for value in labels.values())


def test_segment_cell_core_communities_are_stable(graph, weights):
    """Core nodes away from the two crossing edges keep their community labels."""
    segment_cell(graph, w=weights, verbose=False)
    labels = _compartments(graph)

    assert all(labels[node] == POP1 for node in CORE_T)
    assert all(labels[node] == POP2 for node in CORE_B)
    assert all(labels[node] == "interface" for node in CROSSING_NODES)
    assert labels["tb6"] == POP1
    assert labels[ISLAND_A] == "other"
    assert labels[ISLAND_B] == "other"


def test_segment_cell_without_interface_detection_has_no_interface_labels(
    graph, weights
):
    segment_cell(graph, w=weights, detect_interface=False, verbose=False)
    labels = _compartments(graph)

    assert "interface" not in labels.values()
    assert set(labels.values()) <= {POP1, POP2, "other"}
    assert all(
        labels[node] == POP1 for node in CORE_T | CROSSING_NODES if node.startswith("t")
    )
    assert all(labels[node] == POP2 for node in CORE_B)


def test_segment_cell_high_interface_expansion_does_not_error(graph, weights):
    segment_cell(graph, w=weights, k_interface_expansion=4, verbose=False)
    labels = _compartments(graph)
    assert set(labels.values()) <= {POP1, POP2, "interface", "other"}
    assert labels[ISLAND_A] == "other"


def test_segment_cell_applies_component_filtering(graph, weights):
    out_lcc = PNAGraph.from_edgelist(_conjugate_edgelist())
    segment_cell(
        out_lcc,
        w=weights,
        detect_interface=False,
        keep_largest_comp=True,
        verbose=False,
    )
    out_min_size = PNAGraph.from_edgelist(_conjugate_edgelist())
    segment_cell(
        out_min_size,
        w=weights,
        detect_interface=False,
        keep_largest_comp=False,
        min_comp_size=1,
        verbose=False,
    )

    lcc = _compartments(out_lcc)
    min_size = _compartments(out_min_size)
    n_non_other_lcc = sum(label in {POP1, POP2} for label in lcc.values())
    n_non_other_min_size = sum(label in {POP1, POP2} for label in min_size.values())

    assert n_non_other_lcc <= n_non_other_min_size
    assert sum(label == "other" for label in lcc.values()) > sum(
        label == "other" for label in min_size.values()
    )
    assert min_size[ISLAND_A] == POP1
    assert lcc[ISLAND_A] == "other"


def test_retained_component_nodes_filters_when_every_node_is_kept(graph):
    node_order = list(graph.raw.nodes())
    keep_mask = np.ones(len(node_order), dtype=bool)

    adjacency = graph.get_adjacency_sparse(node_ordering=node_order).tocsr()
    lcc = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=keep_mask,
        keep_largest_comp=True,
        min_comp_size=10,
    )
    assert ISLAND_A not in lcc
    assert ISLAND_B not in lcc
    assert "ta0" in lcc

    min_size = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=keep_mask,
        keep_largest_comp=False,
        min_comp_size=10,
    )
    assert ISLAND_A not in min_size
    assert "ta0" in min_size


def test_retained_component_tie_breaks_toward_the_earlier_node():
    """Two components of equal size: keep the one that starts earlier."""
    node_order = ["a", "b", "c", "d"]
    adjacency = csr_matrix(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    kept = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=np.ones(4, dtype=bool),
        keep_largest_comp=True,
        min_comp_size=1,
    )
    assert kept == ["a", "b"]

    both = _retained_component_nodes(
        adjacency=adjacency,
        node_order=node_order,
        keep_mask=np.array([False, False, True, True]),
        keep_largest_comp=False,
        min_comp_size=2,
    )
    assert both == ["c", "d"]


def test_segment_cell_interface_expansion_one_hop(graph, weights):
    segment_cell(graph, w=weights, k_interface_expansion=1, verbose=False)
    labels = _compartments(graph)

    assert labels["ta7"] == "interface"
    # Direct neighbors of a crossing node are pulled into the interface.
    # Path: ...-ta6-tb6-ta7-tb7, so tb6 is 1 hop from ta7 and ta6 is 2 hops.
    assert labels["tb6"] == "interface"
    assert labels["ta6"] == POP1
    # Far from the bridge, community cores stay cell-type labels.
    assert labels["ta0"] == POP1
    assert labels["ba7"] == POP2
    # The disconnected island is unchanged.
    assert labels[ISLAND_A] == "other"
    assert labels[ISLAND_B] == "other"


def test_segment_cell_is_deterministic_with_seed(weights):
    first = PNAGraph.from_edgelist(_conjugate_edgelist())
    second = PNAGraph.from_edgelist(_conjugate_edgelist())
    segment_cell(first, w=weights, random_state=0, verbose=False)
    segment_cell(second, w=weights, random_state=0, verbose=False)
    assert _compartments(first) == _compartments(second)


def test_segment_cell_replaces_existing_compartment_attribute(graph, weights):
    nx.set_node_attributes(graph.raw, "stale", "compartment")
    segment_cell(graph, w=weights, verbose=False)
    labels = _compartments(graph)
    assert "stale" not in labels.values()
    assert labels["ta0"] == POP1


def test_segment_cell_zero_smoothing_still_classifies(graph, weights):
    segment_cell(graph, w=weights, spatial_smoothing_iter=0, verbose=False)
    labels = _compartments(graph)
    assert all(labels[node] == POP1 for node in CORE_T)
    assert all(labels[node] == POP2 for node in CORE_B)


def test_neighborhood_counts_match_the_dense_marker_table(graph, weights):
    """The one-marker indicator sums to the same neighborhood profiles."""
    node_order = list(graph.raw.nodes())
    marker_labels = _node_marker_labels(graph, node_order)
    shared = _shared_markers(marker_labels, weights.index)
    counts = graph.node_marker_counts.reindex(node_order).fillna(0)
    dense_markers = [marker for marker in counts.columns if marker in weights.index]

    adjacency = graph.get_adjacency_sparse(node_ordering=node_order).tocsr()
    expanded = _expand_adjacency_matrix(adjacency, k=2)
    expanded.setdiag(1)
    from_indicator = expanded @ _marker_indicator(marker_labels, shared)
    from_table = expanded @ counts.loc[:, dense_markers].to_numpy(dtype=float)

    assert shared == dense_markers
    np.testing.assert_allclose(from_indicator.toarray(), from_table)


def test_segment_cell_missing_w_proteins_raises(graph):
    w = pd.DataFrame({"T": [1.0], "B": [0.0]}, index=["not_a_marker"])
    with pytest.raises(ValueError, match="No proteins in w"):
        segment_cell(graph, w=w, verbose=False)


def test_segment_cell_wrong_graph_type_raises(weights):
    with pytest.raises(TypeError, match="graph must be a PNAGraph"):
        segment_cell("not a graph", w=weights)


def test_segment_cell_invalid_w_raises(graph, weights):
    with pytest.raises(TypeError, match="w must be a pandas DataFrame"):
        segment_cell(graph, w="Invalid")
    with pytest.raises(ValueError, match="exactly two columns"):
        segment_cell(graph, w=weights.iloc[:, :1])


def test_segment_cell_invalid_params_raise(graph, weights):
    with pytest.raises(TypeError, match="k must be an int"):
        segment_cell(graph, w=weights, k="Invalid")
    with pytest.raises(ValueError, match="k must be >= 1"):
        segment_cell(graph, w=weights, k=0)
    with pytest.raises(ValueError, match="k must be <= 6"):
        segment_cell(graph, w=weights, k=7)
    with pytest.raises(TypeError, match="detect_interface must be a bool"):
        segment_cell(graph, w=weights, detect_interface="Invalid")
    with pytest.raises(TypeError, match="k_interface_expansion must be an int"):
        segment_cell(graph, w=weights, k_interface_expansion="Invalid")
    with pytest.raises(TypeError, match="keep_largest_comp must be a bool"):
        segment_cell(graph, w=weights, keep_largest_comp="Invalid")
    with pytest.raises(TypeError, match="min_comp_size must be an int"):
        segment_cell(graph, w=weights, min_comp_size="Invalid")
    with pytest.raises(TypeError, match="verbose must be a bool"):
        segment_cell(graph, w=weights, verbose="Invalid")


def test_segment_cell_numpy_integer_params(graph, weights):
    segment_cell(
        graph,
        w=weights,
        k=np.int64(2),
        k_interface_expansion=np.int64(0),
        min_comp_size=np.int64(10),
        spatial_smoothing_iter=np.int64(1),
        verbose=False,
    )
    assert _compartments(graph)["ta0"] == POP1


def test_segment_cell_verbose_logs(graph, weights, caplog):
    with caplog.at_level("INFO"):
        segment_cell(graph, w=weights, verbose=True)
    assert any("NNLS" in rec.message for rec in caplog.records)


def test_segment_cell_exported_from_analysis():
    from pixelator.pna.analysis import segment_cell as exported

    assert exported is segment_cell
    assert inspect.signature(segment_cell).parameters["k"].default == 2
    assert (
        inspect.signature(segment_cell).parameters["detect_interface"].default is True
    )


def _row_normalized_scores(scores: np.ndarray) -> np.ndarray:
    adjusted = scores + 1e-8
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def test_segment_cell_ignores_a_shared_weight_scale(graph, weights):
    """Multiplying both columns by the same factor does not change labels."""
    segment_cell(graph, w=weights, verbose=False)
    baseline = _compartments(graph)

    rescaled = PNAGraph.from_edgelist(_conjugate_edgelist())
    segment_cell(rescaled, w=weights * 7, verbose=False)
    assert _compartments(rescaled) == baseline


def _positive_least_squares(profiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(weights, profiles.T)
    return np.asarray(model.coef_, dtype=float)


def test_project_weights_matches_positive_least_squares():
    """The vectorized two-coefficient solve matches per-node NNLS."""
    rng = np.random.default_rng(0)
    profiles = rng.random((200, 12))
    profiles /= profiles.sum(axis=1, keepdims=True)
    weights = rng.random((12, 2))
    np.testing.assert_allclose(
        _project_weights(profiles, weights),
        _positive_least_squares(profiles, weights),
        rtol=1e-8,
        atol=1e-8,
    )

    exclusive = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.6],
            [0.2, 0.2, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    exclusive_weights = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        _project_weights(exclusive, exclusive_weights),
        _positive_least_squares(exclusive, exclusive_weights),
        rtol=1e-8,
        atol=1e-8,
    )

    duplicated = np.column_stack([weights[:, 0], weights[:, 0]])
    np.testing.assert_allclose(
        _project_weights(profiles, duplicated),
        _positive_least_squares(profiles, duplicated),
        rtol=1e-8,
        atol=1e-8,
    )
    zero_column = weights.copy()
    zero_column[:, 1] = 0.0
    np.testing.assert_allclose(
        _project_weights(profiles, zero_column),
        _positive_least_squares(profiles, zero_column),
        rtol=1e-8,
        atol=1e-8,
    )
    assert _project_weights(np.zeros((0, 12)), weights).shape == (0, 2)


def test_projection_keeps_a_shared_scale_but_not_per_column_scales():
    """Row-normalized scores cancel one scale, not a different scale per column.

    ``cc_protein_weights`` therefore scales each column to sum to 1 before
    returning it.
    """
    profiles = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    raw_weights = np.array([[8.0, 0.2], [0.5, 0.2], [0.2, 1.5]])
    raw = _row_normalized_scores(_project_weights(profiles, raw_weights))
    shared = _row_normalized_scores(_project_weights(profiles, raw_weights * 4))
    per_column = raw_weights / raw_weights.sum(axis=0, keepdims=True)
    rescaled = _row_normalized_scores(_project_weights(profiles, per_column))

    np.testing.assert_allclose(shared, raw, atol=1e-6)
    assert not np.allclose(rescaled, raw, atol=1e-3)


def test_kmeans_midpoint_is_center_average():
    assert _kmeans_midpoint(
        np.array([0.2, 0.2, 0.8, 0.8]), random_state=0
    ) == pytest.approx(0.5)
    assert _kmeans_midpoint(
        np.array([0.1, 0.1, 0.3, 0.3]), random_state=0
    ) == pytest.approx(0.2)


def test_expand_adjacency_matrix_k1_and_k2_on_a_line():
    # a -- b -- c
    adjacency = csr_matrix(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )
    one_hop = _expand_adjacency_matrix(adjacency, k=1).toarray()
    np.testing.assert_array_equal(one_hop, adjacency.toarray())

    two_hop = _expand_adjacency_matrix(adjacency, k=2).toarray()
    # R zeros the diagonal after expansion, so a reaches c but not itself.
    assert two_hop[0, 2] == 1
    assert two_hop[2, 0] == 1
    assert two_hop[0, 0] == 0
    assert two_hop[0, 1] == 1


def _segment_graph(graph: PNAGraph, weights: pd.DataFrame, **kwargs) -> dict:
    """Segment the graph and return the compartment label of each node."""
    segment_cell(graph, w=weights, verbose=False, **kwargs)
    return nx.get_node_attributes(graph.raw, "compartment")


def _segment_component(
    dataset: PNAPixelDataset, weights: pd.DataFrame, component: str, **kwargs
) -> dict:
    """Segment one component of the dataset and return the node labels."""
    graph = next(
        iter(dataset.filter(components=[component]).edgelist().iterator())
    ).graph
    return _segment_graph(graph, weights, **kwargs)


# --- Synthetic dataset ------------------------------------------------------
#
# `synthetic_analysis_pxl_dataset` has four connected cells of two cell types.
# Component filtering (`keep_largest_comp`, `min_comp_size`) only has an effect
# on a graph with several connected pieces, so a two-node island is attached to
# the segmented component. Its nodes get umis that no generated node can have
# (generated umis are below 2**56).
_SYN_POP1 = "A"
_SYN_POP2 = "B"
_SYN_CELL_TYPES = (_SYN_POP1, _SYN_POP2, _SYN_POP1, _SYN_POP2)  # one per cell
_SYN_WEIGHTS_SEED = 7331
_SYN_SEGMENT_COMPONENT = "0000000000000001"
_SYN_ISLAND_NODES = (2**60, 2**60 + 1)
_SYN_ALLOWED_LABELS = {_SYN_POP1, _SYN_POP2, "interface", "other"}


def _segment_component_with_island(
    dataset: PNAPixelDataset, weights: pd.DataFrame, component: str, **kwargs
) -> dict:
    """Segment a component that also has a two-node island, return node labels."""
    edgelist = (
        dataset.filter(components=[component])
        .edgelist()
        .to_polars()
        .select("umi1", "umi2", "marker_1", "marker_2", "read_count")
    )
    # One extra edge, linked to nothing else, copying the markers of an existing edge.
    island_umi1, island_umi2 = _SYN_ISLAND_NODES
    island_edge = edgelist.head(1).with_columns(
        umi1=pl.lit(island_umi1, dtype=edgelist.schema["umi1"]),
        umi2=pl.lit(island_umi2, dtype=edgelist.schema["umi2"]),
    )
    graph = PNAGraph.from_edgelist(pl.concat([edgelist, island_edge]).lazy())
    return _segment_graph(graph, weights, **kwargs)


@pytest.fixture(scope="module")
def synthetic_weights(synthetic_analysis_pxl_dataset: PNAPixelDataset) -> pd.DataFrame:
    """Marker weights telling the two synthetic cell types apart."""
    adata = synthetic_analysis_pxl_dataset.adata().copy()
    adata.obs["cell_type"] = list(_SYN_CELL_TYPES)
    return cc_protein_weights(
        adata,
        group_by="cell_type",
        population_1=_SYN_POP1,
        population_2=_SYN_POP2,
        random_state=_SYN_WEIGHTS_SEED,
        # The generator boosts a block of markers for each cell type but never
        # removes the others, so no marker is almost absent from a population,
        # which the default `max_freq` of 0.01 would require.
        max_freq=1.0,
        verbose=False,
    )


@pytest.fixture(scope="module")
def synthetic_labels(
    synthetic_analysis_pxl_dataset: PNAPixelDataset, synthetic_weights: pd.DataFrame
) -> dict[str, dict]:
    """Node labels of the segmented synthetic component, for several options.

    Segmenting is the slow part, so each option set is computed once and shared
    by the tests below.
    """

    def segment(**kwargs):
        return _segment_component_with_island(
            synthetic_analysis_pxl_dataset,
            synthetic_weights,
            _SYN_SEGMENT_COMPONENT,
            k=2,
            **kwargs,
        )

    return {
        "default": segment(),
        "no_interface": segment(detect_interface=False),
        "large_interface_expansion": segment(k_interface_expansion=4),
        "largest_component_only": segment(
            detect_interface=False, keep_largest_comp=True
        ),
        "all_components": segment(
            detect_interface=False, keep_largest_comp=False, min_comp_size=1
        ),
    }


def test_synthetic_weights_have_population_columns(synthetic_weights):
    """There is one non-negative weight column per population."""
    assert list(synthetic_weights.columns) == [_SYN_POP1, _SYN_POP2]
    assert synthetic_weights.shape[0] >= 5
    assert (synthetic_weights >= 0).all().all()


def test_synthetic_segment_cell_returns_expected_compartments(synthetic_labels):
    """By default nodes are labelled with a population, the interface or other."""
    labels = set(synthetic_labels["default"].values())
    assert labels <= _SYN_ALLOWED_LABELS
    assert labels & {_SYN_POP1, _SYN_POP2}
    assert not any(label.startswith("intra_") for label in labels)


def test_synthetic_segment_cell_without_interface_detection(synthetic_labels):
    """Disabling interface detection leaves no node labelled as interface."""
    labels = set(synthetic_labels["no_interface"].values())
    assert "interface" not in labels
    assert labels <= {_SYN_POP1, _SYN_POP2, "other"}


def test_synthetic_segment_cell_high_interface_expansion(synthetic_labels):
    """A large interface expansion still only produces the allowed labels."""
    labels = set(synthetic_labels["large_interface_expansion"].values())
    assert labels <= _SYN_ALLOWED_LABELS


def test_synthetic_segment_cell_applies_component_filtering(synthetic_labels):
    """The island is dropped with `keep_largest_comp` and kept otherwise.

    "other" is the label of nodes that are filtered out.
    """
    largest_only = synthetic_labels["largest_component_only"]
    all_components = synthetic_labels["all_components"]
    for node in _SYN_ISLAND_NODES:
        assert largest_only[node] == "other"
        assert all_components[node] in {_SYN_POP1, _SYN_POP2}


# Same 5-cell PBMC file as pixelatorR `minimal_pna_pxl_file()`.
# R loads colnames(se)[2] and assigns:
#   se$cell_type <- c("Mono", "pDC", "CD4T", "CD4T", "CD4T")
# Python obs order matches that Seurat colnames order (see
# test_distance_from_node_set.py: colnames(se)[4] == d4074c845bb62800).
# The full-size file is slow, so only the default compartments are checked
# against it; the other options are covered with the synthetic dataset above.
_R_CELL_TYPES = ("Mono", "pDC", "CD4T", "CD4T", "CD4T")
_R_POP1 = "Mono"
_R_POP2 = "CD4T"
_R_WEIGHTS_SEED = 7331
_R_SEGMENT_COMPONENT = "2708240b908e2eba"
_R_ALLOWED = {_R_POP1, _R_POP2, "interface", "other"}


@pytest.fixture(scope="module")
def pbmc_r_weights(pna_pxl_dataset: PNAPixelDataset) -> pd.DataFrame:
    adata = pna_pxl_dataset.adata().copy()
    assert list(adata.obs.index) == [
        "0a45497c6bfbfb22",
        _R_SEGMENT_COMPONENT,
        "c3c393e9a17c1981",
        "d4074c845bb62800",
        "efe0ed189cb499fc",
    ]
    adata.obs["cell_type"] = list(_R_CELL_TYPES)
    return cc_protein_weights(
        adata,
        group_by="cell_type",
        population_1=_R_POP1,
        population_2=_R_POP2,
        random_state=_R_WEIGHTS_SEED,
        verbose=False,
    )


@pytest.mark.slow
def test_pbmc_r_weights_have_population_columns(pbmc_r_weights):
    assert list(pbmc_r_weights.columns) == [_R_POP1, _R_POP2]
    assert pbmc_r_weights.shape[0] >= 5
    assert (pbmc_r_weights >= 0).all().all()


@pytest.mark.slow
def test_pbmc_r_segment_cell_returns_expected_compartments(
    pna_pxl_dataset: PNAPixelDataset, pbmc_r_weights: pd.DataFrame
):
    labels = _segment_component(
        pna_pxl_dataset, pbmc_r_weights, _R_SEGMENT_COMPONENT, k=2
    )
    assert set(labels.values()) <= _R_ALLOWED
    assert {_R_POP1, _R_POP2} & set(labels.values())
    assert not any(value.startswith("intra_") for value in labels.values())
