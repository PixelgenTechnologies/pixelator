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
import pytest
from scipy.sparse import csr_matrix

from pixelator.pna.analysis.segmentation import cc_protein_weights, segment_cell
from pixelator.pna.analysis.segmentation.segment import (
    _expand_adjacency_matrix,
    _kmeans_midpoint,
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


# Same 5-cell PBMC file as pixelatorR `minimal_pna_pxl_file()`.
# R loads colnames(se)[2] and assigns:
#   se$cell_type <- c("Mono", "pDC", "CD4T", "CD4T", "CD4T")
# Python obs order matches that Seurat colnames order (see
# test_distance_from_node_set.py: colnames(se)[4] == d4074c845bb62800).
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


def _segment_pbmc_r_component(
    dataset: PNAPixelDataset, weights: pd.DataFrame, **kwargs
) -> dict:
    graph = next(
        dataset.filter(components=[_R_SEGMENT_COMPONENT]).edgelist().iterator()
    ).graph
    segment_cell(graph, w=weights, verbose=False, **kwargs)
    return nx.get_node_attributes(graph.raw, "compartment")


@pytest.fixture(scope="module")
def pbmc_r_compartments(pna_pxl_dataset: PNAPixelDataset, pbmc_r_weights: pd.DataFrame):
    def run(**kwargs):
        return _segment_pbmc_r_component(pna_pxl_dataset, pbmc_r_weights, k=2, **kwargs)

    return {
        "default": run(),
        "no_interface": run(detect_interface=False),
        "expansion4": run(k_interface_expansion=4),
        "lcc": run(detect_interface=False, keep_largest_comp=True),
        "min_size": run(
            detect_interface=False, keep_largest_comp=False, min_comp_size=1
        ),
    }


def test_pbmc_r_weights_have_population_columns(pbmc_r_weights):
    assert list(pbmc_r_weights.columns) == [_R_POP1, _R_POP2]
    assert pbmc_r_weights.shape[0] >= 5
    assert (pbmc_r_weights >= 0).all().all()


def test_pbmc_r_segment_cell_returns_expected_compartments(pbmc_r_compartments):
    labels = pbmc_r_compartments["default"]
    assert set(labels.values()) <= _R_ALLOWED
    assert {_R_POP1, _R_POP2} & set(labels.values())
    assert not any(value.startswith("intra_") for value in labels.values())


def test_pbmc_r_segment_cell_without_interface_detection(pbmc_r_compartments):
    labels = pbmc_r_compartments["no_interface"]
    assert "interface" not in labels.values()
    assert set(labels.values()) <= {_R_POP1, _R_POP2, "other"}


def test_pbmc_r_segment_cell_high_interface_expansion(pbmc_r_compartments):
    labels = pbmc_r_compartments["expansion4"]
    assert set(labels.values()) <= _R_ALLOWED


def test_pbmc_r_segment_cell_applies_component_filtering(pbmc_r_compartments):
    lcc = pbmc_r_compartments["lcc"]
    min_size = pbmc_r_compartments["min_size"]
    n_non_other_lcc = sum(label in {_R_POP1, _R_POP2} for label in lcc.values())
    n_non_other_min_size = sum(
        label in {_R_POP1, _R_POP2} for label in min_size.values()
    )
    assert n_non_other_lcc <= n_non_other_min_size
    assert sum(label == "other" for label in lcc.values()) > sum(
        label == "other" for label in min_size.values()
    )
