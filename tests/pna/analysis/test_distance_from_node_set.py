"""Tests for `pixelator.pna.analysis.segmentation.distance_from_node_set`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import inspect
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pytest

from pixelator.pna.analysis.segmentation import distance_from_node_set
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset

# Bipartite line a-b-c-d-e plus a disconnected edge f-g:
#   a -- b -- c -- d -- e    f -- g
_EDGES = {
    "umi1": ["a", "c", "c", "e", "f"],
    "umi2": ["b", "b", "d", "d", "g"],
    "marker_1": ["MA", "MC", "MC", "ME", "MF"],
    "marker_2": ["MB", "MB", "MD", "MD", "MG"],
    "read_count": [1, 1, 1, 1, 1],
}


def _synthetic_graph() -> PNAGraph:
    return PNAGraph.from_edgelist(pl.DataFrame(_EDGES).lazy())


def _distances(graph: PNAGraph) -> dict:
    return nx.get_node_attributes(graph.raw, "distance_from_seed")


@pytest.fixture
def graph() -> PNAGraph:
    return _synthetic_graph()


def test_distance_from_node_set_single_seed_exact_hops(graph):
    result = distance_from_node_set(graph, "a")

    assert result is graph
    distances = _distances(graph)
    assert distances == {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": None,
        "g": None,
    }
    assert all(isinstance(distances[n], int) for n in "abcde")


def test_distance_from_node_set_multiple_seeds(graph):
    distance_from_node_set(graph, ["a", "e"])
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["e"] == 0
    assert distances["b"] == 1
    assert distances["d"] == 1
    assert distances["c"] == 2
    assert distances["f"] is None
    assert distances["g"] is None
    assert sum(d == 0 for d in distances.values() if d is not None) == 2


def test_distance_from_node_set_respects_max_iter(graph):
    distance_from_node_set(graph, "a", max_iter=2)
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["b"] == 1
    assert distances["c"] == 2
    assert distances["d"] is None
    assert distances["e"] is None
    assert distances["f"] is None
    assert distances["g"] is None
    reached = [d for d in distances.values() if d is not None]
    assert max(reached) == 2


def test_distance_from_node_set_max_iter_zero(graph):
    distance_from_node_set(graph, ["a", "c"], max_iter=0)
    distances = _distances(graph)

    assert distances["a"] == 0
    assert distances["c"] == 0
    assert all(distances[n] is None for n in "bdefg")


def test_distance_from_node_set_replaces_existing_attribute(graph):
    nx.set_node_attributes(graph.raw, 999, "distance_from_seed")
    distance_from_node_set(graph, "a")
    distances = _distances(graph)

    assert 999 not in distances.values()
    assert distances["a"] == 0
    assert distances["f"] is None


def test_distance_from_node_set_missing_seed_raises(graph):
    with pytest.raises(ValueError, match="seed nodes must be present"):
        distance_from_node_set(graph, "not_a_real_node")


def test_distance_from_node_set_partially_missing_seeds_raises(graph):
    with pytest.raises(ValueError, match="seed nodes must be present"):
        distance_from_node_set(graph, ["a", "missing"])


def test_distance_from_node_set_empty_seeds_raises(graph):
    with pytest.raises(ValueError, match="at least one node"):
        distance_from_node_set(graph, [])


def test_distance_from_node_set_invalid_graph_raises(graph):
    with pytest.raises(TypeError, match="PNAGraph"):
        distance_from_node_set("not a graph", "a")


def test_distance_from_node_set_invalid_max_iter_raises(graph):
    with pytest.raises(TypeError, match="max_iter"):
        distance_from_node_set(graph, "a", max_iter=1.5)
    with pytest.raises(ValueError, match="max_iter"):
        distance_from_node_set(graph, "a", max_iter=-1)


def test_distance_from_node_set_numpy_integer_max_iter(graph):
    distance_from_node_set(graph, "a", max_iter=np.int64(2))
    assert _distances(graph)["c"] == 2
    assert _distances(graph)["d"] is None


def test_distance_from_node_set_verbose_logs(graph, caplog):
    with caplog.at_level("INFO"):
        distance_from_node_set(graph, "a", max_iter=2, verbose=True)

    assert any("Iteration" in rec.message for rec in caplog.records)


def test_distance_from_node_set_exported_from_analysis():
    from pixelator.pna.analysis import distance_from_node_set as exported

    assert exported is distance_from_node_set
    assert (
        inspect.signature(distance_from_node_set).parameters["max_iter"].default == 40
    )


# Five-cell PBMC fixture (`pna_pxl_dataset` / PNA055_Sample07_S7.layout.pxl).
# Same component and seed as pixelatorR tests/testthat/test-distance_from_node_set.R
# (colnames(se)[4] and the first node of that cell graph).
_PBMC_COMPONENT_ID = "d4074c845bb62800"
_PBMC_SEED_NODE = 55840301536286099
_PBMC_DISTANCE_COUNTS_0_TO_9 = (1, 8, 37, 188, 404, 1202, 1808, 4191, 4551, 9130)
_PBMC_NODE_DISTANCES = {
    55840301536286099: 0,
    3550955672846941: 1,
    20167079200998091: 10,
    33849376607686952: 11,
    56704973007062051: 12,
    70796622900050774: 11,
    57946909517865241: 8,
    50521531064131258: 9,
    15898272712678452: 10,
    55203010621091460: 11,
}


@pytest.fixture(scope="module")
def pbmc_component_graph(pna_pxl_dataset: PNAPixelDataset) -> PNAGraph:
    component = next(
        pna_pxl_dataset.filter(components=[_PBMC_COMPONENT_ID]).edgelist().iterator()
    )
    graph = component.graph
    distance_from_node_set(graph, _PBMC_SEED_NODE)
    return graph


@pytest.fixture(scope="module")
def pbmc_distances(pbmc_component_graph: PNAGraph) -> dict:
    return _distances(pbmc_component_graph)


@pytest.fixture(scope="module")
def pbmc_layout_with_distance(
    pna_pxl_dataset: PNAPixelDataset, pbmc_distances: dict
) -> pd.DataFrame:
    """Stored wpmds_3d coordinates joined to hop distance (R plots this too)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xyz = (
            pna_pxl_dataset.filter(components=[_PBMC_COMPONENT_ID])
            .precomputed_layouts(add_marker_counts=False)
            .to_df()
        )
    xyz = xyz.copy()
    xyz["distance_from_seed"] = xyz["index"].map(pbmc_distances)
    return xyz


def test_distance_from_node_set_on_pbmc_component(pbmc_distances):
    """Hop distances on the five-cell PBMC graph match the R test exactly."""
    assert pbmc_distances[_PBMC_SEED_NODE] == 0
    assert sum(d == 0 for d in pbmc_distances.values()) == 1
    assert None not in pbmc_distances.values()

    counts = np.bincount(
        [d for d in pbmc_distances.values() if d is not None],
        minlength=10,
    )
    assert tuple(int(v) for v in counts[:10]) == _PBMC_DISTANCE_COUNTS_0_TO_9

    for node, expected in _PBMC_NODE_DISTANCES.items():
        assert pbmc_distances[node] == expected


def test_distance_from_node_set_increases_away_from_seed_in_layout(
    pbmc_layout_with_distance,
):
    """Nodes near the seed in 3D layout should have smaller hop distance."""
    xyz = pbmc_layout_with_distance
    assert xyz["distance_from_seed"].notna().all()

    seed = xyz.loc[xyz["index"] == _PBMC_SEED_NODE].iloc[0]
    euclidean = np.sqrt(
        (xyz["x"] - seed.x) ** 2 + (xyz["y"] - seed.y) ** 2 + (xyz["z"] - seed.z) ** 2
    )
    spearman = (
        pd.DataFrame(
            {
                "distance_from_seed": xyz["distance_from_seed"],
                "euclidean": euclidean,
            }
        )
        .corr(method="spearman")
        .iloc[0, 1]
    )
    nearest = xyz.assign(euclidean=euclidean).nsmallest(100, "euclidean")
    farthest = xyz.assign(euclidean=euclidean).nlargest(100, "euclidean")

    assert spearman > 0.7
    assert nearest["distance_from_seed"].mean() < farthest["distance_from_seed"].mean()


def _distance_layout_figure(xyz: pd.DataFrame) -> plt.Figure:
    seed = xyz.loc[xyz["index"] == _PBMC_SEED_NODE]
    fig = plt.figure(figsize=(6.5, 5.5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        xyz["x"],
        xyz["y"],
        xyz["z"],
        c=xyz["distance_from_seed"],
        cmap="viridis",
        s=1.5,
        linewidths=0,
        alpha=0.4,
        depthshade=False,
    )
    ax.scatter(
        seed["x"],
        seed["y"],
        seed["z"],
        c="red",
        s=50,
        linewidths=0,
        depthshade=False,
        zorder=10,
    )
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(scatter, ax=ax, label="hop distance", shrink=0.7, pad=0.1)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.98)
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="./snapshots/test_distance_from_node_set",
)
def test_distance_from_node_set_3d_layout_plot(pbmc_layout_with_distance):
    """3D layout colored by hop distance; the red point is the seed."""
    return _distance_layout_figure(pbmc_layout_with_distance)
