"""Tests for component denoising functions.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pixelator.pna.analysis.denoise import (
    DenoiseGraph,
    denoise_ace,
    denoise_one_core_layer,
    denoise_pls,
    get_overexpressed_markers_in_one_core,
    get_stranded_nodes,
)
from pixelator.pna.analysis_engine import AnalysisManager
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PixelDatasetSaver, read
from tests.common.data_generator import write_pna_pxl


def test_get_overexpressed_markers_in_one_core_detects_enriched_marker():
    """A marker enriched in the one-core layer is detected with the inflated excess.

    Uses hand-built marker counts so the expected output can be derived directly
    from the documented formula, independent of any graph topology.
    """
    # Six nodes: three in core 1 (the periphery), three in higher cores.
    node_core_numbers = pd.Series(
        {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2},
        dtype=int,
    )
    # ``enriched`` is concentrated in the one-core nodes, ``balanced`` is spread
    # evenly across both layers so it should not be flagged.
    node_marker_counts = pd.DataFrame(
        {
            "enriched": [20, 20, 20, 0, 0, 0],
            "balanced": [10, 10, 10, 10, 10, 10],
        },
        index=[0, 1, 2, 3, 4, 5],
    )

    result = get_overexpressed_markers_in_one_core(
        node_marker_counts=node_marker_counts,
        node_core_numbers=node_core_numbers,
        inflate_factor=1.5,
    )

    # ``balanced`` is proportional across layers, so only ``enriched`` is flagged.
    assert list(result["name"]) == ["enriched"]

    # Reconstruct the documented excess: counts aggregated per core layer.
    one_core = node_marker_counts.loc[[0, 1, 2]].sum()
    higher_core = node_marker_counts.loc[[3, 4, 5]].sum()
    total_one_core = one_core.sum()
    total_higher_core = higher_core.sum()
    expected_count = np.round(
        total_one_core * higher_core["enriched"] / total_higher_core
    )
    expected_excess = int(np.ceil(1.5 * (one_core["enriched"] - expected_count)))
    assert int(result.loc[result["name"] == "enriched", "count"].iloc[0]) == (
        expected_excess
    )


def test_get_overexpressed_markers_in_one_core_synthetic_structure(
    synthetic_denoise_pxl_dataset,
):
    """get_overexpressed_markers_in_one_core returns well-formed, stable output.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
    """
    panel_markers = set(
        load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0").markers
    )
    components = synthetic_denoise_pxl_dataset.adata().obs.index

    def over_expressed_markers_per_component():
        for comp in components:
            comp_graph = PNAGraph.from_edgelist(
                synthetic_denoise_pxl_dataset.filter(components=[comp])
                .edgelist()
                .to_polars()
                .lazy()
            )
            node_marker_counts = comp_graph.node_marker_counts
            node_core_numbers = pd.Series(nx.core_number(comp_graph.raw))
            yield get_overexpressed_markers_in_one_core(
                node_marker_counts=node_marker_counts,
                node_core_numbers=node_core_numbers,
            )

    first = pd.concat(over_expressed_markers_per_component()).reset_index(drop=True)
    second = pd.concat(over_expressed_markers_per_component()).reset_index(drop=True)

    assert list(first.columns) == ["name", "count"]
    assert not first.empty
    assert (first["count"] > 0).all()
    assert first["count"].map(lambda c: float(c).is_integer()).all()
    assert set(first["name"]).issubset(panel_markers)
    # The computation is deterministic for a fixed input.
    assert_frame_equal(first, second)


def test_get_stranded_nodes():
    """Test the get_stranded_nodes function."""
    # Testing with a simple stranded node
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 5)])
    g.add_node(
        6
    )  # A stranded node, i.e. disconnected from the largest connected component

    pna_g = PNAGraph.from_raw(g)
    stranded_nodes = get_stranded_nodes(pna_g)
    assert set(stranded_nodes) == {6}

    # Testing after removing a node
    stranded_nodes_post_removal = get_stranded_nodes(
        pna_g, nodes_to_remove=[4]
    )  # 5 becomes stranded after removing 4
    assert set(stranded_nodes_post_removal) == {6, 5}

    # Testing with a fully connected graph
    g.add_edge(4, 6)  # Making the graph fully connected
    pna_g = PNAGraph.from_raw(g)
    stranded_nodes_fully_connected = get_stranded_nodes(pna_g)
    assert len(stranded_nodes_fully_connected) == 0


def test_denoise_one_core_layer(synthetic_denoise_pxl_dataset):
    """Test the denoise_one_core_layer function.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
    """
    components = synthetic_denoise_pxl_dataset.adata().obs.index

    any_removed = False
    for comp in components:
        comp_graph = PNAGraph.from_edgelist(
            synthetic_denoise_pxl_dataset.filter(components=[comp])
            .edgelist()
            .to_polars()
            .lazy()
        )
        nodes_to_be_removed = denoise_one_core_layer(comp_graph)
        if not nodes_to_be_removed:
            continue
        any_removed = True

        # The sampler is seeded with random.Random(0), so the result is stable.
        assert denoise_one_core_layer(comp_graph) == nodes_to_be_removed

        node_core_numbers = pd.Series(nx.core_number(comp_graph.raw))
        assert all(node_core_numbers[nodes_to_be_removed] == 1)

        with_stranded = nodes_to_be_removed + get_stranded_nodes(
            comp_graph, nodes_to_be_removed
        )
        denoised_graph = comp_graph.raw.copy()
        denoised_graph.remove_nodes_from(with_stranded)
        assert nx.is_connected(denoised_graph)

    # The synthetic fixture is tuned so at least one component is denoised.
    assert any_removed


@pytest.fixture(name="synthetic_denoise_pxl_dataset", scope="module")
def synthetic_denoise_pxl_dataset_fixture(synthetic_denoise_pxl_file):
    """The synthetic denoise pxl file (see conftest) loaded as a dataset.

    Args:
        synthetic_denoise_pxl_file: Path to the synthetic denoise pxl file.
    """
    return read(synthetic_denoise_pxl_file)


@pytest.fixture(name="synthetic_component_graph", scope="module")
def synthetic_component_graph_fixture(synthetic_denoise_pxl_dataset):
    """A single synthetic component graph for the direct denoise_* function calls.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
    """
    comp = synthetic_denoise_pxl_dataset.adata().obs.index[0]
    return PNAGraph.from_edgelist(
        synthetic_denoise_pxl_dataset.filter(components=[comp])
        .edgelist()
        .to_polars()
        .lazy()
    )


def test_denoise_one_core_analysis(synthetic_denoise_pxl_dataset, tmp_path):
    """Test graph denoising with one-core only.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
        tmp_path: Tmp path.
    """
    pxl_file_target = PixelDatasetSaver(pxl_dataset=synthetic_denoise_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    with mock.patch(
        "pixelator.pna.analysis.denoise.load_antibody_panel"
    ) as mock_load_panel:
        # This is a workaround to make sure that the correct panel is loaded
        # eventhough we no longer set a default panel file.
        def f(*args, **kwargs):
            return load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")

        mock_load_panel.side_effect = f

        manager = AnalysisManager(
            [DenoiseGraph(run_one_core=True, run_ace=False)], n_cores=1
        )
        denoised_dataset = manager.execute(
            synthetic_denoise_pxl_dataset, pxl_file_target
        )

    adata = denoised_dataset.adata()
    obs = adata.obs
    assert "tau" not in obs.columns
    assert "tau_type" not in obs.columns
    assert "tau_thresholds" not in adata.uns
    components = synthetic_denoise_pxl_dataset.adata().obs.index

    # denoising actually removed nodes, otherwise the per-component checks
    # below would pass for a no-op denoiser
    assert obs["number_of_nodes_removed_in_denoise"].sum() > 0

    # (4) only one-core ran: ACE/PLS marked nothing and the per-method counts
    # reconcile with the total removed per component
    assert (obs["denoised_nodes_marked_only_by_ace"] == 0).all()
    assert (obs["denoised_nodes_marked_only_by_pls"] == 0).all()
    assert (
        obs["denoised_nodes_marked_only_by_one_core"]
        + obs["denoised_nodes_marked_stranded"]
        == obs["number_of_nodes_removed_in_denoise"]
    ).all()

    # denoising does not drop or add whole components
    assert set(obs.index) == set(components)

    # the denoised edge list is a pure subset of the original (denoising only
    # removes edges, it never introduces new umis/edges)
    original_edges = synthetic_denoise_pxl_dataset.edgelist().to_polars()
    denoised_edges = denoised_dataset.edgelist().to_polars()
    new_edges = denoised_edges.join(original_edges, on=["umi1", "umi2"], how="anti")
    assert new_edges.height == 0

    for comp in components:
        graph = PNAGraph.from_edgelist(
            synthetic_denoise_pxl_dataset.filter(components=[comp])
            .edgelist()
            .to_polars()
            .lazy()
        )
        denoised_graph = PNAGraph.from_edgelist(
            denoised_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
        )
        node_core_numbers = pd.Series(nx.core_number(graph.raw))
        denoised_node_core_numbers = pd.Series(nx.core_number(denoised_graph.raw))

        # Check that all nodes with core number 1 in the denoised graph were
        # also core number 1 in the original graph
        assert set(
            denoised_node_core_numbers[denoised_node_core_numbers == 1].index
        ).issubset(set(node_core_numbers[node_core_numbers == 1].index))

        # Check that higher core nodes remain intact after denoising
        assert_series_equal(
            node_core_numbers[node_core_numbers > 1],
            denoised_node_core_numbers[denoised_node_core_numbers > 1],
            check_like=True,
        )

        removed_nodes = set(graph.raw.nodes()) - set(denoised_graph.raw.nodes())
        n_removed = int(obs.loc[comp, "number_of_nodes_removed_in_denoise"])
        n_stranded = int(obs.loc[comp, "denoised_nodes_marked_stranded"])

        # the reported removal count matches the actual node-count drop
        assert denoised_graph.vcount() == graph.vcount() - n_removed
        assert len(removed_nodes) == n_removed

        # one-core only removes core-1 nodes; any higher-core removal can only
        # come from the stranding cleanup that follows
        removed_higher_core = {
            node for node in removed_nodes if node_core_numbers[node] > 1
        }
        assert len(removed_higher_core) <= n_stranded

        # the stranding cleanup leaves every denoised component connected
        assert nx.is_connected(denoised_graph.raw)


def _run_one_core_denoise(dataset, target_path):
    """Run one-core-only graph denoising on ``dataset`` writing to ``target_path``."""
    target = PixelDatasetSaver(pxl_dataset=dataset).save(
        "PNA055_Sample07_S7", target_path
    )
    with mock.patch(
        "pixelator.pna.analysis.denoise.load_antibody_panel"
    ) as mock_load_panel:
        mock_load_panel.side_effect = lambda *args, **kwargs: load_antibody_panel(
            pna_config, "proxiome-v1-immuno-155-v1.0"
        )
        manager = AnalysisManager(
            [DenoiseGraph(run_one_core=True, run_ace=False)], n_cores=1
        )
        return manager.execute(dataset, target)


def test_denoise_one_core_analysis_deterministic(
    synthetic_denoise_pxl_dataset, tmp_path
):
    """One-core denoising is reproducible: two runs remove the identical nodes.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
        tmp_path: Tmp path.
    """
    first = _run_one_core_denoise(
        synthetic_denoise_pxl_dataset, Path(tmp_path) / "first.pxl"
    )
    second = _run_one_core_denoise(
        synthetic_denoise_pxl_dataset, Path(tmp_path) / "second.pxl"
    )

    # identical per-component removal counts
    assert_series_equal(
        first.adata().obs["number_of_nodes_removed_in_denoise"],
        second.adata().obs["number_of_nodes_removed_in_denoise"],
    )

    # identical surviving nodes
    def surviving_umis(dataset):
        edgelist = dataset.edgelist().to_polars()
        return set(edgelist["umi1"]) | set(edgelist["umi2"])

    assert surviving_umis(first) == surviving_umis(second)


def test_denoise_one_core_skips_disqualified_component(tmp_path):
    """A component that is almost entirely one-core is left untouched.

    Components whose one-core layer exceeds ``one_core_ratio_threshold`` (90%) are
    disqualified from one-core denoising, so no nodes should be removed.

    Args:
        tmp_path: Tmp path.
    """
    panel = load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")
    markers = panel.markers

    # A star graph: one centre (umi1) joined to many leaves (umi2). Every node
    # has core number 1, so the component is disqualified from one-core denoising.
    n_leaves = 40
    leaves = list(range(2, 2 * n_leaves + 1, 2))
    edgelist = pl.DataFrame(
        {
            "umi1": [1] * n_leaves,
            "umi2": leaves,
            "marker_1": [markers[0]] * n_leaves,
            "marker_2": [markers[i % len(markers)] for i in range(n_leaves)],
            "component": ["star"] * n_leaves,
            "read_count": [1] * n_leaves,
        }
    )
    path = write_pna_pxl(
        edgelist,
        panel,
        Path(tmp_path) / "disqualified.pxl",
        sample_name="PNA055_Sample07_S7",
    )
    dataset = read(path)

    # sanity check: the component really is almost entirely one-core
    graph = PNAGraph.from_edgelist(dataset.edgelist().to_polars().lazy())
    core_numbers = pd.Series(nx.core_number(graph.raw))
    assert (core_numbers <= 1).mean() >= 0.9

    denoised_dataset = _run_one_core_denoise(dataset, Path(tmp_path) / "out.pxl")

    obs = denoised_dataset.adata().obs
    assert int(obs.loc["star", "number_of_nodes_removed_in_denoise"]) == 0

    # the component is unchanged: same nodes and same number of edges
    denoised_edges = denoised_dataset.edgelist().to_polars()
    original_edges = dataset.edgelist().to_polars()
    assert denoised_edges.height == original_edges.height
    assert (set(denoised_edges["umi1"]) | set(denoised_edges["umi2"])) == (
        set(original_edges["umi1"]) | set(original_edges["umi2"])
    )


def test_denoise_pls_reference_component_runs_and_cleans_coreness(
    synthetic_component_graph,
):
    """denoise_pls should return removable nodes and clean temporary coreness attrs.

    Args:
        synthetic_component_graph: A single synthetic component graph.
    """
    original_nodes = set(synthetic_component_graph.raw.nodes())

    removed = denoise_pls(synthetic_component_graph)

    assert removed != [None]
    assert len(removed) > 0
    assert set(removed).issubset(original_nodes)
    # Temporary "coreness" should always be cleaned up.
    assert all(
        "coreness" not in data
        for _, data in synthetic_component_graph.raw.nodes(data=True)
    )
    # The kept nodes are the largest connected set of the passing nodes.
    kept = original_nodes - set(removed)
    assert nx.is_connected(synthetic_component_graph.raw.subgraph(kept))


def test_denoise_pls_returns_empty_with_impossible_correlation_threshold(
    synthetic_component_graph,
):
    """No components can pass when min correlation is set above 1.0.

    Args:
        synthetic_component_graph: A single synthetic component graph.
    """
    removed = denoise_pls(synthetic_component_graph, min_pls_coreness_correlation=1.01)

    assert removed == []
    assert all(
        "coreness" not in data
        for _, data in synthetic_component_graph.raw.nodes(data=True)
    )


def test_denoise_pls_residualized_path_runs(synthetic_component_graph):
    """Residualized PLS denoising path should execute and return node ids.

    Args:
        synthetic_component_graph: A single synthetic component graph.
    """
    original_nodes = set(synthetic_component_graph.raw.nodes())

    removed = denoise_pls(synthetic_component_graph, residualize=True)

    assert removed != [None]
    assert len(removed) > 0
    assert set(removed).issubset(original_nodes)
    assert all(
        "coreness" not in data
        for _, data in synthetic_component_graph.raw.nodes(data=True)
    )

    # The non-residualized path is a distinct code path; both yield valid subsets.
    removed_plain = denoise_pls(synthetic_component_graph, residualize=False)
    assert set(removed_plain).issubset(original_nodes)


def test_denoise_ace_reference_component(synthetic_component_graph):
    """ACE removal list is exactly the peripheral ("low") partition.

    Args:
        synthetic_component_graph: A single synthetic component graph.
    """
    removed = denoise_ace(synthetic_component_graph)
    assert removed != [None]
    assert len(removed) > 0

    partitions = nx.get_node_attributes(synthetic_component_graph.raw, "partition")
    low_ids = {n for n, p in partitions.items() if p == "low"}
    high_ids = {n for n, p in partitions.items() if p == "high"}
    assert set(removed) == low_ids
    # ACE actually split the graph into two non-empty partitions.
    assert low_ids and high_ids

    # The "low" partition is genuinely peripheral: its nodes sit in lower cores.
    core_numbers = pd.Series(nx.core_number(synthetic_component_graph.raw))
    assert core_numbers[list(low_ids)].mean() < core_numbers[list(high_ids)].mean()


def test_denoise_ace_analysis(synthetic_denoise_pxl_dataset, tmp_path):
    """ACE-only graph denoising records ACE removal counts.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
        tmp_path: Tmp path.
    """
    components = synthetic_denoise_pxl_dataset.adata().obs.index

    pxl_file_target = PixelDatasetSaver(pxl_dataset=synthetic_denoise_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )

    with mock.patch(
        "pixelator.pna.analysis.denoise.load_antibody_panel"
    ) as mock_load_panel:
        mock_load_panel.side_effect = lambda *args, **kwargs: load_antibody_panel(
            pna_config, "proxiome-v1-immuno-155-v1.0"
        )
        manager = AnalysisManager(
            [DenoiseGraph(run_one_core=False, run_ace=True)], n_cores=1
        )
        denoised_dataset = manager.execute(
            synthetic_denoise_pxl_dataset, pxl_file_target
        )

    obs = denoised_dataset.adata().obs

    # ACE-only: every removed node is marked only by ACE, and some were removed.
    assert (
        obs["denoised_nodes_marked_only_by_ace"]
        == obs["number_of_nodes_removed_in_denoise"]
    ).all()
    assert obs["number_of_nodes_removed_in_denoise"].sum() > 0

    # ACE-only denoising with an LCC seed should not produce stranded nodes.
    assert obs["denoised_nodes_marked_stranded"].sum() == 0

    # Components are preserved and node counts drop exactly by what was removed.
    assert set(obs.index) == set(components)
    for comp in components:
        orig_graph = PNAGraph.from_edgelist(
            synthetic_denoise_pxl_dataset.filter(components=[comp])
            .edgelist()
            .to_polars()
            .lazy()
        )
        denoised_graph = PNAGraph.from_edgelist(
            denoised_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
        )
        removed = int(obs.loc[comp, "number_of_nodes_removed_in_denoise"])
        assert denoised_graph.vcount() == orig_graph.vcount() - removed


def test_denoise_ace_pls_one_core(synthetic_denoise_pxl_dataset, tmp_path):
    """ACE, PLS, and One Core graph denoising records removal counts.

    Args:
        synthetic_denoise_pxl_dataset: Small synthetic denoise pxl dataset.
        tmp_path: Tmp path.
    """
    components = synthetic_denoise_pxl_dataset.adata().obs.index

    pxl_file_target = PixelDatasetSaver(pxl_dataset=synthetic_denoise_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )

    with mock.patch(
        "pixelator.pna.analysis.denoise.load_antibody_panel"
    ) as mock_load_panel:
        mock_load_panel.side_effect = lambda *args, **kwargs: load_antibody_panel(
            pna_config, "proxiome-v1-immuno-155-v1.0"
        )
        manager = AnalysisManager(
            [DenoiseGraph(run_one_core=True, run_ace=True, run_pls=True)], n_cores=1
        )
        denoised_dataset = manager.execute(
            synthetic_denoise_pxl_dataset, pxl_file_target
        )

    obs = denoised_dataset.adata().obs

    summary_cols = [
        "denoised_nodes_marked_only_by_ace",
        "denoised_nodes_marked_only_by_pls",
        "denoised_nodes_marked_only_by_one_core",
        "denoised_nodes_marked_stranded",
        "denoised_nodes_marked_ace_and_pls",
        "denoised_nodes_marked_ace_and_one_core",
        "denoised_nodes_marked_pls_and_one_core",
        "denoised_nodes_marked_ace_pls_and_one_core",
    ]

    assert all(col in obs.columns for col in summary_cols)

    # The pre-denoise isotype fraction is recorded for every component.
    assert "pre_denoise_isotype_fraction" in obs.columns
    assert obs["pre_denoise_isotype_fraction"].notna().all()

    # The per-method marking columns reconcile with the total removed.
    pd.testing.assert_frame_equal(
        obs.loc[:, summary_cols]
        .sum(axis=1)
        .to_frame("number_of_nodes_removed_in_denoise"),
        obs.loc[:, ["number_of_nodes_removed_in_denoise"]],
    )

    # Removal happened and all three methods contributed across the dataset.
    assert obs["number_of_nodes_removed_in_denoise"].sum() > 0
    assert obs["denoised_nodes_marked_only_by_ace"].sum() > 0
    assert obs["denoised_nodes_marked_only_by_pls"].sum() > 0
    assert obs["denoised_nodes_marked_only_by_one_core"].sum() > 0

    # Components are preserved.
    assert set(obs.index) == set(components)
