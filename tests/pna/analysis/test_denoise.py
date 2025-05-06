"""Tests for the analysis engine.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pathlib import Path

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pixelator.pna.analysis.denoise import (
    DenoiseOneCore,
    denoise_one_core_layer,
    get_overexpressed_markers_in_one_core,
    get_stranded_nodes,
)
from pixelator.pna.analysis_engine import AnalysisManager
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PixelDatasetSaver


def test_get_overexpressed_markers_in_one_core(pna_pxl_dataset, snapshot):
    """Test the get_overexpressed_markers_in_one_core function."""
    components = pna_pxl_dataset.adata().obs.index

    def over_expressed_markers_per_component():
        for comp in components:
            comp_graph = PNAGraph.from_edgelist(
                pna_pxl_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
            )
            node_marker_counts = comp_graph.node_marker_counts
            node_core_numbers = pd.Series(nx.core_number(comp_graph.raw))
            over_expressed_markers = get_overexpressed_markers_in_one_core(
                node_marker_counts=node_marker_counts,
                node_core_numbers=node_core_numbers,
            )
            over_expressed_markers["component"] = comp
            yield over_expressed_markers

    over_expressed_markers = (
        pd.concat(over_expressed_markers_per_component())
        .sort_values(["component", "name"])
        .reset_index(drop=True)
    )
    reference = (
        pd.read_csv(snapshot.snapshot_dir / "over_expressed_markers.csv")
        .sort_values(["component", "name"])
        .reset_index(drop=True)
    )
    assert_frame_equal(over_expressed_markers, reference)


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


def test_denoise_one_core_layer(pna_pxl_dataset, snapshot):
    """Test the denoise_one_core_layer function."""
    components = pna_pxl_dataset.adata().obs.index

    for comp in components:
        comp_graph = PNAGraph.from_edgelist(
            pna_pxl_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
        )
        nodes_to_be_removed = denoise_one_core_layer(comp_graph)
        node_core_numbers = pd.Series(nx.core_number(comp_graph.raw))
        assert all(node_core_numbers[nodes_to_be_removed] == 1)

        denoised_graph = comp_graph.raw.copy()
        denoised_graph.remove_nodes_from(nodes_to_be_removed)
        assert nx.is_connected(denoised_graph)


def test_denoise_one_core_analysis(pna_pxl_dataset, snapshot, tmp_path):
    """Test the DenoiseOneCore analysis."""
    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    manager = AnalysisManager([DenoiseOneCore()])
    denoised_dataset = manager.execute(pna_pxl_dataset, pxl_file_target)

    components = pna_pxl_dataset.adata().obs.index
    for comp in components:
        graph = PNAGraph.from_edgelist(
            pna_pxl_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
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
