"""Tests for component denoising functions.

Copyright © 2025 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from pixelator.pna.analysis.denoise import (
    DenoiseOneCore,
    denoise_one_core_layer,
    get_overexpressed_markers_in_one_core,
    get_stranded_nodes,
)
from pixelator.pna.analysis_engine import AnalysisManager
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PixelDatasetSaver

OVER_EXPRESSED_MARKERS = """name,count,component
CD48,11,0a45497c6bfbfb22
CD45,27,0a45497c6bfbfb22
CD44,50,0a45497c6bfbfb22
CD319,8,0a45497c6bfbfb22
CD6,8,0a45497c6bfbfb22
HLA-DR-DP-DQ,15,0a45497c6bfbfb22
CD45RA,14,0a45497c6bfbfb22
CD95,8,0a45497c6bfbfb22
CD66b,203,0a45497c6bfbfb22
CD3e,6,0a45497c6bfbfb22
CD8,6,0a45497c6bfbfb22
CD2,5,0a45497c6bfbfb22
CD302,11,0a45497c6bfbfb22
CD335,8,0a45497c6bfbfb22
CD274,5,0a45497c6bfbfb22
CD357,5,0a45497c6bfbfb22
CD44,104,2708240b908e2eba
CD56,8,2708240b908e2eba
TCRab,15,2708240b908e2eba
CD19,12,2708240b908e2eba
CD73,9,2708240b908e2eba
mIgG1,11,2708240b908e2eba
CD357,17,2708240b908e2eba
mIgG2a,14,2708240b908e2eba
CD319,30,2708240b908e2eba
CD3e,17,2708240b908e2eba
CD45RB,15,2708240b908e2eba
CD335,8,2708240b908e2eba
IgM,14,2708240b908e2eba
CD62P,11,2708240b908e2eba
CD150,14,2708240b908e2eba
CD72,9,2708240b908e2eba
CD154,11,2708240b908e2eba
CD27,9,2708240b908e2eba
CD95,17,2708240b908e2eba
CD49e,23,2708240b908e2eba
CD37,24,2708240b908e2eba
CD117,14,2708240b908e2eba
CD89,15,2708240b908e2eba
CD158a,11,2708240b908e2eba
CD279,6,2708240b908e2eba
NKp80,8,2708240b908e2eba
CD169,9,2708240b908e2eba
CD8,9,2708240b908e2eba
CD127,8,2708240b908e2eba
CD9,12,2708240b908e2eba
CD66b,8,2708240b908e2eba
IgE,8,2708240b908e2eba
CD273,39,2708240b908e2eba
CD226,12,2708240b908e2eba
CD41,12,2708240b908e2eba
VISTA,17,2708240b908e2eba
CD268,8,2708240b908e2eba
CD158b,11,2708240b908e2eba
CD57,6,2708240b908e2eba
TCRgd,15,2708240b908e2eba
CD192,17,2708240b908e2eba
CD137,6,2708240b908e2eba
CD21,20,2708240b908e2eba
CD159a,12,2708240b908e2eba
CD200,8,2708240b908e2eba
CD229,11,2708240b908e2eba
CD24,11,2708240b908e2eba
CD5,17,2708240b908e2eba
CD314,15,2708240b908e2eba
CD90,12,2708240b908e2eba
CD70,9,2708240b908e2eba
CD163,18,2708240b908e2eba
CD326,9,2708240b908e2eba
CD138,8,2708240b908e2eba
CD1b,6,2708240b908e2eba
CD28,9,2708240b908e2eba
CD6,12,2708240b908e2eba
TCRVB5,9,2708240b908e2eba
CD11b,8,2708240b908e2eba
CD269,8,2708240b908e2eba
CD206,9,2708240b908e2eba
CD103,15,2708240b908e2eba
CD22,14,2708240b908e2eba
CD69,15,2708240b908e2eba
CD1c,9,2708240b908e2eba
CD94,9,2708240b908e2eba
CD134,6,2708240b908e2eba
CD159c,9,2708240b908e2eba
CD209,6,2708240b908e2eba
VISTA,21,c3c393e9a17c1981
CD16,26,c3c393e9a17c1981
CD44,269,c3c393e9a17c1981
CD95,23,c3c393e9a17c1981
CD58,39,c3c393e9a17c1981
CD18,35,c3c393e9a17c1981
CD69,12,c3c393e9a17c1981
CD8,9,c3c393e9a17c1981
CD49e,18,c3c393e9a17c1981
CD9,15,c3c393e9a17c1981
CD7,35,c3c393e9a17c1981
IgM,18,c3c393e9a17c1981
CD85j,32,c3c393e9a17c1981
CD21,14,c3c393e9a17c1981
CD277,21,c3c393e9a17c1981
CD319,36,c3c393e9a17c1981
CD366,14,c3c393e9a17c1981
CD273,17,c3c393e9a17c1981
CD199,20,c3c393e9a17c1981
CD169,17,c3c393e9a17c1981
CD1a,12,c3c393e9a17c1981
CD357,35,c3c393e9a17c1981
CD13,18,c3c393e9a17c1981
NKp80,5,c3c393e9a17c1981
CD328,12,c3c393e9a17c1981
CD73,12,c3c393e9a17c1981
CD335,9,c3c393e9a17c1981
CD231,6,c3c393e9a17c1981
CD28,15,c3c393e9a17c1981
CD79a,6,c3c393e9a17c1981
CD44,297,d4074c845bb62800
CD2,114,d4074c845bb62800
CD45,95,d4074c845bb62800
CD36,17,d4074c845bb62800
CD328,14,d4074c845bb62800
CD154,6,d4074c845bb62800
CD55,21,d4074c845bb62800
CD20,9,d4074c845bb62800
CD274,9,d4074c845bb62800
CD16,17,d4074c845bb62800
mIgG1,6,d4074c845bb62800
mIgG2b,6,d4074c845bb62800
CD357,99,d4074c845bb62800
CD319,21,d4074c845bb62800
CD27,21,d4074c845bb62800
CD72,12,d4074c845bb62800
CD45RA,18,d4074c845bb62800
CD13,11,d4074c845bb62800
CD200,6,d4074c845bb62800
CD7,14,d4074c845bb62800
CD137,6,d4074c845bb62800
CD21,11,d4074c845bb62800
CD35,8,d4074c845bb62800
TCRva7.2,8,d4074c845bb62800
CD117,12,d4074c845bb62800
CD123,6,d4074c845bb62800
CX3CR1,5,d4074c845bb62800
CD369,6,d4074c845bb62800
CD89,11,d4074c845bb62800
CD314,6,d4074c845bb62800
CD62P,5,d4074c845bb62800
CD41,5,d4074c845bb62800
VISTA,8,d4074c845bb62800
CD57,11,d4074c845bb62800
CD158,6,d4074c845bb62800
CD66b,6,d4074c845bb62800
CD44,350,efe0ed189cb499fc
CD2,95,efe0ed189cb499fc
HLA-DR-DP-DQ,23,efe0ed189cb499fc
CD22,6,efe0ed189cb499fc
CD357,59,efe0ed189cb499fc
CD7,23,efe0ed189cb499fc
CD29,42,efe0ed189cb499fc
CD36,15,efe0ed189cb499fc
CD319,14,efe0ed189cb499fc
CD191,6,efe0ed189cb499fc
CD45RA,12,efe0ed189cb499fc
CD8,6,efe0ed189cb499fc
CD328,6,efe0ed189cb499fc
CD35,8,efe0ed189cb499fc
CD16,5,efe0ed189cb499fc
CX3CR1,5,efe0ed189cb499fc
"""


@pytest.mark.slow
def test_get_overexpressed_markers_in_one_core(pna_pxl_dataset):
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
        pd.read_csv(StringIO(OVER_EXPRESSED_MARKERS))
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


@pytest.mark.slow
def test_denoise_one_core_layer(pna_pxl_dataset):
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


@pytest.mark.slow
def test_denoise_one_core_analysis(pna_pxl_dataset, tmp_path):
    """Test the DenoiseOneCore analysis."""
    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    manager = AnalysisManager([DenoiseOneCore()])
    denoised_dataset = manager.execute(pna_pxl_dataset, pxl_file_target)
    assert "tau_type" in denoised_dataset.adata().obs.columns
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
