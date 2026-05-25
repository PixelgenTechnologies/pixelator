"""Tests for component denoising functions.

Copyright © 2025 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path
from unittest import mock

import networkx as nx
import pandas as pd
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
from pixelator.pna.pixeldataset import PixelDatasetSaver

OVER_EXPRESSED_MARKERS = """name,count,component
CD156c,17,1a3afa30f0a90a83
CD18,29,1a3afa30f0a90a83
CD2,24,1a3afa30f0a90a83
CD24,8,1a3afa30f0a90a83
CD371,8,1a3afa30f0a90a83
CD43,63,1a3afa30f0a90a83
CD44,110,1a3afa30f0a90a83
HLA-ABC,102,1a3afa30f0a90a83
HLA-DR-DP-DQ,29,1a3afa30f0a90a83
KLRG1,9,1a3afa30f0a90a83
B2M,825,2084eed04807d6c5
CD162,23,2084eed04807d6c5
CD20,8,2084eed04807d6c5
CD226,9,2084eed04807d6c5
CD27,17,2084eed04807d6c5
CD44,60,2084eed04807d6c5
CD47,27,2084eed04807d6c5
CD58,12,2084eed04807d6c5
CD80,15,2084eed04807d6c5
HLA-ABC,299,2084eed04807d6c5
TIGIT,6,2084eed04807d6c5
CD18,45,2f8e1e63d7f5ee08
CD2,47,2f8e1e63d7f5ee08
CD33,11,2f8e1e63d7f5ee08
CD3e,137,2f8e1e63d7f5ee08
CD41,14,2f8e1e63d7f5ee08
CD43,86,2f8e1e63d7f5ee08
CD44,186,2f8e1e63d7f5ee08
CD45,357,2f8e1e63d7f5ee08
CD49e,15,2f8e1e63d7f5ee08
CD84,12,2f8e1e63d7f5ee08
CD89,12,2f8e1e63d7f5ee08
HLA-ABC,89,2f8e1e63d7f5ee08
IgD,14,2f8e1e63d7f5ee08
IgM,15,2f8e1e63d7f5ee08
B2M,618,31180b6c8ba952c2
CD103,9,31180b6c8ba952c2
CD141,8,31180b6c8ba952c2
CD154,17,31180b6c8ba952c2
CD158,12,31180b6c8ba952c2
CD158b,11,31180b6c8ba952c2
CD159a,11,31180b6c8ba952c2
CD16,18,31180b6c8ba952c2
CD161,23,31180b6c8ba952c2
CD169,14,31180b6c8ba952c2
CD1a,11,31180b6c8ba952c2
CD206,14,31180b6c8ba952c2
CD209,8,31180b6c8ba952c2
CD21,18,31180b6c8ba952c2
CD26,8,31180b6c8ba952c2
CD278,9,31180b6c8ba952c2
CD45,59,31180b6c8ba952c2
CD52,45,31180b6c8ba952c2
CD54,11,31180b6c8ba952c2
CD64,8,31180b6c8ba952c2
CD73,15,31180b6c8ba952c2
CD80,17,31180b6c8ba952c2
CD90,20,31180b6c8ba952c2
CD93,8,31180b6c8ba952c2
CD95,18,31180b6c8ba952c2
HLA-ABC,621,31180b6c8ba952c2
IgE,11,31180b6c8ba952c2
TCRVB5,9,31180b6c8ba952c2
mIgG2a,14,31180b6c8ba952c2
mIgG2b,12,31180b6c8ba952c2
CD11b,11,57129a8b0fff38c6
CD156c,17,57129a8b0fff38c6
CD158,11,57129a8b0fff38c6
CD369,11,57129a8b0fff38c6
CD40,14,57129a8b0fff38c6
CD43,81,57129a8b0fff38c6
CD44,230,57129a8b0fff38c6
CD45,225,57129a8b0fff38c6
CD45RB,107,57129a8b0fff38c6
CD54,9,57129a8b0fff38c6
CD57,14,57129a8b0fff38c6
CD58,15,57129a8b0fff38c6
CD6,26,57129a8b0fff38c6
CD79a,8,57129a8b0fff38c6
CD81,20,57129a8b0fff38c6
CD82,35,57129a8b0fff38c6
GPR56,9,57129a8b0fff38c6
IgD,12,57129a8b0fff38c6
IgE,8,57129a8b0fff38c6
B2M,357,6796966160a5b359
CD117,11,6796966160a5b359
CD13,15,6796966160a5b359
CD14,14,6796966160a5b359
CD154,15,6796966160a5b359
CD192,9,6796966160a5b359
CD26,11,6796966160a5b359
CD33,9,6796966160a5b359
CD357,8,6796966160a5b359
CD45,110,6796966160a5b359
CD45RA,57,6796966160a5b359
CD48,33,6796966160a5b359
CD52,51,6796966160a5b359
CD53,20,6796966160a5b359
CD54,8,6796966160a5b359
CD62P,5,6796966160a5b359
CD82,47,6796966160a5b359
CD90,17,6796966160a5b359
HLA-ABC,279,6796966160a5b359
CD102,27,6e420e5c92a14c35
CD103,11,6e420e5c92a14c35
CD11b,8,6e420e5c92a14c35
CD127,20,6e420e5c92a14c35
CD156c,18,6e420e5c92a14c35
CD159a,11,6e420e5c92a14c35
CD159c,9,6e420e5c92a14c35
CD16,20,6e420e5c92a14c35
CD161,15,6e420e5c92a14c35
CD226,8,6e420e5c92a14c35
CD229,15,6e420e5c92a14c35
CD26,11,6e420e5c92a14c35
CD302,17,6e420e5c92a14c35
CD38,20,6e420e5c92a14c35
CD45,108,6e420e5c92a14c35
CD45RB,51,6e420e5c92a14c35
CD56,9,6e420e5c92a14c35
CD59,42,6e420e5c92a14c35
CD7,12,6e420e5c92a14c35
CD71,8,6e420e5c92a14c35
CD82,65,6e420e5c92a14c35
CD85j,9,6e420e5c92a14c35
CD89,15,6e420e5c92a14c35
CD90,14,6e420e5c92a14c35
CD94,15,6e420e5c92a14c35
GPR56,6,6e420e5c92a14c35
HLA-ABC,173,6e420e5c92a14c35
KLRG1,12,6e420e5c92a14c35
TCRab,20,6e420e5c92a14c35
VISTA,24,6e420e5c92a14c35
B2M,263,70f73b946d989e86
CD11a,21,70f73b946d989e86
CD158b,6,70f73b946d989e86
CD163,6,70f73b946d989e86
CD302,9,70f73b946d989e86
CD369,6,70f73b946d989e86
CD45,198,70f73b946d989e86
CD48,26,70f73b946d989e86
CD49D,14,70f73b946d989e86
CD49e,9,70f73b946d989e86
CD56,6,70f73b946d989e86
CD59,27,70f73b946d989e86
GPR56,6,70f73b946d989e86
HLA-ABC,344,70f73b946d989e86
CD103,8,85078d2392036eb1
CD13,21,85078d2392036eb1
CD150,12,85078d2392036eb1
CD156c,18,85078d2392036eb1
CD16,12,85078d2392036eb1
CD200,18,85078d2392036eb1
CD277,14,85078d2392036eb1
CD305,11,85078d2392036eb1
CD328,14,85078d2392036eb1
CD38,12,85078d2392036eb1
CD47,23,85078d2392036eb1
CD49D,12,85078d2392036eb1
CD58,17,85078d2392036eb1
CD59,45,85078d2392036eb1
CD7,9,85078d2392036eb1
CD79a,11,85078d2392036eb1
CD80,14,85078d2392036eb1
CD84,12,85078d2392036eb1
CD89,12,85078d2392036eb1
CD90,18,85078d2392036eb1
CD93,6,85078d2392036eb1
HLA-ABC,122,85078d2392036eb1
TCRVd2,6,85078d2392036eb1
CD14,14,929e5c4405c4033e
CD169,15,929e5c4405c4033e
CD19,20,929e5c4405c4033e
CD319,15,929e5c4405c4033e
CD37,20,929e5c4405c4033e
CD43,180,929e5c4405c4033e
CD44,191,929e5c4405c4033e
CD45,254,929e5c4405c4033e
CD55,29,929e5c4405c4033e
CD69,17,929e5c4405c4033e
CD80,20,929e5c4405c4033e
CD95,18,929e5c4405c4033e
HLA-ABC,173,929e5c4405c4033e
KLRG1,11,929e5c4405c4033e
TCRVd2,9,929e5c4405c4033e
VISTA,17,929e5c4405c4033e
mIgG2a,14,929e5c4405c4033e
B2M,404,9728fca62445e41b
CD158,9,9728fca62445e41b
CD159a,9,9728fca62445e41b
CD192,11,9728fca62445e41b
CD44,167,9728fca62445e41b
CD45,143,9728fca62445e41b
CD49e,14,9728fca62445e41b
CD52,54,9728fca62445e41b
CD72,9,9728fca62445e41b
CD79a,6,9728fca62445e41b
HLA-ABC,353,9728fca62445e41b
IgE,9,9728fca62445e41b
mIgG2a,12,9728fca62445e41b
B2M,441,a824ef9068d65a0a
CD117,14,a824ef9068d65a0a
CD152,8,a824ef9068d65a0a
CD193,11,a824ef9068d65a0a
CD26,12,a824ef9068d65a0a
CD28,11,a824ef9068d65a0a
CD305,17,a824ef9068d65a0a
CD357,12,a824ef9068d65a0a
CD41,20,a824ef9068d65a0a
CD45,122,a824ef9068d65a0a
CD48,44,a824ef9068d65a0a
CD55,21,a824ef9068d65a0a
CD57,11,a824ef9068d65a0a
CD7,12,a824ef9068d65a0a
CD93,12,a824ef9068d65a0a
CD94,11,a824ef9068d65a0a
HLA-ABC,495,a824ef9068d65a0a
KLRG1,14,a824ef9068d65a0a
TCRVB5,11,a824ef9068d65a0a
CD11a,18,c00d7cb4851e7bfd
CD200,11,c00d7cb4851e7bfd
CD305,12,c00d7cb4851e7bfd
CD4,6,c00d7cb4851e7bfd
CD43,80,c00d7cb4851e7bfd
CD44,81,c00d7cb4851e7bfd
CD45,255,c00d7cb4851e7bfd
CD64,6,c00d7cb4851e7bfd
CD81,24,c00d7cb4851e7bfd
CD94,12,c00d7cb4851e7bfd
HLA-ABC,126,c00d7cb4851e7bfd
TCRva7.2,6,c00d7cb4851e7bfd
B2M,227,c4c3ef9497b3746d
CD14,3,c4c3ef9497b3746d
CD158a,3,c4c3ef9497b3746d
CD1b,3,c4c3ef9497b3746d
CD20,3,c4c3ef9497b3746d
CD209,3,c4c3ef9497b3746d
CD37,6,c4c3ef9497b3746d
CD38,6,c4c3ef9497b3746d
CD45RO,3,c4c3ef9497b3746d
CD49e,6,c4c3ef9497b3746d
CD57,3,c4c3ef9497b3746d
CD79a,3,c4c3ef9497b3746d
CD89,5,c4c3ef9497b3746d
HLA-ABC,209,c4c3ef9497b3746d
HLA-DQ,3,c4c3ef9497b3746d
mIgG2a,5,c4c3ef9497b3746d
CD192,15,c771a99bb8f21eb3
CD2,36,c771a99bb8f21eb3
CD21,14,c771a99bb8f21eb3
CD319,18,c771a99bb8f21eb3
CD37,17,c771a99bb8f21eb3
CD38,12,c771a99bb8f21eb3
CD44,140,c771a99bb8f21eb3
CD45,158,c771a99bb8f21eb3
CD47,36,c771a99bb8f21eb3
CD59,36,c771a99bb8f21eb3
CD95,15,c771a99bb8f21eb3
HLA-ABC,98,c771a99bb8f21eb3
TCRVg9,8,c771a99bb8f21eb3
CD117,12,cd321b7b45c40ac9
CD14,12,cd321b7b45c40ac9
CD159a,8,cd321b7b45c40ac9
CD180,14,cd321b7b45c40ac9
CD2,86,cd321b7b45c40ac9
CD27,26,cd321b7b45c40ac9
CD302,15,cd321b7b45c40ac9
CD319,18,cd321b7b45c40ac9
CD33,12,cd321b7b45c40ac9
CD366,12,cd321b7b45c40ac9
CD40,12,cd321b7b45c40ac9
CD44,98,cd321b7b45c40ac9
CD45,216,cd321b7b45c40ac9
CD49D,15,cd321b7b45c40ac9
CD53,17,cd321b7b45c40ac9
CD54,9,cd321b7b45c40ac9
CD8,21,cd321b7b45c40ac9
CD89,14,cd321b7b45c40ac9
CD90,14,cd321b7b45c40ac9
TCRgd,9,cd321b7b45c40ac9
TIGIT,9,cd321b7b45c40ac9
B2M,114,ce4ff26a68bae7c3
CD103,11,ce4ff26a68bae7c3
CD162,21,ce4ff26a68bae7c3
CD268,9,ce4ff26a68bae7c3
CD337,6,ce4ff26a68bae7c3
CD3e,110,ce4ff26a68bae7c3
CD41,12,ce4ff26a68bae7c3
CD45,218,ce4ff26a68bae7c3
CD69,9,ce4ff26a68bae7c3
HLA-ABC,210,ce4ff26a68bae7c3
TCRVB5,8,ce4ff26a68bae7c3
TCRgd,8,ce4ff26a68bae7c3
VISTA,14,ce4ff26a68bae7c3
mIgG2b,11,ce4ff26a68bae7c3
CD117,9,dd219965cf12498f
CD14,14,dd219965cf12498f
CD156c,23,dd219965cf12498f
CD161,17,dd219965cf12498f
CD162,24,dd219965cf12498f
CD199,18,dd219965cf12498f
CD206,8,dd219965cf12498f
CD244,6,dd219965cf12498f
CD25,9,dd219965cf12498f
CD274,12,dd219965cf12498f
CD31,15,dd219965cf12498f
CD328,11,dd219965cf12498f
CD44,126,dd219965cf12498f
CD45,242,dd219965cf12498f
CD53,26,dd219965cf12498f
CD79a,8,dd219965cf12498f
CD94,6,dd219965cf12498f
HLA-ABC,239,dd219965cf12498f
CD11b,9,ed9c2a47db840b3d
CD13,17,ed9c2a47db840b3d
CD191,11,ed9c2a47db840b3d
CD199,20,ed9c2a47db840b3d
CD32,12,ed9c2a47db840b3d
CD328,12,ed9c2a47db840b3d
CD4,8,ed9c2a47db840b3d
CD43,90,ed9c2a47db840b3d
CD45,129,ed9c2a47db840b3d
CD49e,18,ed9c2a47db840b3d
CD73,12,ed9c2a47db840b3d
HLA-ABC,246,ed9c2a47db840b3d
CD156c,15,fe6336721ae34f6c
CD1c,8,fe6336721ae34f6c
CD273,9,fe6336721ae34f6c
CD366,9,fe6336721ae34f6c
CD3e,135,fe6336721ae34f6c
CD40,9,fe6336721ae34f6c
CD43,158,fe6336721ae34f6c
CD44,236,fe6336721ae34f6c
CD45,188,fe6336721ae34f6c
CD80,14,fe6336721ae34f6c
CD81,18,fe6336721ae34f6c
CD90,14,fe6336721ae34f6c
CD94,9,fe6336721ae34f6c
HLA-ABC,147,fe6336721ae34f6c
TCRab,17,fe6336721ae34f6c
"""


@pytest.mark.slow
def test_get_overexpressed_markers_in_one_core(denoise_pxl_dataset):
    """Test the get_overexpressed_markers_in_one_core function.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    components = denoise_pxl_dataset.adata().obs.index

    def over_expressed_markers_per_component():
        for comp in components:
            comp_graph = PNAGraph.from_edgelist(
                denoise_pxl_dataset.filter(components=[comp])
                .edgelist()
                .to_polars()
                .lazy()
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
def test_denoise_one_core_layer(denoise_pxl_dataset):
    """Test the denoise_one_core_layer function.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    components = denoise_pxl_dataset.adata().obs.index

    for comp in components:
        comp_graph = PNAGraph.from_edgelist(
            denoise_pxl_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
        )
        nodes_to_be_removed = denoise_one_core_layer(comp_graph)
        if not nodes_to_be_removed:
            continue
        node_core_numbers = pd.Series(nx.core_number(comp_graph.raw))
        assert all(node_core_numbers[nodes_to_be_removed] == 1)

        with_stranded = nodes_to_be_removed + get_stranded_nodes(
            comp_graph, nodes_to_be_removed
        )
        denoised_graph = comp_graph.raw.copy()
        denoised_graph.remove_nodes_from(with_stranded)
        assert nx.is_connected(denoised_graph)


@pytest.mark.slow
def test_denoise_one_core_analysis(denoise_pxl_dataset, tmp_path):
    """Test graph denoising with one-core only.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.
        tmp_path: Tmp path.

    """
    pxl_file_target = PixelDatasetSaver(pxl_dataset=denoise_pxl_dataset).save(
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

        manager = AnalysisManager([DenoiseGraph(run_one_core=True, run_ace=False)])
        denoised_dataset = manager.execute(denoise_pxl_dataset, pxl_file_target)

    assert "tau_type" in denoised_dataset.adata().obs.columns
    components = denoise_pxl_dataset.adata().obs.index
    for comp in components:
        graph = PNAGraph.from_edgelist(
            denoise_pxl_dataset.filter(components=[comp]).edgelist().to_polars().lazy()
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


REFERENCE_ACE_COMPONENT = "57129a8b0fff38c6"
REFERENCE_ACE_LOW_NODE_COUNT = 5436


def test_denoise_pls_reference_component_runs_and_cleans_coreness(denoise_pxl_dataset):
    """denoise_pls should return removable nodes and clean temporary coreness attrs.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    comp_graph = PNAGraph.from_edgelist(
        denoise_pxl_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )
    original_nodes = set(comp_graph.raw.nodes())

    removed = denoise_pls(comp_graph)

    assert removed != [None]
    assert set(removed).issubset(original_nodes)
    # Temporary "coreness" should always be cleaned up.
    assert all("coreness" not in data for _, data in comp_graph.raw.nodes(data=True))


def test_denoise_pls_returns_empty_with_impossible_correlation_threshold(
    denoise_pxl_dataset,
):
    """No components can pass when min correlation is set above 1.0.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    comp_graph = PNAGraph.from_edgelist(
        denoise_pxl_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )

    removed = denoise_pls(comp_graph, min_pls_coreness_correlation=1.01)

    assert removed == []
    assert all("coreness" not in data for _, data in comp_graph.raw.nodes(data=True))


def test_denoise_pls_residualized_path_runs(denoise_pxl_dataset):
    """Residualized PLS denoising path should execute and return node ids.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    comp_graph = PNAGraph.from_edgelist(
        denoise_pxl_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )
    original_nodes = set(comp_graph.raw.nodes())

    removed = denoise_pls(comp_graph, residualize=True)

    assert removed != [None]
    assert set(removed).issubset(original_nodes)
    assert all("coreness" not in data for _, data in comp_graph.raw.nodes(data=True))


@pytest.mark.slow
def test_denoise_ace_reference_component(denoise_pxl_dataset):
    """ACE layer removal list matches peripheral partition on reference component.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.

    """
    comp_graph = PNAGraph.from_edgelist(
        denoise_pxl_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )
    removed = denoise_ace(comp_graph)
    assert removed != [None]
    assert len(removed) == REFERENCE_ACE_LOW_NODE_COUNT
    partitions = nx.get_node_attributes(comp_graph.raw, "partition")
    low_ids = {n for n, p in partitions.items() if p == "low"}
    assert set(removed) == low_ids


@pytest.mark.slow
def test_denoise_ace_analysis(denoise_pxl_dataset, tmp_path):
    """ACE-only graph denoising records ACE removal counts.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.
        tmp_path: Tmp path.

    """
    pxl_file_target = PixelDatasetSaver(pxl_dataset=denoise_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )

    manager = AnalysisManager(
        [DenoiseGraph(run_one_core=False, run_ace=True)], n_cores=4
    )
    denoised_dataset = manager.execute(denoise_pxl_dataset, pxl_file_target)

    obs = denoised_dataset.adata().obs

    assert (
        obs["denoised_nodes_marked_only_by_ace"]
        == obs["number_of_nodes_removed_in_denoise"]
    ).all()
    assert int(
        obs.loc[REFERENCE_ACE_COMPONENT, "denoised_nodes_marked_only_by_ace"]
    ) == (REFERENCE_ACE_LOW_NODE_COUNT)

    assert (
        obs.loc[:, "denoised_nodes_marked_stranded"].sum() == 0
    )  # ACE-only denoising with LCC seed should not produce stranded nodes

    orig_graph = PNAGraph.from_edgelist(
        denoise_pxl_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )
    denoised_graph = PNAGraph.from_edgelist(
        denoised_dataset.filter(components=[REFERENCE_ACE_COMPONENT])
        .edgelist()
        .to_polars()
        .lazy()
    )
    assert denoised_graph.vcount() == orig_graph.vcount() - REFERENCE_ACE_LOW_NODE_COUNT


@pytest.mark.slow
def test_denoise_ace_pls_one_core(denoise_pxl_dataset, tmp_path):
    """ACE, PLS, and One Core graph denoising records removal counts.

    Args:
        denoise_pxl_dataset: Denoise pxl dataset.
        tmp_path: Tmp path.

    """

    pxl_file_target = PixelDatasetSaver(pxl_dataset=denoise_pxl_dataset).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )

    manager = AnalysisManager(
        [DenoiseGraph(run_one_core=True, run_ace=True, run_pls=True)], n_cores=4
    )
    denoised_dataset = manager.execute(denoise_pxl_dataset, pxl_file_target)

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
    pd.testing.assert_frame_equal(
        obs.loc[:, summary_cols]
        .sum(axis=1)
        .to_frame("number_of_nodes_removed_in_denoise"),
        obs.loc[:, ["number_of_nodes_removed_in_denoise"]],
    )
