"""Smoke tests for the synthetic analysis pxl fixture.

The tests that use the fixture rely on the properties checked here.

Copyright © 2026 Pixelgen Technologies AB.
"""

import networkx as nx

N_CELLS = 4
N_MARKERS = 40
N_EDGES_PER_CELL = 3000


def test_synthetic_analysis_pxl_dataset_has_expected_structure(
    synthetic_analysis_pxl_dataset,
):
    """The synthetic dataset has the expected size, in cells, markers and edges.

    Args:
        synthetic_analysis_pxl_dataset: Synthetic analysis pxl dataset.
    """
    adata = synthetic_analysis_pxl_dataset.adata()
    edgelist = synthetic_analysis_pxl_dataset.edgelist().to_polars()

    assert adata.n_obs == N_CELLS
    assert adata.n_vars == N_MARKERS
    assert set(edgelist["component"].unique()) == set(adata.obs.index)
    assert edgelist.height == N_CELLS * N_EDGES_PER_CELL

    # Every cell has enough different markers for marker pair analyses.
    markers_per_cell = (adata.X > 0).sum(axis=1)
    assert markers_per_cell.min() > 20


def test_synthetic_analysis_pxl_dataset_cells_are_connected(
    synthetic_analysis_pxl_dataset,
):
    """Each cell is a single connected graph.

    Analyses such as the SVD noise estimate fail on graphs in several pieces.

    Args:
        synthetic_analysis_pxl_dataset: Synthetic analysis pxl dataset.
    """
    edgelist = synthetic_analysis_pxl_dataset.edgelist().to_polars()

    for component, cell_edges in edgelist.group_by("component"):
        graph = nx.Graph(zip(cell_edges["umi1"], cell_edges["umi2"]))
        assert nx.is_connected(graph), f"component {component} is not connected"
