"""Compare the spectral layout against the default WPMDS layout.

Copyright © 2026 Pixelgen Technologies AB.

We compare pairwise Euclidean distances between the same pairs of nodes,
which is invariant to rotation and scaling.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset

N_PAIRS = 10_000
SEED = 42


def _sample_node_pairs(n_nodes: int, n_pairs: int, rng: np.random.Generator):
    """Sample distinct-node index pairs, resampling only the rare self-pairs."""
    i = rng.integers(0, n_nodes, size=n_pairs)
    j = rng.integers(0, n_nodes, size=n_pairs)
    self_pairs = i == j
    while self_pairs.any():
        j[self_pairs] = rng.integers(0, n_nodes, size=self_pairs.sum())
        self_pairs = i == j
    return i, j


@pytest.mark.slow
def test_spectral_layout_correlates_with_wpmds_on_real_component(
    pna_pxl_dataset: PNAPixelDataset,
):
    """Pairwise distances from spectral_3d should rank-correlate with wpmds_3d."""

    obs = pna_pxl_dataset.adata().obs
    component_id = obs["n_edges"].idxmax()

    edgelist = (
        pna_pxl_dataset.filter(components=[component_id]).edgelist().to_record_batches()
    )
    graph = PNAGraph.from_record_batches(edgelist)

    wpmds = graph.layout_coordinates(
        layout_algorithm="wpmds_3d", get_node_marker_matrix=False, random_seed=SEED
    ).set_index("index")
    spectral = graph.layout_coordinates(
        layout_algorithm="spectral_3d", get_node_marker_matrix=False, random_seed=SEED
    ).set_index("index")

    assert set(wpmds.index) == set(spectral.index)
    nodes = wpmds.index
    coords_wpmds = wpmds.loc[nodes, ["x", "y", "z"]].to_numpy()
    coords_spectral = spectral.loc[nodes, ["x", "y", "z"]].to_numpy()

    rng = np.random.default_rng(SEED)
    i, j = _sample_node_pairs(len(nodes), N_PAIRS, rng)

    dist_wpmds = np.linalg.norm(coords_wpmds[i] - coords_wpmds[j], axis=1)
    dist_spectral = np.linalg.norm(coords_spectral[i] - coords_spectral[j], axis=1)

    correlation = spearmanr(dist_wpmds, dist_spectral)

    assert correlation.statistic > 0.5
