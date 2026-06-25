"""Shared fixtures for the PNA denoise tests.

Copyright © 2025 Pixelgen Technologies AB.
"""

import pytest

from pixelator.common.utils.test_data_generator import (
    generate_edgelist,
    write_pna_pxl,
)
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel


@pytest.fixture(name="synthetic_denoise_pxl_file", scope="module")
def synthetic_denoise_pxl_file_fixture(tmp_path_factory):
    """A small synthetic denoise pxl file generated on the fly.

    Four independent cell graphs are generated and populated with markers from
    the proxiome-v1 panel (no hashing markers), then written to a pxl file. The
    edge count is tuned so each component stays connected while keeping a sizable
    peripheral one-core layer (~7% of nodes) on top of a denser core, so one-core
    denoising actually removes nodes. Connectivity matters: a disconnected piece
    would be stranded and removed, which could drop core>1 nodes the dataset
    tests expect to be preserved. The fixed seed makes these properties
    deterministic. The same file backs both the in-process denoise tests and the
    CLI tests.

    Args:
        tmp_path_factory: Pytest tmp path factory.
    """
    panel = load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")
    edgelist = generate_edgelist(
        n_cells=4,
        n_nodes=2000,
        n_edges=3900,
        min_neighbors=20,
        panel=panel,
        n_crossing_edges=0,
        rng=0,
    )
    path = tmp_path_factory.mktemp("denoise") / "synthetic_denoise.pxl"
    return write_pna_pxl(edgelist, panel, path, sample_name="PNA055_Sample07_S7")
