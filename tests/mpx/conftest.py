"""Configuration and shared files/objects for the testing framework.

Copyright © 2022 Pixelgen Technologies AB.
"""

from pathlib import Path

import pytest
from anndata import AnnData, read_h5ad

DATA_ROOT = Path(__file__).parent / "data"


@pytest.fixture(name="data_root", scope="session")
def data_root_fixture():
    """Return the data root directory."""
    return DATA_ROOT


@pytest.fixture(name="density_scatter_plot_adata")
def density_scatter_plot_adata_fixture(data_root) -> AnnData:
    """Return AnnData used by the density scatter plot image-compare test."""
    return read_h5ad(data_root / "density_scatter_plot_input.h5ad")
