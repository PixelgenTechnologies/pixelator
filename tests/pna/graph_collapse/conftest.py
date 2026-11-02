"""Fixtures for the graph-collapse tests.

Copyright © 2026 Pixelgen Technologies AB.
"""

import shutil
from pathlib import Path

import pytest

from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.pixeldataset.io import PixelFileWriter

# The panel the layout fixture was generated with.
LAYOUT_FIXTURE_PANEL = "proxiome-v1-immuno-155-v1.0"


@pytest.fixture(scope="module")
def graph_pxl_file(pna_pxl_file, tmp_path_factory) -> Path:
    """A real pxl file that stores its panel, as files from the graph step do.

    The ``layout.pxl`` fixture only records a panel name, which graph-collapse
    does not use, so the panel is written into a copy of it.
    """
    path = tmp_path_factory.mktemp("graph_collapse") / "PNA055_Sample07_S7.pxl"
    shutil.copy(pna_pxl_file, path)
    with PixelFileWriter(path) as writer:
        writer.write_panel(load_antibody_panel(pna_config, LAYOUT_FIXTURE_PANEL))
    return path
