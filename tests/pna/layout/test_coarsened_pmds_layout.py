from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator import read_pna as read
from pixelator.common.graph.backends.implementations._networkx import (
    coarsened_pmds_layout,
)

PXL_FILE = Path(__file__).parents[2] / "pna" / "data" / "PNA055_Sample07_S7.layout.pxl"
COMPONENT_ID = "0a45497c6bfbfb22"


@pytest.fixture(scope="module")
def graph_from_pxl():
    """Load a single graph from the minimal PNA PBMC pxl file."""
    pg_data = read(str(PXL_FILE))
    g = pg_data.filter(components=COMPONENT_ID).edgelist().iterator().__next__().graph
    return g


def test_coarsened_pmds_layout_tp_matches_reference(graph_from_pxl):

    layout_head_ref = pd.DataFrame(
        {
            61208583141770358: np.array([0.1764863, -0.23645603, -0.49442918]),
            50526950249468550: np.array([0.2351768, -0.34720963, -0.44479688]),
            69733109123764664: np.array([-0.99167468, -0.01452478, -0.29084893]),
        }
    )

    g = graph_from_pxl.raw
    layout = coarsened_pmds_layout(g, weight_edges_by="tp", seed=42)
    layout_head = pd.DataFrame(dict(list(layout.items())[:3]))

    assert_frame_equal(layout_head_ref, layout_head)


def test_coarsened_pmds_layout_crossing_edges_matches_reference(graph_from_pxl):

    layout_head_ref = pd.DataFrame(
        {
            61208583141770358: np.array([-0.17772736, 0.59633194, -0.33318281]),
            50526950249468550: np.array([-0.20552174, 0.55308193, -0.42281924]),
            69733109123764664: np.array([0.8492639, 0.24823581, -0.02972813]),
        }
    )

    g = graph_from_pxl.raw
    layout = coarsened_pmds_layout(g, weight_edges_by="crossing_edges", seed=42)
    layout_head = pd.DataFrame(dict(list(layout.items())[:3]))

    assert_frame_equal(layout_head_ref, layout_head)
