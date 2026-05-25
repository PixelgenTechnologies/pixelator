"""Copyright © 2025 Pixelgen Technologies AB."""

from unittest.mock import create_autospec

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pixelator.common.config.panel import AntibodyPanelMetadata
from pixelator.pna.config import PNAAntibodyPanel


@pytest.fixture(name="mock_panel")
def mock_panel_fixture(request):
    """Mock panel fixture.

    Args:
    request: request.

    """
    version = getattr(request, "param", "0.0.0")

    mock_antibody_panel = create_autospec(PNAAntibodyPanel)
    mock_antibody_panel.markers = ["MarkerA", "MarkerB", "MarkerC"]
    # Each marker is duplicated in the panel,
    # on these parameters so this accounts for that.
    mock_antibody_panel.df = pd.DataFrame(
        {
            "marker_id": [
                "MarkerA",
                "MarkerB",
                "MarkerC",
            ],
            "uniprot_id": ["P61769", "P05107", ""],
            "control": [False, False, True],
            "nuclear": ["yes", "no", "no"],
            "sequence_1": ["AAAA", "CCCC", "GGGG"],
            "sequence_2": ["TTTT", "AAAA", "CCCC"],
        }
    )
    mock_antibody_panel.df.index = mock_antibody_panel.df.marker_id

    if version.startswith("2"):
        mock_antibody_panel.df.drop(columns=["nuclear"], inplace=True)

    mock_antibody_panel.metadata = AntibodyPanelMetadata.model_validate(
        {
            "name": "mock-panel",
            "version": version,
            "description": "Dummy panel data",
            "aliases": ["mock_alias"],
        }
    )
    mock_antibody_panel.name = mock_antibody_panel.metadata.name
    mock_antibody_panel.version = mock_antibody_panel.metadata.version
    mock_antibody_panel.aliases = mock_antibody_panel.metadata.aliases
    mock_antibody_panel.description = mock_antibody_panel.metadata.description
    return mock_antibody_panel


@pytest.fixture(name="edgelist_karate_graph")
def edgelist_karate_graph_fixture():
    """Edgelist karate graph fixture."""
    rng = np.random.default_rng(0)

    markers = ["MarkerA", "MarkerB", "MarkerC"]
    multiple_graph = nx.karate_club_graph()
    nodes_to_marker_dict = {
        node: rng.choice(markers) for node in multiple_graph.nodes()
    }
    edges = [
        {
            "umi1": edge[0],
            "umi2": edge[1],
            "read_count": 10,
            "uei_count": 2,
        }
        for edge in list(nx.to_edgelist(multiple_graph))
    ]
    for edge in edges:
        edge.update(
            {
                "marker_1": nodes_to_marker_dict[edge["umi1"]],
                "marker_2": nodes_to_marker_dict[edge["umi2"]],
            }
        )
        # Make the graph bipartite
        edge.update({"umi2": edge["umi2"] + len(multiple_graph.nodes())})

    return edges
