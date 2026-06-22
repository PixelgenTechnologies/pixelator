"""Shared fixtures for the test data generator tests.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pandas as pd
import polars as pl
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import PNAAntibodyPanel

# Six regular markers and two hashing ones. A hashing marker is placed first so
# that tests can verify hashing markers are excluded from the abundance tiers by
# mask rather than merely by trailing position.
_MARKERS = [
    ("Hash0", "yes"),
    ("MarkerA", "no"),
    ("MarkerB", "no"),
    ("MarkerC", "no"),
    ("MarkerD", "no"),
    ("MarkerE", "no"),
    ("MarkerF", "no"),
    ("Hash1", "yes"),
]

_BASES = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT"]


@pytest.fixture(name="marker_panel")
def marker_panel_fixture():
    """A small PNA panel with six regular markers and two hashing markers."""
    panel_df = pd.DataFrame(
        {
            "marker_id": [marker for marker, _ in _MARKERS],
            "control": [False] * len(_MARKERS),
            "sequence_1": [f"ACGTACGT{suffix}" for suffix in _BASES],
            "sequence_2": [f"TGCATGCA{suffix}" for suffix in _BASES],
            "sample_hashing": [hashing for _, hashing in _MARKERS],
        }
    ).set_index("marker_id")

    return PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(name="test-marker-panel", version="0.1.0"),
    )


@pytest.fixture(name="assay")
def assay_fixture():
    """The proxiome-v2 assay describing the read structure."""
    return pna_config.get_assay("proxiome-v2")


@pytest.fixture(name="populated_edgelist")
def populated_edgelist_fixture():
    """A small populated edge list with umi/marker columns for both endpoints.

    Markers are taken from the ``marker_panel`` fixture so the panel joins in
    the read generation code resolve to sequences.
    """
    return pl.DataFrame(
        {
            "umi1": [0b00000000, 0b11100100, 1234567890],
            "marker1": ["MarkerA", "MarkerB", "MarkerC"],
            "umi2": [0b11111111, 42, 987654321],
            "marker2": ["MarkerD", "MarkerE", "MarkerF"],
        }
    )
