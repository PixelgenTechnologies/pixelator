"""Shared fixtures for the test data generator tests.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pandas as pd
import polars as pl
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import PNAAntibodyPanel

# Six regular markers and two groups of hashing markers, indexed 1-8 via the
# `-X` suffix (mirroring the real panel naming). A hashing marker is placed
# first so that tests can verify hashing markers are excluded from the abundance
# tiers by mask rather than merely by trailing position.
_REGULAR = ["MarkerA", "MarkerB", "MarkerC", "MarkerD", "MarkerE", "MarkerF"]
_HASHING = [f"{base}-{i}" for base in ("HashA", "HashB") for i in range(1, 9)]
_MARKERS = [(_HASHING[0], "yes")]
_MARKERS += [(marker, "no") for marker in _REGULAR]
_MARKERS += [(marker, "yes") for marker in _HASHING[1:]]


def _dna(i: int, prefix: str) -> str:
    """Build a unique length-10 dna sequence from an integer (3-base-4 suffix)."""
    suffix = "".join("ACGT"[(i >> (2 * k)) & 3] for k in range(3))
    return f"{prefix}{suffix}"


@pytest.fixture(name="marker_panel")
def marker_panel_fixture():
    """A small PNA panel with six regular markers and indexed hashing markers."""
    panel_df = pd.DataFrame(
        {
            "marker_id": [marker for marker, _ in _MARKERS],
            "control": [False] * len(_MARKERS),
            "sequence_1": [_dna(i, "ACGTACG") for i in range(len(_MARKERS))],
            "sequence_2": [_dna(i, "TGCATGC") for i in range(len(_MARKERS))],
            "sample_hashing": [hashing for _, hashing in _MARKERS],
        }
    ).set_index("marker_id")

    return PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(name="test-marker-panel", version="0.1.0"),
    )


@pytest.fixture(name="no_hashing_panel")
def no_hashing_panel_fixture():
    """A small PNA panel with only regular markers and no ``sample_hashing`` column."""
    panel_df = pd.DataFrame(
        {
            "marker_id": _REGULAR,
            "control": [False] * len(_REGULAR),
            "sequence_1": [_dna(i, "ACGTACG") for i in range(len(_REGULAR))],
            "sequence_2": [_dna(i, "TGCATGC") for i in range(len(_REGULAR))],
        }
    ).set_index("marker_id")

    return PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(name="test-no-hashing-panel", version="0.1.0"),
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
