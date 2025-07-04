"""Copyright © 2025 Pixelgen Technologies AB."""

import pandas as pd
import pytest

from pixelator.pna.config.panel import PNAAntibodyPanel


def test_panel_validation():
    # all is ok
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
        "conj_id": ["conj1", "conj2", "conj3"],
    }
    df = pd.DataFrame(data)
    PNAAntibodyPanel(df=df, metadata=None)


def test_panel_validation_fails_on_underscores_in_marker_names():
    data = {
        "marker_id": ["marker_1", "marker2", "marker3"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
        "conj_id": ["conj1", "conj2", "conj3"],
    }
    df = pd.DataFrame(data)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain underscores.*Offending values:.*",
    ):
        PNAAntibodyPanel(df=df, metadata=None)
