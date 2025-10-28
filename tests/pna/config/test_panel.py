"""Copyright © 2025 Pixelgen Technologies AB."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.config.panel import PNAAntibodyPanel


def test_panel_validation():
    # all is ok
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["P61769", "P05107", "P15391"],
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
        "uniprot_id": ["P61769", "P05107", "P15391"],
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


def test_panel_validation_fails_on_invalid_uniprot_ids():
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["PAAAAA", "P05107", "P15391"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
        "conj_id": ["conj1", "conj2", "conj3"],
    }
    df = pd.DataFrame(data)

    with pytest.raises(
        AssertionError,
        match=r".*Invalid UniProt IDs found.*Please conform to the naming convention or remove the following IDs:.*",
    ):
        PNAAntibodyPanel(df=df, metadata=None)


def test_panel_validation_ok_on_concatenated_uniprot_ids():
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["P05107;P15391", "P05107", "P15391"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
        "conj_id": ["conj1", "conj2", "conj3"],
    }
    df = pd.DataFrame(data)
    PNAAntibodyPanel(df=df, metadata=None)


def test_panel_validation_ok_uniprotid_empty():
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["P05107", "P05107", ""],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
        "conj_id": ["conj1", "conj2", "conj3"],
    }
    df = pd.DataFrame(data)
    PNAAntibodyPanel(df=df, metadata=None)


def test_panel_from_pxl(pxl_file):
    panel = PNAAntibodyPanel.from_pxl(pxl_file)
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.1.0"
    assert panel.description == "Test R&D panel for RNA"
    assert panel.aliases == ["test-pna"]

    expected_data = {
        "marker_id": ["MarkerA", "MarkerB", "MarkerC"],
        "control": ["no", "no", "yes"],
        "nuclear": ["yes", "no", "no"],
        "uniprot_id": ["P12345", "P56890;P65470", ""],
        "sequence_1": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
        "conj_id": ["pna_rnd01", "pna_rnd02", "pna_rnd03"],
        "sequence_2": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df.index = expected_df.conj_id
    assert_frame_equal(panel.df, expected_df)
