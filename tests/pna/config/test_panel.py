"""Copyright © 2025 Pixelgen Technologies AB."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import (
    PNAAntibodyPanelCombination,
    PartialPNAAntibodyPanel,
)
from pixelator.pna.pixeldataset import read


@pytest.fixture
def panel_df():
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["P61769", "P05107", "P15391"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
    }
    return pd.DataFrame(data).set_index("marker_id")


def test_panel_validation(panel_df):
    # all is ok
    metadata = {
        "name": "test_panel",
        "version": "0.0.0",
        "description": "panel description",
        "aliases": ["test_alias"],
    }
    panel = PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(**metadata),
        file_name="test.csv",
    )

    assert panel.name == metadata["name"]
    assert panel.version == metadata["version"]
    assert panel.description == metadata["description"]
    assert panel.aliases == metadata["aliases"]

    assert panel.markers_control == ["marker2"]
    assert panel.markers == ["marker1", "marker2", "marker3"]
    assert_frame_equal(
        panel.df.drop(columns=["partial_panel_name", "partial_panel_type"]), panel_df
    )
    assert panel.filename == "test.csv"
    assert panel.size == 3


def test_panel_properties(panel_df):
    panel = PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="mock-name",
            version="0.0.0",
        ),
    )


def test_panel_validation_fails_on_underscores_in_marker_names(panel_df):
    panel_df.rename(index={"marker1": "marker_1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain underscores.*Offending values:.*",
    ):
        PNAAntibodyPanelCombination(
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )


def test_panel_validation_fails_on_white_space_in_marker_names(panel_df):
    panel_df.rename(index={"marker1": "marker 1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain white-spaces.*Offending values:.*",
    ):
        PNAAntibodyPanelCombination(
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )


def test_panel_validation_fails_on_invalid_uniprot_ids(panel_df):
    panel_df.loc["marker1", "uniprot_id"] = "PAAAAA"

    with pytest.raises(
        AssertionError,
        match=r".*Invalid UniProt IDs found.*Please conform to the naming convention or remove the following IDs:.*",
    ):
        PNAAntibodyPanelCombination(
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )


def test_panel_validation_ok_on_concatenated_uniprot_ids(panel_df):
    panel_df.loc["marker1", "uniprot_id"] = "P05107;P15391"
    PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="mock-name",
            version="0.0.0",
        ),
    )


def test_panel_validation_ok_uniprotid_empty(panel_df):
    panel_df.loc["marker1", "uniprot_id"] = ""
    PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="mock-name",
            version="0.0.0",
        ),
    )


def test_panel_from_pxl(pxl_file):
    panel = PNAAntibodyPanelCombination.from_pxl_dataset(read(pxl_file))
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.1.0"
    assert panel.description == "Test R&D panel for RNA"
    assert panel.aliases == ["test-pna"]

    expected_data = {
        "marker_id": ["MarkerA", "MarkerB", "MarkerC"],
        "control": [False, False, True],
        "uniprot_id": ["P12345", "P56890;P65470", ""],
        "sequence_1": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
        "sequence_2": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
    }
    expected_df = pd.DataFrame(expected_data).set_index("marker_id")
    assert_frame_equal(
        panel.df.drop(columns=["partial_panel_name", "partial_panel_type"]),
        expected_df,
    )
