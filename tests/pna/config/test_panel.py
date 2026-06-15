"""Copyright © 2025 Pixelgen Technologies AB."""

from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import ruamel.yaml as yaml
from pandas.testing import assert_frame_equal

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import read


@pytest.fixture
def panel_df():
    """Panel df."""
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
    """Verify panel validation.

    Args:
        panel_df: panel df.
    """
    metadata = {
        "name": "test_panel",
        "version": "0.0.0",
        "description": "panel description",
        "aliases": ["test_alias"],
    }
    panel = PNAAntibodyPanel(
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
    assert_frame_equal(panel.df, panel_df)
    assert panel.filename == "test.csv"
    assert panel.size == 3


def test_panel_properties(panel_df):
    """Verify panel properties.

    Args:
        panel_df: panel df.
    """
    panel = PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_validation_fails_on_underscores_in_marker_names(panel_df):
    """Verify panel validation fails on underscores in marker names.

    Args:
        panel_df: panel df.
    """
    panel_df.rename(index={"marker1": "marker_1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain underscores.*Offending values:.*",
    ):
        PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_validation_fails_on_white_space_in_marker_names(panel_df):
    """Verify panel validation fails on white space in marker names.

    Args:
        panel_df: panel df.
    """
    panel_df.rename(index={"marker1": "marker 1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain white-spaces.*Offending values:.*",
    ):
        PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_validation_fails_on_invalid_uniprot_ids(panel_df):
    """Verify panel validation fails on invalid uniprot ids.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = "PAAAAA"

    with pytest.raises(
        AssertionError,
        match=r".*Invalid UniProt IDs found.*Please conform to the naming convention or remove the following IDs:.*",
    ):
        PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_validation_ok_on_concatenated_uniprot_ids(panel_df):
    """Verify panel validation ok on concatenated uniprot ids.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = "P05107;P15391"
    PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_validation_ok_uniprotid_empty(panel_df):
    """Verify panel validation ok uniprotid empty.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = ""
    PNAAntibodyPanel(df=panel_df, metadata=None)


def test_panel_from_pxl(pxl_file):
    """Verify panel from pxl.

    Args:
        pxl_file: pxl file.
    """
    panel = PNAAntibodyPanel.from_pxl_dataset(read(pxl_file))
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
    assert_frame_equal(panel.df, expected_df)


def test_panel_header_trailing_commas_warns_and_recovers(caplog):
    """Verify panel header trailing commas warns and recovers.

    Args:
        caplog: caplog.
    """
    panel_content = """# ---
# name: test-pna-panel,
# product: test-product,
# aliases:
#   - test-pna
# description: Test R&D panel for PNA,
# version: 1.0.0,
# ---
marker_id,control,sequence_1,sequence_2
MarkerA,no,ACTTCCTAGG,ACTTCCTAGG
"""
    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(panel_content)
        tmp_file.flush()

        with caplog.at_level("WARNING"):
            panel = PNAAntibodyPanel.from_csv(tmp_file.name)

    assert panel.name == "test-pna-panel"
    assert panel.version == "1.0.0"
    assert "trailing comma" in caplog.text.lower()


def test_panel_header_non_recoverable_yaml_still_fails():
    """Verify panel header non recoverable yaml still fails."""
    panel_content = """# ---
# name: test panel
# aliases: [test-alias
# version: 0.1.0
# ---
marker_id,control,nuclear,sequence,conj_id
CD45,no,no,TCCCTTGCGATTTAC,test001
"""
    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(panel_content)
        tmp_file.flush()

        with pytest.raises(yaml.YAMLError):
            PNAAntibodyPanel.from_csv(tmp_file.name)
