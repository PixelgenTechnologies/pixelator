"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path
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
    assert panel.filepath is None
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
    assert panel.filepath == Path(tmp_file.name).resolve()
    assert "trailing comma" in caplog.text.lower()


def test_panel_header_multiple_trailing_commas_warns_and_recovers(caplog):
    """Verify panel header with multiple trailing commas per line warns and recovers.

    This reproduces the pattern left behind when a spreadsheet application
    pads every row (including the YAML front-matter comment lines) to a
    fixed column count on save.

    Args:
        caplog: caplog.
    """
    panel_content = """# ---,,,,,,,
# name: test-pna-panel,,,,,,,
# product: test-product,,,,,,,
# description: Test R&D panel for PNA,,,,,,,
# version: 1.0.0,,,,,,,
# ---,,,,,,,
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
    assert panel.filepath == Path(tmp_file.name).resolve()
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


def _marker_row(marker_id: str, sequence: str, **extra) -> dict:
    row = {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
    }
    row.update(extra)
    return row


def test_duplicate_marker_ids_are_rejected():
    """Marker ids must be unique even when sequences differ."""
    frame = pd.DataFrame(
        [_marker_row("CD3", "AAAA"), _marker_row("CD3", "CCCC")]
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="marker_id were not unique"):
        PNAAntibodyPanel(
            frame,
            AntibodyPanelMetadata(name="base", version="1.0.0"),
        )


def test_hashing_marker_ids_need_a_numeric_suffix_and_must_not_nest():
    """Hashing ids end with -<digits> and must not collapse onto each other."""
    missing_suffix = pd.DataFrame(
        [_marker_row("B2M", "AAAA", sample_hashing=True)]
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="must end with -<digits>"):
        PNAAntibodyPanel(
            missing_suffix,
            AntibodyPanelMetadata(name="base", version="1.0.0"),
        )

    nested = pd.DataFrame(
        [
            _marker_row("B2M-1", "AAAA", sample_hashing=True),
            _marker_row("B2M-1-1", "CCCC", sample_hashing=True),
        ]
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="must not collapse"):
        PNAAntibodyPanel(
            nested,
            AntibodyPanelMetadata(name="base", version="1.0.0"),
        )


def test_panel_validation_reports_missing_required_columns(panel_df):
    """A panel missing a required column is rejected before later checks."""
    errors = PNAAntibodyPanel.validate_antibody_panel(
        panel_df.drop(columns=["sequence_1"])
    )
    assert len(errors) == 1
    assert "missing required columns" in errors[0]
    assert "sequence_1" in errors[0]


def test_panel_validation_reports_an_empty_panel(panel_df):
    """An empty panel is rejected."""
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df.iloc[0:0])
    assert "Panel file is empty" in errors


def test_panel_validation_requires_marker_id_index(panel_df):
    """The marker id must be the dataframe index."""
    panel_df.index.name = "other"
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df, validate_types=False)
    assert "`marker_id` is missing or is not set as index" in errors


def test_panel_validation_requires_unique_sequences(panel_df):
    """Sequence columns must contain unique values."""
    panel_df.loc["marker2", "sequence_1"] = panel_df.loc["marker1", "sequence_1"]
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df)
    assert "All values in column: sequence_1 were not unique" in errors


def test_panel_validation_requires_boolean_control(panel_df):
    """The control column must be boolean once types are not re-checked."""
    panel_df["control"] = ["no", "yes", "no"]
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df, validate_types=False)
    assert "`control` column is not boolean" in errors


def test_panel_validation_reports_column_type_mismatch(panel_df):
    """Column types are checked when validate_types is left on."""
    panel_df["control"] = ["no", "yes", "no"]
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df)
    assert any(
        error.startswith("Column control has incorrect type.") for error in errors
    )
