"""
Tests for the panel.py module

Copyright © 2022 Pixelgen Technologies AB.
"""

from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import ruamel.yaml as yaml

from pixelator.common.config import AntibodyPanel


def test_panel(data_root):
    # Load the default panel
    """Verify panel.

    Args:
        data_root: data root.
    """
    panel = AntibodyPanel.from_csv(str(data_root / "test_panel.csv"))

    # test the size
    assert panel.size == 22

    # test the control antibodies
    assert sorted(panel.markers_control) == sorted(["IgG1ctrl", "IsoT_ctrl"])

    # test the uniqueness
    assert panel.size == panel.df.barcode.nunique()
    assert panel.size == panel.df.sequence.nunique()

    assert panel.version == "0.1.0"
    assert panel.description == "Just a test description"
    assert panel.name == "test panel"


def test_panel_version_metadata(data_root):
    # Load the default panel
    """Verify panel version metadata.

    Args:
        data_root: data root.
    """
    panel = AntibodyPanel.from_csv(str(data_root / "test_panel_v2.csv"))

    # test the size
    assert panel.size == 22

    # test the control antibodies
    assert sorted(panel.markers_control) == sorted(["IgG1ctrl", "IsoT_ctrl"])

    # test the uniqueness
    assert panel.size == panel.df.barcode.nunique()
    assert panel.size == panel.df.sequence.nunique()

    assert panel.version == "0.2.0"
    assert panel.description == "Just a test description"
    assert panel.name == "test panel"


def test_panel_with_non_unique_values(panel: pd.DataFrame):
    """Verify panel with non unique values.

    Args:
        panel: Panel.
    """
    panel_copy = panel.df.copy()
    duplicated_panel = pd.concat([panel_copy, panel.df])

    errors = AntibodyPanel.validate_antibody_panel(duplicated_panel)
    assert len(errors) == 2
    assert set(errors) == {
        "All values in column: marker_id were not unique",
        "All values in column: sequence were not unique",
    }


def test_panel_raise_when_there_is_issue(panel: pd.DataFrame):
    """Verify panel raise when there is issue.

    Args:
        panel: Panel.
    """
    panel_copy = panel.df.copy()
    duplicated_panel = pd.concat([panel_copy, panel.df])

    with NamedTemporaryFile(suffix=".csv") as tmp_file:
        duplicated_panel.to_csv(tmp_file.name)

        with pytest.raises(AssertionError):
            AntibodyPanel.from_csv(tmp_file.name)


def test_panel_header_trailing_commas_warns_and_recovers(caplog):
    """Verify panel header trailing commas warns and recovers.

    Args:
        caplog: caplog.
    """
    panel_content = """# ---
# name: test panel,
# description: test description,
# version: 0.1.0,
# ---
marker_id,control,nuclear,sequence,conj_id
CD45,no,no,TCCCTTGCGATTTAC,test001
"""
    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(panel_content)
        tmp_file.flush()

        with caplog.at_level("WARNING"):
            panel = AntibodyPanel.from_csv(tmp_file.name)

    assert panel.name == "test panel"
    assert panel.version == "0.1.0"
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
            AntibodyPanel.from_csv(tmp_file.name)
