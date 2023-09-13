"""
Tests for the panel.py module

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from pixelator.config import AntibodyPanel


def test_panel(data_root):
    # Load the default panel
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
    panel_copy = panel.df.copy()
    duplicated_panel = pd.concat([panel_copy, panel.df])

    errors = AntibodyPanel.validate_antibody_panel(duplicated_panel)
    assert len(errors) == 2
    assert set(errors) == {
        "All values in column: marker_id were not unique",
        "All values in column: sequence were not unique",
    }


def test_panel_raise_when_there_is_issue(panel: pd.DataFrame):
    panel_copy = panel.df.copy()
    duplicated_panel = pd.concat([panel_copy, panel.df])

    with NamedTemporaryFile(suffix=".csv") as tmp_file:
        duplicated_panel.to_csv(tmp_file.name)

        with pytest.raises(AssertionError):
            AntibodyPanel.from_csv(tmp_file.name)
