"""Copyright © 2025 Pixelgen Technologies AB."""

import click
import pytest

from pixelator.pna.cli.common import validate_panel


def test_validate_panel():
    """Verify validate panel."""
    panel = "proxiome-v1-immuno-155-v1.0"
    assert validate_panel(None, None, panel) == panel


def test_validate_custom_panel(pna_data_root):
    """Verify validate custom panel.

    Args:
        pna_data_root: pna data root.
    """
    panel_file = pna_data_root / "panels/test-pna-panel.csv"
    assert validate_panel(None, None, panel_file) == panel_file


def test_validate_non_existing_panel():
    """Test that validating a non-existing panel raises a UsageError."""
    panel = "non-existing-panel"
    with pytest.raises(click.UsageError):
        validate_panel(None, None, panel)
