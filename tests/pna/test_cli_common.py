"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.cli.common import validate_panel


def test_validate_panel():
    panel = "proxiome-immuno-155"
    assert validate_panel(None, None, panel) == panel


def test_validate_custom_panel(pna_data_root):
    panel_file = pna_data_root / "panels/test-pna-panel.csv"
    assert validate_panel(None, None, panel_file) == panel_file
